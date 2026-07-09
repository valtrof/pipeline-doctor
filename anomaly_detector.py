import logging
import os

import anthropic
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MODEL = "claude-haiku-4-5-20251001"

DIAGNOSIS_PROMPT = """\
You are a senior data engineer reviewing a data quality report.
Dataset: {dataset_name}
Anomalies detected:
{anomalies}
For each anomaly: explain the likely cause and recommend how the pipeline should handle it.
Be concise and technical."""

FIX_PROMPT = """\
You are a senior data engineer. Given the following anomalies in a pandas DataFrame, \
return concrete Python/pandas fix code for each one.

Anomalies:
{anomalies}

Rules:
- Return only a numbered list of fixes. No prose.
- Each fix must be a single executable pandas line using `df` as the variable name.
- Example format:
  1. passenger_count nulls → df['passenger_count'].fillna(df['passenger_count'].median(), inplace=True)
  2. trip_miles zeros → df = df[df['trip_miles'] > 0]\
"""


def get_client() -> anthropic.Anthropic:
    """Return a configured Anthropic client using the ANTHROPIC_API_KEY environment variable."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY is not set. Add it to your .env file.")
    return anthropic.Anthropic(api_key=api_key)


def _call_claude(client: anthropic.Anthropic, prompt: str) -> str:
    """Send a single-turn message to Claude and return the text response."""
    response = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


def profile_anomalies(df: pd.DataFrame) -> list[dict]:
    """Scan a DataFrame for nulls, zero values, and IQR outliers; return structured records.

    Each record has the shape {"column", "kind": "null" | "zero" | "outlier", "count", "pct"}.
    This is the machine-readable source of truth that both the formatted reports and the
    agent sandbox verification (sandbox.py) are built on.
    """
    records = []

    nulls = df.isnull().sum()
    for col, count in nulls[nulls > 0].items():
        # float() so records stay JSON-serializable (numpy float64 otherwise)
        pct = float(round(count / len(df) * 100, 1))
        records.append({"column": col, "kind": "null", "count": int(count), "pct": pct})

    numeric_cols = df.select_dtypes(include="number").columns

    for col in numeric_cols:
        zeros = int((df[col] == 0).sum())
        if zeros > 0:
            pct = round(zeros / len(df) * 100, 1)
            records.append({"column": col, "kind": "zero", "count": zeros, "pct": pct})

    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        if IQR == 0:
            continue

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        outlier_count = len(outliers)

        if outlier_count > 0:
            pct = round(outlier_count / len(df) * 100, 1)
            records.append({"column": col, "kind": "outlier", "count": outlier_count, "pct": pct})

    return records


def _report_line(rec: dict) -> str:
    """Format one anomaly record as a report line."""
    if rec["kind"] == "null":
        return f"- {rec['column']}: {rec['count']} nulls ({rec['pct']}%)"
    if rec["kind"] == "zero":
        return f"- {rec['column']}: {rec['count']} zero values"
    return f"- {rec['column']}: {rec['count']} outliers ({rec['pct']}%) — IQR method"


def _log_anomaly(rec: dict) -> None:
    """Emit the warning log entry for one anomaly record."""
    if rec["kind"] == "null":
        logger.warning("Null anomaly: %s — %d nulls (%.1f%%)", rec["column"], rec["count"], rec["pct"])
    elif rec["kind"] == "zero":
        logger.warning("Zero value anomaly: %s — %d zeros", rec["column"], rec["count"])
    else:
        logger.warning("Outlier anomaly: %s — %d outliers (%.1f%%)", rec["column"], rec["count"], rec["pct"])


def detect_outliers(df: pd.DataFrame) -> str:
    """Scan numeric columns for outliers using the IQR method.

    For each numeric column, flag values outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR].
    Returns a formatted report string.
    """
    report = []
    for rec in profile_anomalies(df):
        if rec["kind"] == "outlier":
            report.append(_report_line(rec))
            _log_anomaly(rec)
    return "\n".join(report)


def detect_anomalies(df: pd.DataFrame) -> str:
    """Scan a DataFrame for nulls, zero values, and outliers; return a formatted anomaly report."""
    report = []
    for rec in profile_anomalies(df):
        report.append(_report_line(rec))
        _log_anomaly(rec)

    if not report:
        logger.info("No anomalies detected.")

    return "\n".join(report)


def suggest_fixes(anomalies: str, client: anthropic.Anthropic) -> str:
    """Return concrete pandas fix code for each anomaly in the report."""
    if not anomalies:
        return "No fixes needed."

    try:
        result = _call_claude(client, FIX_PROMPT.format(anomalies=anomalies))
        logger.info("Fix suggestions generated.")
        return result
    except Exception as e:
        logger.error("Fix suggestion failed: %s", e)
        raise


def diagnose(df: pd.DataFrame, dataset_name: str, client: anthropic.Anthropic) -> tuple[str, str]:
    """Detect anomalies in df and return a Claude-generated diagnosis.

    Returns:
        A tuple of (anomaly_report, diagnosis).
    """
    logger.info("Starting diagnosis for dataset: %s", dataset_name)
    anomalies = detect_anomalies(df)

    if not anomalies:
        logger.info("No anomalies to diagnose for %s.", dataset_name)
        return anomalies, "No anomalies detected."

    try:
        prompt = DIAGNOSIS_PROMPT.format(anomalies=anomalies, dataset_name=dataset_name)
        result = _call_claude(client, prompt)
        logger.info("Diagnosis complete for dataset: %s", dataset_name)
        return anomalies, result
    except Exception as e:
        logger.error("LLM diagnosis failed for %s: %s", dataset_name, e)
        raise
