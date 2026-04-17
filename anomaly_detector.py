import logging
import os

import pandas as pd
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MODEL_NAME = "gpt-4o-mini"

DIAGNOSIS_PROMPT = """
You are a senior data engineer reviewing a data quality report.
Dataset: {dataset_name}
Anomalies detected:
{anomalies}
For each anomaly: explain the likely cause and recommend how the pipeline should handle it.
Be concise and technical.
"""

FIX_PROMPT = """
You are a senior data engineer. Given the following anomalies in a pandas DataFrame, \
return concrete Python/pandas fix code for each one.

Anomalies:
{anomalies}

Rules:
- Return only a numbered list of fixes. No prose.
- Each fix must be a single executable pandas line using `df` as the variable name.
- Example format:
  1. passenger_count nulls → df['passenger_count'].fillna(df['passenger_count'].median(), inplace=True)
  2. trip_miles zeros → df = df[df['trip_miles'] > 0]
"""


def get_llm() -> ChatOpenAI:
    """Return a configured ChatOpenAI instance using the OPENAI_API_KEY environment variable."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set. Add it to your .env file.")
    return ChatOpenAI(model=MODEL_NAME, api_key=api_key)


def detect_outliers(df: pd.DataFrame) -> str:
    """Scan numeric columns for outliers using the IQR method.

    For each numeric column, flag values outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR].
    Returns a formatted report string.
    """
    report = []

    for col in df.select_dtypes(include="number").columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        # Skip if IQR is zero (all values are the same)
        if IQR == 0:
            continue

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        outlier_count = len(outliers)

        if outlier_count > 0:
            pct = round(outlier_count / len(df) * 100, 1)
            report.append(f"- {col}: {outlier_count} outliers ({pct}%) — IQR method")
            logger.warning("Outlier anomaly: %s — %d outliers (%.1f%%)", col, outlier_count, pct)

    return "\n".join(report)


def detect_anomalies(df: pd.DataFrame) -> str:
    """Scan a DataFrame for nulls, zero values, and outliers; return a formatted anomaly report."""
    report = []

    nulls = df.isnull().sum()
    for col, count in nulls[nulls > 0].items():
        pct = round(count / len(df) * 100, 1)
        report.append(f"- {col}: {count} nulls ({pct}%)")
        logger.warning("Null anomaly: %s — %d nulls (%.1f%%)", col, count, pct)

    for col in df.select_dtypes(include="number").columns:
        zeros = (df[col] == 0).sum()
        if zeros > 0:
            report.append(f"- {col}: {zeros} zero values")
            logger.warning("Zero value anomaly: %s — %d zeros", col, zeros)

    outlier_report = detect_outliers(df)
    if outlier_report:
        report.append(outlier_report)

    if not report:
        logger.info("No anomalies detected.")

    return "\n".join(report)


def suggest_fixes(anomalies: str, llm: ChatOpenAI) -> str:
    """Return concrete pandas fix code for each anomaly in the report.

    Args:
        anomalies: The anomaly report string produced by detect_anomalies().
        llm: A configured ChatOpenAI instance.

    Returns:
        A numbered list of executable pandas fix statements.
    """
    if not anomalies:
        return "No fixes needed."

    try:
        prompt = ChatPromptTemplate.from_template(FIX_PROMPT)
        chain = prompt | llm
        response = chain.invoke({"anomalies": anomalies})
        logger.info("Fix suggestions generated.")
        return response.content
    except Exception as e:
        logger.error("Fix suggestion failed: %s", e)
        raise


def diagnose(df: pd.DataFrame, dataset_name: str, llm: ChatOpenAI) -> tuple[str, str]:
    """Detect anomalies in df and return an LLM-generated diagnosis.

    Args:
        df: The dataset to analyse.
        dataset_name: Human-readable name used in the LLM prompt and logs.
        llm: A configured ChatOpenAI instance.

    Returns:
        A tuple of (anomaly_report, llm_diagnosis).
    """
    logger.info("Starting diagnosis for dataset: %s", dataset_name)
    anomalies = detect_anomalies(df)

    if not anomalies:
        logger.info("No anomalies to diagnose for %s.", dataset_name)
        return anomalies, "No anomalies detected."

    try:
        prompt = ChatPromptTemplate.from_template(DIAGNOSIS_PROMPT)
        chain = prompt | llm
        response = chain.invoke({"anomalies": anomalies, "dataset_name": dataset_name})
        logger.info("Diagnosis complete for dataset: %s", dataset_name)
        return anomalies, response.content
    except Exception as e:
        logger.error("LLM diagnosis failed for %s: %s", dataset_name, e)
        raise
