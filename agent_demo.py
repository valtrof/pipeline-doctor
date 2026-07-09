"""Smoke run for the v2 remediation agent (DESIGN.md milestone 3).

Runs the real agent loop against the Claude API and saves the full report,
including the turn-by-turn transcript, to demo_run.json.

Default mode needs no GCP — it builds a synthetic DataFrame with seeded anomalies:

    python agent_demo.py

To run against a BigQuery public dataset instead (needs GCP_PROJECT_ID + gcloud auth):

    python agent_demo.py --dataset bigquery-public-data.new_york_taxi_trips.tlc_yellow_trips_2022 --limit 1000
"""

import argparse
import json

import numpy as np
import pandas as pd

from agent import run_remediation
from anomaly_detector import get_client


def synthetic_df(rows: int = 200) -> pd.DataFrame:
    """Deterministic synthetic dataset with seeded null, zero, and outlier anomalies."""
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "trip_miles": rng.uniform(0.5, 15.0, rows).round(2),
        "fare": rng.uniform(5.0, 60.0, rows).round(2),
        "passenger_count": rng.integers(1, 5, rows).astype(float),
        "tip": rng.uniform(0.0, 12.0, rows).round(2) + 0.01,  # keep clean of zeros
    })
    df.loc[rng.choice(rows, 20, replace=False), "passenger_count"] = None  # nulls
    df.loc[rng.choice(rows, 8, replace=False), "trip_miles"] = 0.0         # zeros
    df.loc[rng.choice(rows, 3, replace=False), "fare"] = 9500.0            # outliers
    return df


def fetch_bigquery_df(dataset: str, limit: int) -> pd.DataFrame:
    import os
    from google.cloud import bigquery

    bq_client = bigquery.Client(project=os.environ["GCP_PROJECT_ID"])
    return bq_client.query(f"SELECT * FROM `{dataset}` LIMIT {limit}").to_dataframe()


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke run of the remediation agent.")
    parser.add_argument("--dataset", help="BigQuery table (project.dataset.table). Omit for synthetic data.")
    parser.add_argument("--limit", type=int, default=1000, help="Row sample size for BigQuery mode.")
    parser.add_argument("--out", default="demo_run.json", help="Where to save the full report.")
    args = parser.parse_args()

    if args.dataset:
        name, df = args.dataset, fetch_bigquery_df(args.dataset, args.limit)
    else:
        name, df = "synthetic-taxi-demo", synthetic_df()

    print(f"Dataset: {name} ({len(df)} rows)")
    report = run_remediation(df, name, get_client())

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\nStatus: {report['status']}")
    print(f"Summary: {report['summary']}")
    print(f"Turns used: {report['turns_used']}")
    print(f"Verified fixes: {report['verified_fix_count']} of {len(report['baseline'])} baseline anomalies")
    print(f"Rows: {report['rows_original']} -> {report['rows_final']}")
    if report["remaining"]:
        print("Remaining anomalies:")
        for rec in report["remaining"]:
            print(f"  - {rec['column']}: {rec['kind']} (count={rec['count']})")
    else:
        print("Remaining anomalies: none")
    print(f"\nFull report with transcript saved to {args.out}")


if __name__ == "__main__":
    main()
