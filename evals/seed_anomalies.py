"""Seeded-anomaly benchmark for the remediation agent (DESIGN.md milestone 5).

Known defects are injected into a clean DataFrame, so ground truth is known by
construction — no LLM judge needed. Each case returns the seeded frame plus the
exact list of anomalies the agent is expected to resolve.
"""

import zlib

import numpy as np
import pandas as pd


def make_clean_df(rows: int = 200, seed: int = 42) -> pd.DataFrame:
    """Deterministic DataFrame with no nulls, zeros, or IQR outliers."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "trip_miles": rng.uniform(0.5, 15.0, rows).round(2),
        "fare": rng.uniform(5.0, 60.0, rows).round(2),
        "passenger_count": rng.integers(1, 5, rows).astype(float),
        "tip": rng.uniform(0.5, 12.0, rows).round(2),
    })


def seed_nulls(df: pd.DataFrame, column: str, count: int, rng) -> dict:
    df.loc[rng.choice(len(df), count, replace=False), column] = None
    return {"column": column, "kind": "null"}


def seed_zeros(df: pd.DataFrame, column: str, count: int, rng) -> dict:
    df.loc[rng.choice(len(df), count, replace=False), column] = 0.0
    return {"column": column, "kind": "zero"}


def seed_outliers(df: pd.DataFrame, column: str, count: int, rng, value: float = 9500.0) -> dict:
    df.loc[rng.choice(len(df), count, replace=False), column] = value
    return {"column": column, "kind": "outlier"}


def build_cases(rows: int = 200) -> list[dict]:
    """The benchmark: each case is {name, df, ground_truth: [{column, kind}, ...]}."""
    cases = []

    def new_case(name, seeders):
        # zlib.crc32, not hash(): builtin str hash is salted per process, which
        # would silently make the benchmark non-reproducible across runs
        rng = np.random.default_rng(zlib.crc32(name.encode()))
        df = make_clean_df(rows)
        ground_truth = [seeder(df, rng) for seeder in seeders]
        cases.append({"name": name, "df": df, "ground_truth": ground_truth})

    new_case("nulls_single", [lambda df, rng: seed_nulls(df, "passenger_count", 15, rng)])
    new_case("zeros_single", [lambda df, rng: seed_zeros(df, "trip_miles", 10, rng)])
    new_case("outliers_single", [lambda df, rng: seed_outliers(df, "fare", 4, rng)])
    new_case("mixed_three_kinds", [
        lambda df, rng: seed_nulls(df, "passenger_count", 20, rng),
        lambda df, rng: seed_zeros(df, "trip_miles", 8, rng),
        lambda df, rng: seed_outliers(df, "fare", 3, rng),
    ])
    # 40% nulls: dropping rows would violate the retention budget — imputation required
    new_case("heavy_nulls", [lambda df, rng: seed_nulls(df, "tip", 80, rng)])

    return cases


def score_case(report: dict, ground_truth: list[dict]) -> dict:
    """Score one agent run against the seeded ground truth. Pure function, no LLM.

    - resolved: seeded anomaly absent from the final `remaining` list
    - verified: a fix for it passed sandbox verification
    - false_fix: resolved without a passing verdict, or with collateral damage
      (row retention below 0.8 or unseeded anomalies present at the end)
    """
    remaining = {(r["column"], r["kind"]) for r in report["remaining"]}
    seeded = {(g["column"], g["kind"]) for g in ground_truth}
    verified = {
        (f["target"]["column"], f["target"]["kind"])
        for f in report["fixes"]
        if f["verdict"] and f["verdict"]["passed"]
    }

    rows_retained = report["rows_final"] / report["rows_original"] if report["rows_original"] else 1.0
    collateral = sorted(remaining - seeded)  # anomalies at the end that were never seeded
    resolved = {g for g in seeded if g not in remaining}

    # a resolved anomaly counts as a false fix if it lacks a passing verdict, or if
    # the run as a whole did collateral damage (row loss / new anomalies)
    run_damaged = bool(collateral) or rows_retained < 0.8
    false_fixes = {g for g in resolved if g not in verified or run_damaged}

    return {
        "seeded": len(seeded),
        "resolved": len(resolved),
        "verified": len(resolved & verified),
        "false_fixes": len(false_fixes),
        "gave_up": len(seeded & remaining),
        "collateral_anomalies": [{"column": c, "kind": k} for c, k in collateral],
        "rows_retained": round(rows_retained, 3),
        "attempts": len(report["fixes"]),
        "turns": report["turns_used"],
        "status": report["status"],
    }
