import pandas as pd
import pytest

from anomaly_detector import profile_anomalies
from evals.seed_anomalies import build_cases, make_clean_df, score_case


# ---------------------------------------------------------------------------
# seeding
# ---------------------------------------------------------------------------

def test_clean_base_frame_has_no_anomalies():
    assert profile_anomalies(make_clean_df()) == []


def test_cases_are_deterministic():
    first, second = build_cases(), build_cases()
    for a, b in zip(first, second):
        assert a["name"] == b["name"]
        assert a["ground_truth"] == b["ground_truth"]
        pd.testing.assert_frame_equal(a["df"], b["df"])


def test_every_seeded_anomaly_is_detectable():
    """Ground truth must be real: detection must find every seeded anomaly."""
    for case in build_cases():
        detected = {(r["column"], r["kind"]) for r in profile_anomalies(case["df"])}
        for gt in case["ground_truth"]:
            assert (gt["column"], gt["kind"]) in detected, f"{case['name']}: {gt} not detected"


def test_heavy_nulls_case_would_violate_retention_if_dropped():
    case = next(c for c in build_cases() if c["name"] == "heavy_nulls")
    null_share = case["df"]["tip"].isnull().mean()
    assert null_share > 0.2  # dropping null rows would breach the 0.8 budget


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------

GT = [{"column": "a", "kind": "null"}]


def make_report(remaining, fixes, rows_final=100, turns=5, status="completed"):
    return {
        "remaining": remaining,
        "fixes": fixes,
        "rows_original": 100,
        "rows_final": rows_final,
        "turns_used": turns,
        "status": status,
    }


def passing_fix(column="a", kind="null"):
    return {"target": {"column": column, "kind": kind}, "verdict": {"passed": True}}


def test_score_counts_verified_fix():
    report = make_report(remaining=[], fixes=[passing_fix()])
    score = score_case(report, GT)
    assert score == {
        "seeded": 1, "resolved": 1, "verified": 1, "false_fixes": 0, "gave_up": 0,
        "collateral_anomalies": [], "rows_retained": 1.0, "attempts": 1, "turns": 5,
        "status": "completed",
    }


def test_score_counts_give_up():
    report = make_report(remaining=[{"column": "a", "kind": "null"}], fixes=[])
    score = score_case(report, GT)
    assert score["gave_up"] == 1
    assert score["resolved"] == 0
    assert score["false_fixes"] == 0


def test_score_flags_false_fix_when_resolved_without_passing_verdict():
    # anomaly gone from `remaining` but no fix ever passed verification
    report = make_report(remaining=[], fixes=[{"target": {"column": "a", "kind": "null"}, "verdict": {"passed": False}}])
    score = score_case(report, GT)
    assert score["resolved"] == 1
    assert score["verified"] == 0
    assert score["false_fixes"] == 1


def test_score_flags_false_fix_on_row_loss():
    report = make_report(remaining=[], fixes=[passing_fix()], rows_final=50)
    score = score_case(report, GT)
    assert score["rows_retained"] == 0.5
    assert score["false_fixes"] == 1


def test_score_reports_collateral_anomalies():
    report = make_report(
        remaining=[{"column": "b", "kind": "zero"}],  # never seeded
        fixes=[passing_fix()],
    )
    score = score_case(report, GT)
    assert score["collateral_anomalies"] == [{"column": "b", "kind": "zero"}]
    assert score["false_fixes"] == 1
