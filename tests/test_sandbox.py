import pandas as pd
import pytest

from anomaly_detector import profile_anomalies
from sandbox import FixExecutionError, FixRejected, Sandbox, check_fix_code


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def df_with_nulls():
    # one null anomaly in `a`; `b` is clean (no zeros, no outliers)
    return pd.DataFrame({"a": [1.0, None, 2.0, 1.5], "b": [1, 2, 3, 4]})


@pytest.fixture
def df_mostly_null():
    # 3 of 5 rows null in `a` — dropping them violates the row-retention budget
    return pd.DataFrame({"a": [1.0, None, None, None, 2.0]})


# ---------------------------------------------------------------------------
# profile_anomalies (structured detection)
# ---------------------------------------------------------------------------

def test_profile_anomalies_returns_structured_null_record():
    df = pd.DataFrame({"a": [None, None, 1, 1]})
    records = profile_anomalies(df)
    assert {"column": "a", "kind": "null", "count": 2, "pct": 50.0} in records


def test_profile_anomalies_finds_zeros_and_outliers():
    df = pd.DataFrame({"a": [0, 1, 2, 1, 2, 999]})
    kinds = {(r["column"], r["kind"]) for r in profile_anomalies(df)}
    assert ("a", "zero") in kinds
    assert ("a", "outlier") in kinds


def test_profile_anomalies_empty_for_clean_data():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    assert profile_anomalies(df) == []


# ---------------------------------------------------------------------------
# check_fix_code (static safety checks)
# ---------------------------------------------------------------------------

def test_check_fix_code_accepts_typical_fixes():
    check_fix_code("df['a'].fillna(df['a'].median(), inplace=True)")
    check_fix_code("df = df[df['a'] > 0]")
    check_fix_code("df['a'] = df['a'].replace(0, df['a'].median())")


@pytest.mark.parametrize(
    "code",
    [
        "import os",
        "from pathlib import Path",
        "open('secrets.txt')",
        "eval('1+1')",
        "df.__class__.__init__",
        "getattr(df, 'to_csv')",
        "while True: pass",
        "pd.read_csv('data.csv')",
        "df.to_csv('out.csv')",
        "_secret = 1",
        "df['a'] = df['a']]",  # syntax error
    ],
)
def test_check_fix_code_rejects_unsafe_code(code):
    with pytest.raises(FixRejected):
        check_fix_code(code)


# ---------------------------------------------------------------------------
# Sandbox.apply
# ---------------------------------------------------------------------------

def test_apply_fix_resolves_nulls_in_sandbox(df_with_nulls):
    sandbox = Sandbox(df_with_nulls)
    result = sandbox.apply("df['a'] = df['a'].fillna(df['a'].median())")
    assert result == {"rows_before": 4, "rows_after": 4}
    assert sandbox.df["a"].isnull().sum() == 0


def test_apply_never_mutates_the_original(df_with_nulls):
    sandbox = Sandbox(df_with_nulls)
    sandbox.apply("df['a'] = df['a'].fillna(0)")
    assert df_with_nulls["a"].isnull().sum() == 1


def test_apply_supports_rebinding_df(df_with_nulls):
    sandbox = Sandbox(df_with_nulls)
    result = sandbox.apply("df = df[df['a'].notna()]")
    assert result == {"rows_before": 4, "rows_after": 3}
    assert len(sandbox.df) == 3


def test_apply_raises_on_runtime_error(df_with_nulls):
    sandbox = Sandbox(df_with_nulls)
    with pytest.raises(FixExecutionError, match="raised during execution"):
        sandbox.apply("df['nonexistent'] = df['nonexistent'].fillna(0)")


def test_failed_apply_leaves_sandbox_unchanged(df_with_nulls):
    sandbox = Sandbox(df_with_nulls)
    before = sandbox.df.copy()
    with pytest.raises(FixExecutionError):
        sandbox.apply("df['nonexistent'] = df['nonexistent'].fillna(0)")
    pd.testing.assert_frame_equal(sandbox.df, before)


def test_apply_raises_when_df_becomes_non_dataframe(df_with_nulls):
    sandbox = Sandbox(df_with_nulls)
    with pytest.raises(FixExecutionError, match="no longer a pandas DataFrame"):
        sandbox.apply("df = 42")


def test_apply_rejected_code_is_never_executed(df_with_nulls):
    sandbox = Sandbox(df_with_nulls)
    before = sandbox.df.copy()
    with pytest.raises(FixRejected):
        sandbox.apply("import os\ndf = df")
    pd.testing.assert_frame_equal(sandbox.df, before)


def test_fixes_accumulate_across_applies():
    df = pd.DataFrame({"a": [0, 1, 2, 3], "b": [1.0, None, 2.0, 1.5]})
    sandbox = Sandbox(df)
    sandbox.apply("df['a'] = df['a'].replace(0, 1)")
    sandbox.apply("df['b'] = df['b'].fillna(df['b'].median())")
    assert (sandbox.df["a"] == 0).sum() == 0
    assert sandbox.df["b"].isnull().sum() == 0


# ---------------------------------------------------------------------------
# Sandbox.reset
# ---------------------------------------------------------------------------

def test_reset_restores_original_state(df_with_nulls):
    sandbox = Sandbox(df_with_nulls)
    sandbox.apply("df = df[df['a'].notna()]")
    sandbox.reset()
    pd.testing.assert_frame_equal(sandbox.df, df_with_nulls)


# ---------------------------------------------------------------------------
# Sandbox.verify
# ---------------------------------------------------------------------------

def test_verify_passes_for_a_good_fix(df_with_nulls):
    sandbox = Sandbox(df_with_nulls)
    sandbox.apply("df['a'] = df['a'].fillna(df['a'].median())")
    verdict = sandbox.verify("a", "null")
    assert verdict["passed"] is True
    assert verdict["resolved"] is True
    assert verdict["regression_free"] is True
    assert verdict["rows_ok"] is True
    assert verdict["new_anomalies"] == []


def test_verify_fails_without_a_fix(df_with_nulls):
    sandbox = Sandbox(df_with_nulls)
    verdict = sandbox.verify("a", "null")
    assert verdict["resolved"] is False
    assert verdict["passed"] is False


def test_verify_catches_degenerate_row_deletion(df_mostly_null):
    # dropping the null rows "resolves" the anomaly but discards 60% of the data
    sandbox = Sandbox(df_mostly_null)
    sandbox.apply("df = df[df['a'].notna()]")
    verdict = sandbox.verify("a", "null")
    assert verdict["resolved"] is True
    assert verdict["rows_retained"] == 0.4
    assert verdict["rows_ok"] is False
    assert verdict["passed"] is False


def test_verify_catches_regressions_in_clean_columns(df_with_nulls):
    # this "fix" introduces zeros into the previously clean column b
    sandbox = Sandbox(df_with_nulls)
    sandbox.apply("df['b'] = df['b'].replace(1, 0)")
    verdict = sandbox.verify("a", "null")
    assert verdict["regression_free"] is False
    assert {"column": "b", "kind": "zero"} in verdict["new_anomalies"]
    assert verdict["passed"] is False


def test_verify_flags_target_missing_from_baseline(df_with_nulls):
    sandbox = Sandbox(df_with_nulls)
    verdict = sandbox.verify("b", "null")
    assert verdict["target_in_baseline"] is False
    assert verdict["resolved"] is False
    assert verdict["passed"] is False


def test_verify_zero_anomaly_fix():
    df = pd.DataFrame({"a": [0, 1, 2, 3]})
    sandbox = Sandbox(df)
    sandbox.apply("df['a'] = df['a'].replace(0, df['a'].median())")
    verdict = sandbox.verify("a", "zero")
    assert verdict["passed"] is True
