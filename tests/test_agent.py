import itertools
from types import SimpleNamespace

import pandas as pd
import pytest

from agent import MAX_ATTEMPTS_PER_ANOMALY, _column_profile, run_remediation


# ---------------------------------------------------------------------------
# scripted mock client
# ---------------------------------------------------------------------------

_ids = itertools.count()


def tool_use(name: str, **tool_input) -> SimpleNamespace:
    return SimpleNamespace(type="tool_use", id=f"tu_{next(_ids)}", name=name, input=tool_input)


def text(content: str) -> SimpleNamespace:
    return SimpleNamespace(type="text", text=content)


def response(*blocks) -> SimpleNamespace:
    stop = "tool_use" if any(b.type == "tool_use" for b in blocks) else "end_turn"
    return SimpleNamespace(content=list(blocks), stop_reason=stop)


class ScriptedClient:
    """Stands in for anthropic.Anthropic: returns pre-scripted responses in order."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        # snapshot the messages list — the loop mutates it in place across turns
        snapshot = dict(kwargs)
        snapshot["messages"] = list(kwargs["messages"])
        self.calls.append(snapshot)
        if not self._responses:
            raise AssertionError("script exhausted: the loop made more API calls than scripted")
        return self._responses.pop(0)


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def df_with_nulls():
    # one null anomaly in `a`; `b` clean
    return pd.DataFrame({"a": [1.0, None, 2.0, 1.5], "b": [1, 2, 3, 4]})


GOOD_FIX = "df['a'] = df['a'].fillna(df['a'].median())"
ROW_DELETING_FIX = "df = df[df['a'].notna()]"
FAILING_FIX = "df['nope'] = df['nope'].fillna(0)"


def fix_call(code=GOOD_FIX, column="a", kind="null"):
    return tool_use("apply_fix", target_column=column, target_kind=kind, fix_code=code,
                    rationale="median imputation preserves the distribution")


def verify_call(column="a", kind="null"):
    return tool_use("verify", target_column=column, target_kind=kind)


def finalize_call(summary="done"):
    return tool_use("finalize", summary=summary)


# ---------------------------------------------------------------------------
# _column_profile
# ---------------------------------------------------------------------------

def test_column_profile_reports_dtypes_nulls_and_stats(df_with_nulls):
    profile = _column_profile(df_with_nulls)
    assert profile["rows"] == 4
    assert profile["columns"]["a"]["null_count"] == 1
    assert profile["columns"]["a"]["median"] == 1.5
    assert profile["columns"]["b"]["zero_count"] == 0
    assert "float" in profile["columns"]["a"]["dtype"]


# ---------------------------------------------------------------------------
# happy path
# ---------------------------------------------------------------------------

def test_happy_path_fix_verify_finalize(df_with_nulls):
    client = ScriptedClient([
        response(text("Fixing nulls in a."), fix_call()),
        response(verify_call()),
        response(finalize_call("Fixed nulls in a via median imputation.")),
    ])
    report = run_remediation(df_with_nulls, "test-data", client)

    assert report["status"] == "completed"
    assert report["summary"] == "Fixed nulls in a via median imputation."
    assert report["verified_fix_count"] == 1
    assert report["fixes"][0]["verdict"]["passed"] is True
    assert report["remaining"] == []
    assert report["rows_final"] == 4
    assert report["turns_used"] == 3
    assert len(client.calls) == 3


def test_original_dataframe_is_never_mutated(df_with_nulls):
    client = ScriptedClient([
        response(fix_call()),
        response(verify_call()),
        response(finalize_call()),
    ])
    run_remediation(df_with_nulls, "test-data", client)
    assert df_with_nulls["a"].isnull().sum() == 1


def test_clean_dataframe_skips_llm_entirely():
    df = pd.DataFrame({"a": [1, 2, 3]})
    client = ScriptedClient([])
    report = run_remediation(df, "clean-data", client)
    assert report["status"] == "clean"
    assert client.calls == []


# ---------------------------------------------------------------------------
# gating: verify before finalize / before next fix
# ---------------------------------------------------------------------------

def test_finalize_without_verify_is_rejected_and_loop_continues(df_with_nulls):
    client = ScriptedClient([
        response(fix_call()),
        response(finalize_call("skipping verification")),  # must be rejected
        response(verify_call()),
        response(finalize_call("verified now")),
    ])
    report = run_remediation(df_with_nulls, "test-data", client)

    rejected = report["transcript"][1]["results"][0]["result"]
    assert rejected["ok"] is False
    assert "verify" in rejected["error"]
    assert report["status"] == "completed"
    assert report["summary"] == "verified now"


def test_second_fix_before_verify_is_rejected(df_with_nulls):
    client = ScriptedClient([
        response(fix_call()),
        response(fix_call()),  # must be rejected: previous fix unverified
        response(verify_call()),
        response(finalize_call()),
    ])
    report = run_remediation(df_with_nulls, "test-data", client)

    rejected = report["transcript"][1]["results"][0]["result"]
    assert rejected["ok"] is False
    assert "not been verified" in rejected["error"]
    # only the first apply counted as an attempt
    assert len([f for f in report["fixes"] if f["applied"]]) == 1


# ---------------------------------------------------------------------------
# failure feedback and retry
# ---------------------------------------------------------------------------

def test_row_deleting_fix_fails_verification_then_retry_succeeds():
    # 3 of 5 rows null: dropping them violates row retention
    df = pd.DataFrame({"a": [1.0, None, None, None, 2.0]})
    client = ScriptedClient([
        response(fix_call(code=ROW_DELETING_FIX)),
        response(verify_call()),          # verifier reports failure
        response(tool_use("reset_sandbox")),
        response(fix_call()),             # median imputation instead
        response(verify_call()),
        response(finalize_call("second attempt passed")),
    ])
    report = run_remediation(df, "test-data", client)

    first_verdict = report["fixes"][0]["verdict"]
    assert first_verdict["resolved"] is True
    assert first_verdict["rows_ok"] is False
    assert first_verdict["passed"] is False

    assert report["status"] == "completed"
    assert report["verified_fix_count"] == 1
    assert report["fixes"][1]["verdict"]["passed"] is True
    assert report["rows_final"] == 5  # reset restored the dropped rows


def test_erroring_fix_returns_error_and_counts_attempt(df_with_nulls):
    client = ScriptedClient([
        response(fix_call(code=FAILING_FIX)),
        response(fix_call()),  # pending was not set by the failed fix, so this is allowed
        response(verify_call()),
        response(finalize_call()),
    ])
    report = run_remediation(df_with_nulls, "test-data", client)

    first_result = report["transcript"][0]["results"][0]["result"]
    assert first_result["ok"] is False
    assert "raised during execution" in first_result["error"]
    assert report["fixes"][0]["attempt"] == 1
    assert report["fixes"][1]["attempt"] == 2
    assert report["status"] == "completed"


def test_attempt_budget_per_anomaly_is_enforced(df_with_nulls):
    failing_attempts = [response(fix_call(code=FAILING_FIX)) for _ in range(MAX_ATTEMPTS_PER_ANOMALY)]
    client = ScriptedClient([
        *failing_attempts,
        response(fix_call()),  # 4th attempt: must be refused without executing
        response(finalize_call("giving up on a/null")),
    ])
    report = run_remediation(df_with_nulls, "test-data", client)

    refused = report["transcript"][MAX_ATTEMPTS_PER_ANOMALY]["results"][0]["result"]
    assert refused["ok"] is False
    assert "attempt budget" in refused["error"]
    assert len(report["fixes"]) == MAX_ATTEMPTS_PER_ANOMALY  # the refused call created no record
    assert report["status"] == "completed"
    assert report["verified_fix_count"] == 0
    assert report["remaining"] != []  # honest report: anomaly still there


# ---------------------------------------------------------------------------
# budgets and nudges
# ---------------------------------------------------------------------------

def test_turn_budget_forces_report(df_with_nulls):
    client = ScriptedClient([
        response(fix_call()),
        response(verify_call()),
        # never finalizes
        response(tool_use("get_profile")),
    ])
    report = run_remediation(df_with_nulls, "test-data", client, max_turns=3)

    assert report["status"] == "budget_exhausted"
    assert report["turns_used"] == 3
    assert report["verified_fix_count"] == 1  # work done before the cutoff is kept


def test_text_only_response_is_nudged(df_with_nulls):
    client = ScriptedClient([
        response(text("Let me think about this.")),  # no tool call
        response(fix_call()),
        response(verify_call()),
        response(finalize_call()),
    ])
    report = run_remediation(df_with_nulls, "test-data", client)

    assert report["status"] == "completed"
    nudge = client.calls[1]["messages"][-1]
    assert nudge["role"] == "user"
    assert "Continue using the tools" in nudge["content"]


def test_loop_sends_system_prompt_and_tools(df_with_nulls):
    client = ScriptedClient([
        response(fix_call()),
        response(verify_call()),
        response(finalize_call()),
    ])
    run_remediation(df_with_nulls, "test-data", client)

    first_call = client.calls[0]
    assert "remediation agent" in first_call["system"]
    tool_names = {t["name"] for t in first_call["tools"]}
    assert tool_names == {"get_profile", "apply_fix", "verify", "reset_sandbox", "finalize"}
    assert "Confirmed anomalies" in first_call["messages"][0]["content"]
