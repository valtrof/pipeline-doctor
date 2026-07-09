"""Agentic remediation loop (v2) — see DESIGN.md.

The agent (Claude) does the planning; every tool it can call is deterministic and
already tested (sandbox.py). All budgets — turns, attempts per anomaly, the
verify-before-finalize gate — are enforced by this loop in code, never by asking
the model to police itself. The loop's only terminal states are a finalize report
or a forced budget-exhaustion report.
"""

import json
import logging
from collections import Counter

import pandas as pd

from anomaly_detector import profile_anomalies
from sandbox import FixExecutionError, FixRejected, Sandbox

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MODEL = "claude-haiku-4-5-20251001"
MAX_TOKENS = 1024
MAX_TURNS = 20
MAX_ATTEMPTS_PER_ANOMALY = 3

ANOMALY_KINDS = ("null", "zero", "outlier")

SYSTEM_PROMPT = f"""\
You are Pipeline Doctor's remediation agent. A deterministic scanner has already \
confirmed data anomalies in a pandas DataFrame called `df`. Your job is to fix them \
one at a time in a sandbox and prove each fix worked.

Rules:
- Work on one anomaly at a time: call apply_fix, then immediately call verify for \
the same target. You cannot apply another fix or finalize until the last applied \
fix is verified.
- fix_code must be plain executable pandas operating on `df` (pd is also in scope). \
No imports, no file access, no while loops.
- Verification is statistical, not up to you: a fix passes only if the target \
anomaly is gone, no new anomalies appeared, and at least 80% of rows survive. \
Deleting rows to make an anomaly disappear will fail verification.
- If verification fails, read the diff and either try a different fix (budget: \
{MAX_ATTEMPTS_PER_ANOMALY} attempts per anomaly), call reset_sandbox to start over, \
or move on to the next anomaly.
- Call get_profile whenever you need column types and statistics to design a fix.
- When every anomaly is resolved or you have run out of safe options, call finalize \
with an honest summary. "Could not fix this safely" is an acceptable outcome; a \
destructive fix is not.
- You have a hard budget of {MAX_TURNS} turns."""

TOOLS = [
    {
        "name": "get_profile",
        "description": "Column-level profile of the current sandbox DataFrame: dtype, null/zero "
                       "counts, and basic statistics for numeric columns. Use it to design fixes.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "apply_fix",
        "description": "Execute one pandas fix snippet against the sandbox copy of `df`. "
                       "The snippet may mutate df in place or rebind it (df = df[...]). "
                       "Counts as one attempt for the target anomaly.",
        "input_schema": {
            "type": "object",
            "properties": {
                "target_column": {"type": "string", "description": "Column of the anomaly this fix targets."},
                "target_kind": {"type": "string", "enum": list(ANOMALY_KINDS)},
                "fix_code": {"type": "string", "description": "Executable pandas code using `df`."},
                "rationale": {"type": "string", "description": "One sentence: why this fix is appropriate."},
            },
            "required": ["target_column", "target_kind", "fix_code", "rationale"],
        },
    },
    {
        "name": "verify",
        "description": "Re-run statistical detection on the sandbox and judge the last fix: "
                       "target resolved, no new anomalies, row retention within budget.",
        "input_schema": {
            "type": "object",
            "properties": {
                "target_column": {"type": "string"},
                "target_kind": {"type": "string", "enum": list(ANOMALY_KINDS)},
            },
            "required": ["target_column", "target_kind"],
        },
    },
    {
        "name": "reset_sandbox",
        "description": "Discard every applied fix and restore the sandbox to the original data. "
                       "Attempt counts are NOT reset.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "finalize",
        "description": "End the run and emit the remediation report for human review. "
                       "Only allowed once the last applied fix has been verified.",
        "input_schema": {
            "type": "object",
            "properties": {
                "summary": {"type": "string", "description": "Honest summary of what was fixed, what wasn't, and why."},
            },
            "required": ["summary"],
        },
    },
]


def _column_profile(df: pd.DataFrame) -> dict:
    """JSON-serializable per-column profile of a DataFrame."""
    profile = {}
    for col in df.columns:
        info = {"dtype": str(df[col].dtype), "null_count": int(df[col].isnull().sum())}
        if pd.api.types.is_numeric_dtype(df[col]):
            series = df[col]
            info["zero_count"] = int((series == 0).sum())
            for stat, value in (("min", series.min()), ("max", series.max()), ("median", series.median())):
                info[stat] = None if pd.isna(value) else float(value)
        profile[str(col)] = info
    return {"rows": len(df), "columns": profile}


class _AgentState:
    """Mutable state the loop tracks across turns — the trusted record of the run."""

    def __init__(self, df: pd.DataFrame, dataset_name: str):
        self.sandbox = Sandbox(df)
        self.dataset_name = dataset_name
        self.attempts: Counter = Counter()
        self.pending: tuple | None = None  # (column, kind) applied but not yet verified
        self.fixes: list[dict] = []
        self.transcript: list[dict] = []
        self.usage: dict = {"input_tokens": 0, "output_tokens": 0}


def _apply_fix(state: _AgentState, tool_input: dict) -> dict:
    column = tool_input["target_column"]
    kind = tool_input["target_kind"]
    if kind not in ANOMALY_KINDS:
        return {"ok": False, "error": f"target_kind must be one of {ANOMALY_KINDS}"}
    if state.pending is not None:
        return {"ok": False, "error": f"the fix for {state.pending[0]}/{state.pending[1]} has not been verified yet; call verify first"}

    target = (column, kind)
    if state.attempts[target] >= MAX_ATTEMPTS_PER_ANOMALY:
        return {"ok": False, "error": f"attempt budget ({MAX_ATTEMPTS_PER_ANOMALY}) exhausted for {column}/{kind}; move on or finalize"}

    state.attempts[target] += 1
    record = {
        "target": {"column": column, "kind": kind},
        "fix_code": tool_input["fix_code"],
        "rationale": tool_input["rationale"],
        "attempt": state.attempts[target],
        "applied": False,
        "error": None,
        "verdict": None,
    }
    state.fixes.append(record)

    try:
        rows = state.sandbox.apply(tool_input["fix_code"])
    except (FixRejected, FixExecutionError) as e:
        record["error"] = str(e)
        return {"ok": False, "error": str(e), "attempt": record["attempt"]}

    record["applied"] = True
    state.pending = target
    return {"ok": True, "attempt": record["attempt"], **rows}


def _verify(state: _AgentState, tool_input: dict) -> dict:
    column = tool_input["target_column"]
    kind = tool_input["target_kind"]
    verdict = state.sandbox.verify(column, kind)

    if state.pending == (column, kind):
        state.pending = None
        for record in reversed(state.fixes):
            if record["target"] == {"column": column, "kind": kind} and record["applied"] and record["verdict"] is None:
                record["verdict"] = verdict
                break
    return verdict


def _build_report(state: _AgentState, status: str, summary: str, turns_used: int) -> dict:
    verified = [f for f in state.fixes if f["verdict"] and f["verdict"]["passed"]]
    return {
        "status": status,
        "dataset": state.dataset_name,
        "summary": summary,
        "baseline": state.sandbox.baseline,
        "remaining": profile_anomalies(state.sandbox.df),
        "fixes": state.fixes,
        "verified_fix_count": len(verified),
        "rows_original": len(state.sandbox._original),
        "rows_final": len(state.sandbox.df),
        "turns_used": turns_used,
        "usage": state.usage,
        "transcript": state.transcript,
    }


def _execute_tool(state: _AgentState, name: str, tool_input: dict, turns_used: int) -> tuple[dict, dict | None]:
    """Dispatch one tool call. Returns (tool_result, final_report_or_None)."""
    if name == "get_profile":
        return _column_profile(state.sandbox.df), None
    if name == "apply_fix":
        return _apply_fix(state, tool_input), None
    if name == "verify":
        return _verify(state, tool_input), None
    if name == "reset_sandbox":
        state.sandbox.reset()
        state.pending = None
        return {"ok": True, "note": "sandbox restored to original data; attempt counts kept"}, None
    if name == "finalize":
        if state.pending is not None:
            return {"ok": False, "error": f"the fix for {state.pending[0]}/{state.pending[1]} has not been verified; call verify before finalize"}, None
        report = _build_report(state, "completed", tool_input["summary"], turns_used)
        return {"ok": True}, report
    return {"ok": False, "error": f"unknown tool: {name}"}, None


def run_remediation(df: pd.DataFrame, dataset_name: str, client, max_turns: int = MAX_TURNS) -> dict:
    """Run the bounded agent loop against a DataFrame with confirmed anomalies.

    The client is injected (same pattern as v1) so tests drive the loop with a
    scripted mock. Returns the remediation report; the source DataFrame is never
    mutated.
    """
    state = _AgentState(df, dataset_name)

    if not state.sandbox.baseline:
        logger.info("No anomalies in %s — nothing to remediate.", dataset_name)
        return _build_report(state, "clean", "No anomalies detected; nothing to remediate.", 0)

    anomaly_lines = "\n".join(
        f"- {r['column']}: {r['kind']} (count={r['count']}, {r['pct']}%)" for r in state.sandbox.baseline
    )
    messages = [{
        "role": "user",
        "content": f"Dataset: {dataset_name}\nRows: {len(df)}\nConfirmed anomalies:\n{anomaly_lines}\n\nBegin remediation.",
    }]

    for turn in range(1, max_turns + 1):
        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        )
        messages.append({"role": "assistant", "content": response.content})

        usage = getattr(response, "usage", None)  # scripted mocks don't provide one
        if usage is not None:
            state.usage["input_tokens"] += getattr(usage, "input_tokens", 0) or 0
            state.usage["output_tokens"] += getattr(usage, "output_tokens", 0) or 0

        entry = {"turn": turn, "assistant": [], "results": []}
        tool_results = []
        final_report = None

        for block in response.content:
            if block.type == "text":
                entry["assistant"].append({"type": "text", "text": block.text})
            elif block.type == "tool_use":
                entry["assistant"].append({"type": "tool_use", "name": block.name, "input": dict(block.input)})
                result, report = _execute_tool(state, block.name, dict(block.input), turn)
                entry["results"].append({"tool": block.name, "result": result})
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(result, default=str),
                })
                if report is not None:
                    final_report = report

        state.transcript.append(entry)

        if final_report is not None:
            logger.info("Remediation finalized after %d turns: %d verified fixes.",
                        turn, final_report["verified_fix_count"])
            return final_report

        if tool_results:
            messages.append({"role": "user", "content": tool_results})
        else:
            # model produced text only — nudge it back to the tool loop
            messages.append({"role": "user", "content": "Continue using the tools; call finalize when done."})

    logger.warning("Turn budget (%d) exhausted before finalize.", max_turns)
    return _build_report(
        state, "budget_exhausted",
        f"Turn budget ({max_turns}) exhausted before the agent finalized; partial results reported.",
        max_turns,
    )
