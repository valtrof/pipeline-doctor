# Design: Agentic Remediation Loop (v2)

**Status:** Draft · **Author:** Valeriy Trofimov · **Date:** 2026-07-08

## Summary

Pipeline Doctor v1 is a one-shot advisor: it detects anomalies deterministically, then asks Claude to *explain* them and *suggest* pandas fixes as text. The human does everything after that. v2 closes the loop: an agent takes a confirmed anomaly report, proposes a fix, **applies it in a sandbox, verifies the result by re-running detection, retries on failure**, and produces a remediation report with evidence — never touching the original data and never acting without a human review gate at the end.

One sentence: v1 answers *"what's wrong and what could I do?"* — v2 answers *"here's the fix, here's proof it worked, approve it."*

## Goals

- Turn `suggest_fixes` output from prose into **verified, executable remediation**: fix code that has demonstrably reduced anomalies on a sandboxed copy of the data.
- A bounded agent loop (propose → apply → verify → retry) driven by Claude tool use, with deterministic tools doing all execution and measurement.
- Preserve the v1 core principle: **the LLM never decides whether data is bad — statistics do.** In v2 this extends to: the LLM never decides whether a fix *worked* — re-running detection does.
- Human-in-the-loop by construction: the agent's terminal action is a report (and optionally a GitHub PR containing the fix script); it has no tool that can mutate source data or merge anything.
- Full test coverage without API calls, same as v1 (client injected, mocked in tests).

## Non-goals

- No writes to BigQuery — source data stays read-only; all fixes run on an in-memory DataFrame copy.
- No auto-merge / auto-apply to production. The output is a *proposal with evidence*, always.
- No multi-agent orchestration, no framework dependency (LangGraph etc.) — a single agent with a hand-rolled tool-use loop keeps the control flow inspectable and testable.
- No new detection methods in this iteration (nulls/zeros/IQR outliers stay as-is).

## Architecture

```
                        ┌────────────────────────────────────────────┐
                        │              agent.py (new)                │
                        │      bounded tool-use loop, max N turns    │
                        └────────────────────────────────────────────┘
                            │           │            │          │
                    tool: run_detection │   tool: apply_fix     │ tool: finalize
                            │   tool: get_profile  (sandboxed)  │  (report + verdict)
                            ▼           ▼            ▼          ▼
                        ┌────────────────────────────────────────────┐
                        │            sandbox.py (new)                │
                        │  df copy · restricted exec · diff metrics  │
                        └────────────────────────────────────────────┘
                                          │ reuses
                                          ▼
                        ┌────────────────────────────────────────────┐
                        │       anomaly_detector.py (existing)       │
                        │  detect_anomalies() as the verification    │
                        │  oracle — deterministic, already tested    │
                        └────────────────────────────────────────────┘
```

### The loop (state machine)

```
DETECT ──> PLAN ──> APPLY ──> VERIFY ──┬─ improved & clean ──> REPORT (success)
              ▲                        │
              └── retry (≤ 3/anomaly) ─┴─ worse / error / budget hit ──> REPORT (give-up, with reason)
```

1. **DETECT** — deterministic. `detect_anomalies(df)` produces the baseline report. No LLM.
2. **PLAN** — the agent (Claude) sees the anomaly report plus a column profile and proposes one fix at a time as structured output: `{anomaly_id, fix_code, rationale, expected_effect}`.
3. **APPLY** — `sandbox.apply(fix_code, df_copy)` executes the fix on a copy under a restricted namespace (see Safety). Never on the original.
4. **VERIFY** — deterministic. Re-run `detect_anomalies` on the result and compute the diff: target anomaly resolved? new anomalies introduced? row-loss within budget? The verdict comes from code, not from the model.
5. **Retry or report** — on failure the agent gets the verification diff as feedback and may retry (≤ 3 attempts per anomaly, ≤ 20 total turns). Terminal state is always a report; "couldn't fix it safely" is an acceptable and honest outcome.

### Tools exposed to the agent

| Tool | Type | Description |
|---|---|---|
| `get_profile` | deterministic | Column dtypes, null/zero counts, quantiles — the context a fix needs |
| `apply_fix` | deterministic | Execute one fix snippet in the sandbox; returns execution result or error |
| `verify` | deterministic | Re-run detection on sandbox state; return structured before/after diff |
| `reset_sandbox` | deterministic | Discard sandbox state, start over from the original copy |
| `finalize` | terminal | Emit the remediation report; ends the loop |

Deliberate shape: the agent holds the *reasoning*; every tool is deterministic, unit-testable, and safe to call in any order. This mirrors the v1 boundary and keeps the failure modes legible.

## Safety model

- **Sandboxed execution:** fixes run via `exec` in a namespace containing only `df` and `pd`; builtins reduced to a safe allowlist. Static pre-checks reject imports, dunder access, and file/network/OS references before execution. This is defense against a *confused* model, not a malicious adversary — acceptable for a portfolio system analyzing public data, and stated honestly in the README.
- **Row-loss budget:** a fix that drops > 20% of rows fails verification even if it "resolves" the anomaly — deleting the data always resolves the anomaly, which is exactly the degenerate solution the verifier must catch.
- **Regression check:** verification fails if the fix introduces anomalies in columns that were previously clean.
- **Hard budgets:** max 3 attempts per anomaly, max 20 agent turns, max token spend per run (configurable). The loop cannot run away.
- **Terminal gate:** the agent's only exit is `finalize` — a report for human review. Optional GitHub PR output (fix script + evidence) is opened, never merged.

## Verification metrics (also the eval metrics)

Per anomaly: `resolved` (target gone from report), `regression_free` (no new anomalies), `rows_retained` (≥ 0.8), `attempts`. Per run: fix success rate, false-fix rate (resolved but failed regression/row checks — the dangerous quadrant), give-up rate, mean attempts, token cost.

**Offline eval:** a seeded-anomaly benchmark (`evals/`) injects known defects (nulls, zeros, outliers, mixed) into clean frames, then scores the agent against ground truth. This reuses the LLM-as-judge philosophy from my [llm-eval-harness](https://github.com/valtrof/llm-eval-harness) but needs no judge — ground truth is known by construction. Results table goes in the README.

## API surface

New endpoint alongside the existing `/analyze`:

```
POST /remediate  {dataset, limit} ──> {baseline_report, fixes: [{anomaly, code, verdict, metrics, attempts}], summary, transcript}
```

`/analyze` (v1) remains unchanged — advisory mode is still useful and cheaper.

## Testing strategy

Same discipline as v1: the Anthropic client is injected, so the agent loop is tested with a scripted mock client that returns canned tool calls — including adversarial scripts (fix that deletes all rows, fix that errors, fix that fixes the wrong column) to prove the verifier catches them. Sandbox and verifier are pure functions over DataFrames: plain pytest. No test makes a network call.

## Milestones

1. [x] `DESIGN.md` (this doc) committed — 2026-07-08
2. [x] `sandbox.py` + `verify` diff logic + tests — the deterministic backbone — 2026-07-08 (`profile_anomalies()` added to `anomaly_detector.py` as the structured detection source; string reports rebuilt on top of it, output unchanged; 48 tests green)
3. [x] `agent.py` tool-use loop + mock-client tests — 2026-07-08 (60 tests green; live smoke run on seeded synthetic data: 3/3 anomalies fixed in 10 turns, including one fare-outlier fix that failed verification and was retried successfully — transcript in `demo_run.json`)
4. [x] `/remediate` endpoint + Docker/CI green — 2026-07-08 (7 endpoint tests via TestClient with mocked BigQuery/LLM clients; Docker image now ships sandbox.py + agent.py; CI green on push)
5. [x] Seeded-anomaly eval + results table in README — 2026-07-08 (5 cases / 7 seeded anomalies: 100% verified-fix rate, 0 false fixes, $0.086 total; heavy_nulls case forced two verifier rejections before an acceptable fix — the adaptation loop working as designed)

## Open questions

- Structured fix output: free-form code in a tool argument vs. constrained templates (fillna/replace/filter with parameters). Templates are safer and easier to verify; free-form shows more agent capability. Leaning: free-form with static checks, because the verifier — not the input format — is the real safety boundary, and that's the more interesting engineering claim.
- Model split: Haiku for the loop with Sonnet escalation after two failed attempts, or Haiku-only to keep cost claims simple. Decide after measuring Haiku's fix success rate in the eval.
