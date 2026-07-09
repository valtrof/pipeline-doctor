"""Sandboxed execution and verification of agent-proposed pandas fixes.

This is the deterministic backbone of the v2 agent loop (see DESIGN.md): fix code
executes only on a copy of the data under a restricted namespace, and verification
re-runs the same statistical detection that produced the baseline. The verdict on
whether a fix worked comes from code, not from the model.

The static checks defend against a *confused* model (imports, file access, runaway
loops, dunder escapes), not a determined adversary — see the Safety model section
of DESIGN.md.
"""

import ast
import builtins
import logging

import pandas as pd

from anomaly_detector import profile_anomalies

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# A fix that "resolves" an anomaly by discarding more than this share of rows fails
# verification — deleting the data always resolves the anomaly.
ROW_RETENTION_MIN = 0.8

_SAFE_BUILTIN_NAMES = (
    "abs", "bool", "dict", "enumerate", "float", "int", "len", "list", "max",
    "min", "range", "round", "set", "sorted", "str", "sum", "tuple", "zip",
)
_SAFE_BUILTINS = {name: getattr(builtins, name) for name in _SAFE_BUILTIN_NAMES}

_BANNED_NAMES = frozenset({
    "os", "sys", "subprocess", "socket", "shutil", "pathlib", "importlib", "builtins",
    "open", "eval", "exec", "compile", "input", "breakpoint", "exit", "quit",
    "globals", "locals", "vars", "getattr", "setattr", "delattr", "type", "object",
    "memoryview", "help",
})

# pandas file-I/O entry points — the escape hatch the namespace restriction alone
# would leave open (e.g. pd.read_csv / df.to_csv).
_BANNED_ATTRS = frozenset({
    "read_csv", "read_table", "read_parquet", "read_json", "read_pickle", "read_sql",
    "read_sql_query", "read_sql_table", "read_html", "read_excel", "read_feather",
    "read_orc", "read_hdf", "read_clipboard", "read_fwf", "read_xml", "read_stata",
    "read_sas", "read_spss",
    "to_csv", "to_parquet", "to_json", "to_pickle", "to_sql", "to_html", "to_excel",
    "to_feather", "to_orc", "to_hdf", "to_clipboard", "to_stata", "to_xml", "to_latex",
})


class FixRejected(ValueError):
    """Fix code failed static safety checks and was not executed."""


class FixExecutionError(RuntimeError):
    """Fix code raised during sandboxed execution or corrupted `df`."""


def check_fix_code(code: str) -> None:
    """Statically check one fix snippet; raise FixRejected if it is unsafe.

    Rejected: syntax errors, imports, while loops (unbounded), access to private
    or dunder attributes, banned builtin/module names, and pandas file-I/O calls.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        raise FixRejected(f"syntax error: {e}")

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise FixRejected("imports are not allowed in fix code")
        if isinstance(node, ast.While):
            raise FixRejected("while loops are not allowed in fix code")
        if isinstance(node, ast.Attribute):
            if node.attr.startswith("_"):
                raise FixRejected(f"access to private attribute `{node.attr}` is not allowed")
            if node.attr in _BANNED_ATTRS:
                raise FixRejected(f"file I/O via `{node.attr}` is not allowed in fix code")
        if isinstance(node, ast.Name):
            if node.id in _BANNED_NAMES:
                raise FixRejected(f"use of `{node.id}` is not allowed in fix code")
            if node.id.startswith("_"):
                raise FixRejected(f"use of private name `{node.id}` is not allowed")


class Sandbox:
    """A working copy of a DataFrame that agent fixes are applied to and verified against.

    The original DataFrame is never mutated. Applying a fix is atomic: execution runs
    on a copy of the current state, so a fix that raises leaves the sandbox unchanged.
    """

    def __init__(self, df: pd.DataFrame):
        self._original = df.copy()
        self.df = df.copy()
        self.baseline = profile_anomalies(df)

    def apply(self, fix_code: str) -> dict:
        """Execute one fix snippet against the sandbox DataFrame.

        The snippet runs with only `df` and `pd` in scope and a minimal builtins
        allowlist. It may mutate `df` in place or rebind it (`df = df[...]`).

        Returns:
            {"rows_before": int, "rows_after": int}

        Raises:
            FixRejected: the code failed static checks and was not executed.
            FixExecutionError: the code raised, or left `df` as a non-DataFrame.
        """
        check_fix_code(fix_code)

        namespace = {"df": self.df.copy(), "pd": pd, "__builtins__": _SAFE_BUILTINS}
        try:
            exec(fix_code, namespace)
        except Exception as e:
            raise FixExecutionError(f"fix raised during execution: {e}") from e

        result = namespace.get("df")
        if not isinstance(result, pd.DataFrame):
            raise FixExecutionError("after execution `df` is no longer a pandas DataFrame")

        rows_before = len(self.df)
        self.df = result
        logger.info("Fix applied: rows %d -> %d", rows_before, len(result))
        return {"rows_before": rows_before, "rows_after": len(result)}

    def reset(self) -> None:
        """Discard all applied fixes and restore the original data."""
        self.df = self._original.copy()
        logger.info("Sandbox reset to original data.")

    def verify(self, target_column: str, target_kind: str) -> dict:
        """Re-run detection on the sandbox state and judge it against the baseline.

        A fix passes only if the target anomaly is gone, no new (column, kind)
        anomalies appeared anywhere, and at least ROW_RETENTION_MIN of the original
        rows survive — the last two catch fixes that "work" by deleting or
        distorting data.

        Args:
            target_column: column of the anomaly the fix was meant to resolve.
            target_kind: "null", "zero", or "outlier".

        Returns:
            A structured verdict dict; see keys below. The verdict is computed
            entirely from detection output — no LLM involvement.
        """
        current = profile_anomalies(self.df)
        baseline_keys = {(r["column"], r["kind"]) for r in self.baseline}
        current_keys = {(r["column"], r["kind"]) for r in current}
        target = (target_column, target_kind)

        target_in_baseline = target in baseline_keys
        resolved = target_in_baseline and target not in current_keys
        new_keys = sorted(current_keys - baseline_keys)
        rows_retained = len(self.df) / len(self._original) if len(self._original) else 1.0
        rows_ok = rows_retained >= ROW_RETENTION_MIN
        regression_free = not new_keys
        passed = resolved and regression_free and rows_ok

        verdict = {
            "target": {"column": target_column, "kind": target_kind},
            "target_in_baseline": target_in_baseline,
            "resolved": resolved,
            "regression_free": regression_free,
            "new_anomalies": [{"column": c, "kind": k} for c, k in new_keys],
            "rows_retained": round(rows_retained, 3),
            "rows_ok": rows_ok,
            "passed": passed,
            "anomalies_before": len(self.baseline),
            "anomalies_after": len(current),
        }
        logger.info(
            "Verification for %s/%s: passed=%s (resolved=%s, regression_free=%s, rows_retained=%.3f)",
            target_column, target_kind, passed, resolved, regression_free, rows_retained,
        )
        return verdict
