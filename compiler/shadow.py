"""Shadow-mode harness.

Runs the NEW path (LLM -> IR-1 -> compiler -> engine) alongside the existing
pipeline and compares outputs, WITHOUT affecting what the user sees. This is the
safe way to validate the v2 path against real traffic before any cut-over: the
legacy result is always what's returned; the shadow comparison is recorded only.

Disabled by default (config.SHADOW_MODE). The planner is injectable so the harness
is fully testable without a live Gemini call.
"""
import pandas as pd

from compiler.core import compile_logical
from agents.plan_executor import execute_plan


def run_new_path(df, query, snapshot_dates=None, planner=None):
    """Execute the new path. Returns (result_df | None, ir1 | None, error_str).
    `planner` defaults to the live Gemini planner but can be injected with a stub
    that returns a hand-written IR-1 for deterministic tests."""
    if planner is None:
        from agents.logical_planner import plan_logical as planner

    try:
        ir1 = planner(query, snapshot_dates)
    except Exception as e:
        return None, None, f"planner failed: {e}"

    plan, errs = compile_logical(ir1, list(df.columns))
    if errs:
        return None, ir1, "; ".join(errs)

    try:
        out, err = execute_plan(df, plan)
    except Exception as e:
        return None, ir1, f"execute failed: {e}"
    if err:
        return None, ir1, err
    return out, ir1, ""


def _numeric_signature(df) -> dict | None:
    """Schema-name-agnostic signature of a result frame: row count + the sorted
    multiset of numeric cell values (rounded). The 'Rank' column added by the plan
    engine is excluded so it doesn't skew the comparison. Heuristic by design  - 
    shadow mode FLAGS divergence for review, it does not gate execution."""
    if df is None:
        return None
    work = df.drop(columns=[c for c in df.columns if str(c).lower() == "rank"], errors="ignore")
    nums = work.select_dtypes("number").to_numpy().ravel()
    vals = sorted(round(float(v), 2) for v in nums if pd.notna(v))
    return {"rows": int(len(df)), "nums": vals}


def compare_results(legacy_df, new_df) -> dict:
    """Compare legacy vs new result frames. Returns a verdict dict:
    status in {match, mismatch, new_error}; with row_count_match / values_match."""
    new_sig = _numeric_signature(new_df)
    if new_sig is None:
        return {"status": "new_error", "row_count_match": False, "values_match": False}
    legacy_sig = _numeric_signature(legacy_df) or {"rows": -1, "nums": None}

    row_match = legacy_sig["rows"] == new_sig["rows"]
    val_match = legacy_sig["nums"] == new_sig["nums"]
    return {
        "status": "match" if (row_match and val_match) else "mismatch",
        "row_count_match": row_match,
        "values_match": val_match,
        "legacy_rows": legacy_sig["rows"],
        "new_rows": new_sig["rows"],
    }


def shadow_evaluate(df, query, legacy_result_df, snapshot_dates=None, planner=None) -> dict:
    """End-to-end shadow step: run the new path and compare to the legacy result.
    Always returns a verdict dict (never raises) so it is safe to call inline in
    the live pipeline behind the SHADOW_MODE flag."""
    try:
        new_df, ir1, err = run_new_path(df, query, snapshot_dates, planner)
        if err:
            return {"status": "new_error", "error": err, "ir1": ir1}
        verdict = compare_results(legacy_result_df, new_df)
        verdict["ir1"] = ir1
        return verdict
    except Exception as e:  # belt-and-suspenders: shadow must never break the user path
        return {"status": "exception", "error": str(e)}
