import re
import threading
import uuid as _uuid
from typing import Any, Callable, Optional, TypedDict

import pandas as pd
from langgraph.graph import END, START, StateGraph
from langsmith import traceable
from langsmith.run_helpers import get_current_run_tree

from agents.data_executor import (
    _build_mask,
    compute_contextual_rankings,
    compute_result_kpis,
    execute_priority_mode,
)
from agents.plan_executor import execute_plan
from agents.logical_planner import plan_logical
from agents.insight_generator import generate_insights
from compiler.core import compile_logical, _expand_filters
from registry.semantic_model import resolve_dimension
from registry.views import VIEWS, _METRIC_DIRECTION, _METRIC_AGG, normalize_view_output, resolve_view_fn
from query_log import log_query_outcome


# ── Per-thread step callback ──────────────────────────────────────────────────
_tls = threading.local()

# ── Large-object side channel ────────────────────────────────────────────────
# The actual fix for the LangGraph-tracing payload blowup: result_df_full,
# df_prev, precomputed_views, and result_df (the per-query answer table, which
# can itself be as large as the whole file if a query matches most rows) never
# go INTO the QueryState dict that LangGraph passes between nodes and traces
# at every transition. They live here instead, in the SAME thread the query
# runs in (matching the existing _tls step-callback pattern -- one query runs
# synchronously per thread, so no cross-query collision risk). Every node that
# used to read these via state.get(...) now reads them via _fetch_large(...)
# instead; state itself only ever carries a small placeholder for them, so
# LangGraph's own tracing (which serializes the full state on every node
# transition, independent of and not controllable via this file's
# process_inputs/process_outputs on the @traceable node wrappers) never sees
# anything but tiny scalar/dict fields, regardless of how large the real file is.
def _stash_large(key: str, value) -> None:
    if not hasattr(_tls, "large_data"):
        _tls.large_data = {}
    _tls.large_data[key] = value


def _fetch_large(state: dict, key: str, default=None):
    """Check the thread-local stash first (the real path via run_query()); if
    nothing was stashed (e.g. a node function called directly with a hand-built
    state dict, as tests do), fall back to reading the field straight out of
    state -- preserves backward compatibility for any caller that still puts
    the real object directly in state, while run_query()'s real invocation
    benefits from never putting it there in the first place."""
    stash = getattr(_tls, "large_data", {})
    if key in stash:
        return stash[key]
    return state.get(key, default)


_STEP_LABELS: dict[str, str] = {
    "planner":  "🧠  Query Planner: understanding your question",
    "compile":  "⚙️   Compiler: building and validating the execution plan",
    "view":     "⚡  Fast Path: fetching a pre-computed answer",
    "execute":  "⚡  Data Executor: computing the answer",
    "analyze":  "💡  Insight Generator: writing AI observations",
}

_MAX_REPAIRS = 1


def _trace_metadata(fields: dict) -> None:
    """Attach observability metadata to the CURRENT LangSmith run (the
    @traceable-wrapped node function this is called from), same established
    pattern as agents/domain_expert.py's/agents/insight_generator.py's token-
    usage recording. Only the 2 Gemini-call functions were traced before this
    pass -- graph.py's own nodes (view matching, compiler repair, executor)
    had zero visibility: no way to see which view matched, whether a fast-path
    view fell through and why, whether a repair attempt fired, or basic result
    shape, without re-deriving it from the raw query outcome log. A no-op
    outside a traced context (e.g. running graph.py functions directly in a
    unit test with no LangSmith run active) -- never raises."""
    try:
        rt = get_current_run_tree()
        if rt is not None:
            rt.add_metadata(fields)
    except Exception:
        pass


# Every node function's sole argument/return value is the full QueryState --
# which carries the ENTIRE raw uploaded file (result_df_full, potentially tens
# of thousands of rows) plus df_prev/result_df/precomputed_views, all full
# DataFrames. @traceable's DEFAULT behavior serializes the complete input/
# output of every traced call to send to LangSmith. Without stripping these
# fields first, a single node call on a real ~60k-row file produced a ~47MB
# trace payload -- LangSmith's limit is 20MB, so every one of those calls
# failed to upload and RETRIED repeatedly, adding ~14+ seconds of pure
# network-retry overhead PER NODE CALL (confirmed via direct measurement).
# With ~4 traced nodes firing per query, that's 45-60+ seconds of pure
# instrumentation overhead added on top of the real pipeline work -- the
# actual root cause of a severe felt-latency regression after this session's
# LangSmith tracing pass. process_inputs/process_outputs below replace those
# large fields with a cheap shape summary before @traceable ever serializes
# anything, so the trace stays small and fast while still showing which
# fields were present and how big they were.
_LARGE_STATE_FIELDS = ("result_df_full", "df_prev", "result_df", "precomputed_views", "alerts_curr", "alerts_prev")

# Customer-identifying columns that must never leave this machine in an AI
# prompt or a trace payload. result_kpis["_agg_rows"] (built below, and in
# execute_node for aggregation results) is a small sample of real row data
# handed to the Insight Generator so it has something concrete to describe --
# for the 4 customer/loan-grain fast-path views (top_accounts, fleet_exposure,
# repossession, good_customers), those raw rows carry real customer name/phone
# columns. The Insight Generator only needs the analytical columns (SOH,
# bucket, region, branch, arrears, etc.) to write a useful observation -- it
# never needed the customer's identity, so stripping these costs nothing in
# insight quality. Applied at construction (below) AND independently inside
# the LangSmith trace-stripping code (_strip_large_state_fields) as a second
# layer, matching this codebase's existing dual-layer pattern (see
# agents/plan_executor.py's derive-expression check at both compile and
# execute time) -- so a future code path populating _agg_rows without going
# through the same construction site still can't leak this to LangSmith.
_PII_COLS_IN_AGG_ROWS = ("Cust Name", "Cust Mob No", "Guar Name", "Guar Mob No")


def _strip_pii_from_agg_rows(rows: list) -> list:
    if not rows:
        return rows
    return [{k: v for k, v in row.items() if k not in _PII_COLS_IN_AGG_ROWS} for row in rows]


def _strip_large_state_fields(d: dict) -> dict:
    if not isinstance(d, dict):
        return d
    out = dict(d)
    for k in _LARGE_STATE_FIELDS:
        v = out.get(k)
        if v is not None and hasattr(v, "__len__"):
            out[k] = f"<{type(v).__name__} len={len(v)}>"
    kpis = out.get("result_kpis")
    if isinstance(kpis, dict) and kpis.get("_agg_rows"):
        out["result_kpis"] = {**kpis, "_agg_rows": _strip_pii_from_agg_rows(kpis["_agg_rows"])}
    return out


def _trace_process_inputs(inputs: dict) -> dict:
    # process_inputs receives {"state": <QueryState>} -- the param name -> value.
    state = inputs.get("state")
    if isinstance(state, dict):
        return {**inputs, "state": _strip_large_state_fields(state)}
    return inputs


def _trace_process_outputs(outputs: dict) -> dict:
    # process_outputs receives the raw returned QueryState dict directly.
    return _strip_large_state_fields(outputs)


def _announce(node: str) -> None:
    cb: Optional[Callable[[str], None]] = getattr(_tls, "step_callback", None)
    if cb is None:
        return
    label = _STEP_LABELS.get(node)
    if label:
        try:
            cb(label)
        except Exception:
            pass


# ── Graph state ───────────────────────────────────────────────────────────────

class QueryState(TypedDict):
    # Input
    query: str
    result_df_full: Any
    snapshot_dates: dict

    # Fast-path view inputs (threaded from app.py's already-cached analysis/ results)
    df_prev: Any
    precomputed_views: dict
    alerts_curr: list
    alerts_prev: list
    rr_meta: dict

    # IR-1 (from logical planner)
    ir1: dict

    # Backward-compat UI fields  -  set from ir1 in logical_planner_node
    enriched_query: str
    query_category: str
    query_title: str
    focus_kpis: list
    insight_focus: str
    risk_flag: str
    priority_mode: bool
    aggregation_mode: bool
    aggregation_spec: dict
    plan_mode: bool
    plan: list
    result_type: str
    result_grain: str  # row grain of result_df: "loan" (default), "region", "branch", "executive", "customer", "segment", "signal"
    view_render: str  # UI hint from a fast-path view (e.g. "kpi_cards"); "" for the normal compiler path

    # Clarification
    allow_clarification: bool
    needs_clarification: bool
    clarification_question: str
    clarification_options: list

    # Opt-out of the Insight Generator's Gemini call (skip_insights=True) --
    # saves the entire 2nd Gemini call's latency (observed to sometimes be the
    # SLOWER of the two calls, not always the cheap second step) for users who
    # only want the table/KPIs, not the written narrative.
    skip_insights: bool

    # Kept for UI backward compat (plain_english display)
    parsed_filters: dict

    # Executor outputs
    result_df: Any
    result_kpis: dict
    result_rankings: dict
    result_highlights: list  # query-aware standout KPI cards (view path only; see view_node)
    result_portfolio_kpis: list  # unconditional portfolio-wide rollup (view path only; see view_node)

    # Insight output
    insights: str

    # Error
    error: str

    # LangSmith trace ID
    run_id: str

    # Shadow-mode comparison slot (kept for backward compat; no longer populated)
    shadow: dict


# ── Node 1: Logical Planner (Gemini) ─────────────────────────────────────────
@traceable(run_type="chain", name="LogicalPlannerNode", tags=["node"], process_inputs=_trace_process_inputs, process_outputs=_trace_process_outputs)
def logical_planner_node(state: QueryState) -> QueryState:
    _announce("planner")
    try:
        ir1 = plan_logical(
            state["query"],
            state.get("snapshot_dates"),
            allow_clarification=state.get("allow_clarification", True),
        )
        intent = ir1.get("intent", "loan_table")

        # Map intent → UI mode flags (backward compat for UI rendering branches)
        priority_mode = (intent == "priority_action")
        # single_value with no dimensions = scalar filter query, not a group-by.
        # aggregation_mode=True only when there is an actual group dimension to rank.
        has_dims = bool(ir1.get("dimensions"))
        aggregation_mode = intent == "aggregation" or (intent == "single_value" and has_dims)

        # Build a synthetic aggregation_spec so the existing UI renderer can show
        # the correct column headers without needing schema changes.
        dims = ir1.get("dimensions") or []
        measures_ir = ir1.get("measures") or []
        if dims:
            group_by_cols = resolve_dimension(dims[0])
            # executive dimension → show as combined "MNT NAME (Unit)" label in the UI header
            if dims[0] == "executive" and len(group_by_cols) >= 2:
                group_col = f"{group_by_cols[0]} ({group_by_cols[1]})"
            else:
                group_col = group_by_cols[0] if group_by_cols else dims[0]
        else:
            group_col = "Group"
        metric_labels = [
            {"label": m.get("alias") or m.get("metric", "metric")}
            for m in measures_ir[:4]
        ]
        agg_spec = {"group_by": group_col, "metrics": metric_labels}

        # result_type for UI branching.
        # single_value with no dimensions = scalar KPI display (single_stat).
        # Everything else (including single_value+dims, which is a ranking) = loan_table.
        result_type = "single_stat" if intent == "single_value" and not has_dims else "loan_table"

        # insight_focus gives the insight generator contextual direction
        _focus_map = {
            "loan_table":       "individual account risk and payment behavior",
            "aggregation":      "group performance and comparison across portfolio",
            "single_value":     "portfolio summary metric and what drives it",
            "priority_action":  "business priority, urgency, and next action",
        }
        insight_focus = _focus_map.get(intent, "portfolio analysis")

        # query_category badge in the UI
        _cat_map = {
            "loan_table":      "risk",
            "aggregation":     "analytics",
            "single_value":    "summary",
            "priority_action": "priority",
        }
        query_category = _cat_map.get(intent, "general")

        _trace_metadata({
            "intent": intent,
            "view_requested": (ir1.get("view") or {}).get("name"),
            "needs_clarification": bool(ir1.get("needs_clarification", False)),
            "priority_mode": priority_mode,
            "risk_flag": ir1.get("risk_flag") or "medium",
        })

        return {
            **state,
            "ir1":               ir1,
            "enriched_query":    ir1.get("description") or state["query"],
            "query_category":    query_category,
            "query_title":       ir1.get("query_title") or "Custom Query",
            "focus_kpis":        [],
            "insight_focus":     insight_focus,
            "risk_flag":         ir1.get("risk_flag") or "medium",
            "priority_mode":     priority_mode,
            "aggregation_mode":  aggregation_mode,
            "aggregation_spec":  agg_spec,
            "plan_mode":         False,          # never set; all results display via agg/filter UI
            "plan":              [],
            "result_type":       result_type,
            "result_grain":      "loan",
            "needs_clarification":    bool(ir1.get("needs_clarification", False)),
            "clarification_question": ir1.get("clarification_question") or "",
            "clarification_options":  ir1.get("clarification_options") or [],
            "parsed_filters":    {"plain_english": ir1.get("description") or state["query"]},
            "error":             "",
        }
    except Exception as e:
        return {**state, "error": f"Query planning failed: {e}"}


def _normalize_col(s: str) -> str:
    # "Δ" (month-over-month delta prefix, e.g. "Δ NPA%") must NOT be stripped as
    # punctuation -- doing so collapsed "NPA%"/"NPA"/"Δ NPA%" to the identical
    # normalized string "npa", so a near-miss column request like "NPA %" could
    # resolve to the WRONG column (the delta) via silent dict-key collision in
    # _resolve_column_fuzzy's norm_map. Spelling it out keeps them distinct
    # ("npa" vs "deltanpa") while still normalizing away spacing/case/punctuation.
    return re.sub(r"[^a-z0-9]", "", s.replace("Δ", "delta").lower())


def _resolve_column_fuzzy(name: str, columns) -> str | None:
    """Resolve a planner-supplied column name against a view's real columns,
    tolerating minor cross-view spelling drift (e.g. the planner naming
    "Strike Rate %" -- executive_scorecard's spelling -- while looking at
    region_scorecard, whose own column is "Strike%"). Exact match first, then
    a normalized (strip spacing/punctuation/case) exact match, then a
    normalized substring match in either direction, gated to a minimum length
    so short names like "NPA" can't spuriously match unrelated columns.
    Returns None (never raises) when nothing reasonably matches -- the caller
    treats that as "leave the view's default order alone"."""
    if name in columns:
        return name
    norm_name = _normalize_col(name)
    norm_map = {_normalize_col(c): c for c in columns}
    if norm_name in norm_map:
        return norm_map[norm_name]
    if len(norm_name) >= 4:
        for norm_c, c in norm_map.items():
            if len(norm_c) >= 4 and (norm_name in norm_c or norm_c in norm_name):
                return c
    return None


def _apply_sort_by(result_df: pd.DataFrame, sort_by: dict | None) -> pd.DataFrame:
    """Query-aware ranking: "rank regions by Collection%" re-sorts the view's own
    table by whatever column the user asked for, instead of always showing the
    view's built-in default order. An unresolvable column, or no request at
    all, is a no-op -- the view's default sort stands. Never worth failing the
    whole query over a sort request that doesn't quite resolve."""
    if not sort_by:
        return result_df
    col = _resolve_column_fuzzy(sort_by.get("column", ""), result_df.columns)
    if col is None:
        return result_df
    result_df = result_df.sort_values(
        col, ascending=(sort_by.get("dir") == "asc")
    ).reset_index(drop=True)
    # branch_quadrant ships its own "Rank" column (1..N by Concern Score). Since
    # we just re-sorted by a DIFFERENT column, that Rank now describes the OLD
    # order -- e.g. row 1 could show "Rank: 7", contradicting its new position.
    # Recompute it to match the display order actually shown, same convention
    # compute_branch_quadrant itself uses (1-indexed, best-to-worst per this sort).
    if "Rank" in result_df.columns:
        result_df["Rank"] = range(1, len(result_df) + 1)
    return result_df


def _format_metric_value(col: str, value) -> str:
    """Shared display formatting for a metric value, used by both highlight
    cards and the portfolio KPI rollup -- a percent-named column always shows
    2 decimals + '%'; a whole-number value shows with thousands separators;
    anything else shows 2 decimals."""
    if "%" in col:
        return f"{value:.2f}%"
    if float(value) == int(value):
        return f"{int(value):,}"
    return f"{value:.2f}"


def _build_highlights(spec: dict, result_df: pd.DataFrame, requests: list) -> list:
    """Turn the planner's highlight_metrics requests into standout KPI cards.
    The planner only names WHICH columns/direction matter to the question --
    every value here is a real idxmax/idxmin lookup on result_df, never an
    LLM-supplied number. Any request referencing a column this view doesn't
    declare in "metrics" (typo, wrong view, or a column with no known
    good/bad direction) is silently skipped, same fallthrough-safe pattern as
    the rest of the view layer -- a bad highlight request should never fail
    the whole query, just render fewer cards."""
    label_col = spec.get("label_col")
    allowed = set(spec.get("metrics") or [])
    if not requests or not label_col or label_col not in result_df.columns:
        return []

    highlights = []
    for req in requests:
        col, agg = req.get("column"), req.get("agg")
        if col not in allowed or col not in result_df.columns or col not in _METRIC_DIRECTION:
            continue
        series = result_df[col].dropna()
        if series.empty:
            continue
        idx = series.idxmax() if agg == "max" else series.idxmin()
        row = result_df.loc[idx]
        value = row[col]
        direction = _METRIC_DIRECTION[col]
        is_bad = (agg == "max" and direction == "high_bad") or (agg == "min" and direction == "low_bad")
        highlights.append({
            "label":  f"{'Highest' if agg == 'max' else 'Lowest'} {col}",
            "entity": str(row[label_col]),
            "value":  _format_metric_value(col, value),
            "bad":    is_bad,
        })
    return highlights


def _build_portfolio_kpis(spec: dict, result_df: pd.DataFrame) -> list:
    """Unconditional portfolio-wide rollup for every view that declares
    "metrics" -- e.g. "Avg NPA% 15.7%" alongside region_scorecard's per-region
    table. Unlike highlight_metrics/sort_by, this needs no LLM judgment call
    (which columns, mean vs sum, is static per-column knowledge in
    _METRIC_AGG), so it always renders regardless of what the planner asked
    for, giving portfolio-level context for whatever entities are ranked."""
    metrics = spec.get("metrics") or []
    if not metrics or result_df is None or result_df.empty:
        return []

    kpis = []
    for col in metrics:
        if col not in result_df.columns or col not in _METRIC_AGG:
            continue
        series = result_df[col].dropna()
        if series.empty:
            continue
        agg = _METRIC_AGG[col]
        value = series.mean() if agg == "mean" else series.sum()
        kpis.append({
            "label": f"{'Avg' if agg == 'mean' else 'Total'} {col}",
            "value": _format_metric_value(col, value),
        })
    return kpis


# ── Node 1b: Fast-Path View (deterministic, reuses cached analysis/ results) ──
# "If the query is already computed then just fetch that result, if not computed
# then do v2 ai query search normally" -- this node IS that fetch. It never fails
# hard: any reason a view can't serve the request (unknown name, missing df_prev,
# a filter on a non-filterable view, filter-expansion errors, a runtime error
# from the analysis/ function itself) clears ir1["view"] and routes back into the
# normal compile_and_validate_node -> execute_node path with the SAME ir1, so the
# query still gets answered via the general pipeline rather than failing outright.
@traceable(run_type="tool", name="ViewNode", tags=["node"], process_inputs=_trace_process_inputs, process_outputs=_trace_process_outputs)
def view_node(state: QueryState) -> QueryState:
    _announce("view")
    ir1 = state.get("ir1") or {}
    view_spec = ir1.get("view") or {}
    name = view_spec.get("name")
    spec = VIEWS.get(name)

    def _fallthrough(reason: str, new_state=None):
        # Reason is genuinely useful observability on its own: it tells you
        # whether the fast-path view layer is actually serving requests or
        # silently falling through most of the time (and why) -- invisible
        # before this pass, since a fallthrough looks identical to "no view
        # matched at all" from outside view_node.
        _trace_metadata({"view_name": name, "view_served": False, "view_fallthrough_reason": reason})
        s = new_state if new_state is not None else state
        return {**s, "ir1": {**ir1, "view": None}}

    if spec is None:
        return _fallthrough("no_view_matched")

    df_curr = _fetch_large(state, "result_df_full")
    if df_curr is None or len(df_curr) == 0:
        return _fallthrough("no_data")

    df_prev = _fetch_large(state, "df_prev")
    if "df_prev" in (spec.get("requires") or []) and (df_prev is None or len(df_prev) == 0):
        return _fallthrough("missing_required_prev_file")

    filters = view_spec.get("filters") or []
    if filters and not spec.get("filterable", True):
        # Planner attached a filter to a view that can't take one -- don't
        # silently drop the user's requested restriction, fall through instead.
        return _fallthrough("filter_on_unfilterable_view")

    param_specs = spec.get("params") or {}
    requested_params = view_spec.get("params") or {}
    non_default_params = {
        k: v for k, v in requested_params.items()
        if k in param_specs and v != param_specs[k].get("default")
    }

    precomputed = _fetch_large(state, "precomputed_views") or {}

    rr_meta = state.get("rr_meta")

    cache_hit = False
    try:
        if not filters and not non_default_params:
            # Reuse the literal cached object -- zero recomputation, guarantees
            # numeric identity with what the dashboard tabs already show.
            cache_key = spec.get("cache_key")
            if cache_key and cache_key in precomputed:
                raw = precomputed[cache_key]
                cache_hit = True
            else:
                raw = _call_view_fn(name, spec, df_curr, df_prev, {}, rr_meta)
        else:
            call_params = {k: d.get("default") for k, d in param_specs.items()}
            call_params.update(non_default_params)

            input_df, input_df_prev = df_curr, df_prev
            if filters:
                errs: list = []
                conditions = _expand_filters(filters, errs)
                if errs:
                    return _fallthrough("filter_expansion_error")
                input_df = df_curr[_build_mask(df_curr, conditions)]
                # Apply the SAME conditions to df_prev too -- a view that consumes
                # both (e.g. region_scorecard) must compare like-for-like (e.g.
                # "Pune this month" against "Pune last month", never "Pune this
                # month" against "the whole portfolio last month"). Conditions on
                # columns df_prev doesn't have (e.g. curr_bucket, which only
                # exists post-merge on df_curr) are gracefully skipped by
                # _apply_condition, not an error.
                if input_df_prev is not None and len(input_df_prev):
                    input_df_prev = input_df_prev[_build_mask(input_df_prev, conditions)]

            raw = _call_view_fn(name, spec, input_df, input_df_prev, call_params, rr_meta)
    except Exception:
        return _fallthrough("analysis_fn_raised")

    try:
        result_df, result_kpis, result_rankings = normalize_view_output(spec, raw)
    except Exception:
        return _fallthrough("output_normalization_failed")

    result_df = _apply_sort_by(result_df, view_spec.get("sort_by"))

    result_kpis = {"Count": len(result_df), **result_kpis}
    # analyze_node's insight generator only understands two kpis shapes: an
    # "aggregation result" (has "_agg_rows") or a "loan-level filter result"
    # (expects Total POS/Avg Arrears/EMI/etc, defaulting missing keys to 0).
    # View kpis never match the second shape, so without this it would silently
    # describe a KPI-card view (e.g. pulse_kpis) as "8 accounts with zero POS" --
    # a confidently fabricated narrative built from defaulted zeros. Always give
    # it real sample rows to describe instead, regardless of which view this is.
    if "_agg_rows" not in result_kpis and result_df is not None and len(result_df):
        # Customer/loan-grain views (top_accounts, fleet_exposure, repossession,
        # good_customers) return real Cust Name/Cust Mob No columns here -- the
        # Insight Generator only needs the analytical columns to describe the
        # pattern, so strip identity columns before this ever reaches a Gemini
        # prompt. See _PII_COLS_IN_AGG_ROWS's docstring above for the full story.
        result_kpis["_agg_rows"] = _strip_pii_from_agg_rows(result_df.head(5).to_dict(orient="records"))

    result_highlights = _build_highlights(
        spec, result_df, view_spec.get("highlight_metrics") or []
    )
    result_portfolio_kpis = _build_portfolio_kpis(spec, result_df)

    _trace_metadata({
        "view_name": name,
        "view_served": True,
        "view_cache_hit": cache_hit,
        "view_had_filters": bool(filters),
        "view_row_count": len(result_df),
        "highlight_metrics_count": len(result_highlights),
        "sort_by_requested": bool(view_spec.get("sort_by")),
    })

    return {
        **state,
        "query_title":       ir1.get("query_title") or spec.get("label") or "Custom Query",
        "enriched_query":    ir1.get("description") or state["query"],
        "priority_mode":     False,
        "aggregation_mode":  False,
        "plan_mode":         False,
        "plan":              [],
        "result_type":       "loan_table",
        "result_grain":      spec.get("grain", "loan"),
        "view_render":       spec.get("render") or "",
        "parsed_filters":    {"plain_english": ir1.get("description") or state["query"]},
        "result_df":         result_df,
        "result_kpis":       result_kpis,
        "result_rankings":   result_rankings,
        "result_highlights": result_highlights,
        "result_portfolio_kpis": result_portfolio_kpis,
        "error":             "",
    }


def _call_view_fn(name: str, spec: dict, df_curr, df_prev, call_params: dict, rr_meta=None):
    """Resolve and invoke a VIEWS entry's analysis/ function, supplying whichever
    inputs it declares (df_curr / df_prev / rr_meta, in order) plus any accepted
    params. A view with no fresh-callable inputs (e.g. good_bad_summary, which
    composes from OTHER views' outputs) will TypeError here on a cache miss --
    caught by view_node's try/except and treated as a normal fall-through."""
    fn = resolve_view_fn(name)
    args = []
    for inp in spec.get("inputs") or []:
        if inp == "df_curr":
            args.append(df_curr)
        elif inp == "df_prev":
            args.append(df_prev)
        elif inp == "rr_meta":
            args.append(rr_meta or {})
        else:
            raise ValueError(f"view '{name}': unsupported input '{inp}'")
    kwargs = {k: v for k, v in call_params.items() if v is not None}
    return fn(*args, **kwargs)


# ── Node 2: Compiler + Validator, with one-shot repair ────────────────────────
# Merged: compile_logical() already calls validate_plan() internally as its final
# gate (compiler/core.py), so a separate validate-only pass was largely redundant.
# The repair loop now covers BOTH failure classes (unknown concept/metric/entity
# from the compiler, AND unknown-column/structural errors from the validator) --
# previously only validator failures got a retry, so any compiler-stage error
# (e.g. "unknown metric 'curr_npa_count'") failed hard with zero chance to
# self-correct. This was the root cause behind several real query failures.
@traceable(run_type="tool", name="CompileAndValidateNode", tags=["node"], process_inputs=_trace_process_inputs, process_outputs=_trace_process_outputs)
def compile_and_validate_node(state: QueryState) -> QueryState:
    _announce("compile")
    ir1 = state.get("ir1") or {}
    df: pd.DataFrame = _fetch_large(state, "result_df_full")
    cols = list(df.columns) if df is not None and len(df) > 0 else []

    for attempt in range(_MAX_REPAIRS + 1):
        try:
            plan, errs = compile_logical(ir1, cols)
        except Exception as e:
            _trace_metadata({"repair_attempts": attempt, "compiled_ok": False, "compiler_exception": str(e)})
            return {**state, "error": f"Compiler failed: {e}"}

        if not errs:
            # repair_attempts tells you directly, per query, whether the
            # Logical Planner got it right first try or needed a repair --
            # previously invisible (query_log.py's classify_outcome only ever
            # recorded the end state "compiled_ok", the same label whether or
            # not a repair fired). Useful signal for whether prompt growth
            # (e.g. this session's highlight_metrics/sort_by additions) is
            # increasing how often the planner needs correcting.
            _trace_metadata({"repair_attempts": attempt, "compiled_ok": True})
            return {**state, "ir1": ir1, "plan": plan}

        err_msg = "; ".join(errs)
        if attempt == _MAX_REPAIRS:
            if "prev_bucket" in err_msg:
                err_msg += ". Tip: upload a previous-period file to enable snapshot comparisons."
            _trace_metadata({"repair_attempts": attempt, "compiled_ok": False, "compile_errors": err_msg})
            return {**state, "error": f"Could not compile query: {err_msg}"}

        # One repair attempt: re-run the planner with the compile/validate errors as context.
        feedback = (
            f"Your previous output was invalid. "
            f"Errors: {err_msg}. "
            f"The ONLY valid column names are: {', '.join(map(str, cols))}. "
            "Correct the IR so every column, concept, metric, entity, and dimension "
            "alias reference is valid. Return corrected JSON."
        )
        try:
            ir1 = plan_logical(
                state["query"], state.get("snapshot_dates"),
                repair_feedback=feedback, allow_clarification=False,
            )
        except Exception as e:
            return {**state, "error": f"Query repair failed: {e}"}

    return state  # unreachable -- loop always returns on its final attempt


# ── Node 4: Data Executor (pandas) ───────────────────────────────────────────
@traceable(run_type="tool", name="ExecuteNode", tags=["node"], process_inputs=_trace_process_inputs, process_outputs=_trace_process_outputs)
def execute_node(state: QueryState) -> QueryState:
    _announce("execute")
    df: pd.DataFrame = _fetch_large(state, "result_df_full")
    if df is None or len(df) == 0:
        return {**state, "error": "No data loaded."}

    ir1    = state.get("ir1") or {}
    intent = ir1.get("intent", "loan_table")
    # A single_value query with NO dimensions is a scalar filter (no GROUP BY produced).
    # Treat it like loan_table so we get full KPI cards and correct ranking cards.
    has_dims     = bool(ir1.get("dimensions"))
    is_true_agg  = intent == "aggregation" or (intent == "single_value" and has_dims)

    try:
        if intent == "priority_action":
            display_df, err = execute_priority_mode(df)
        else:
            display_df, err = execute_plan(df, state.get("plan") or [])

        if err:
            return {**state, "result_df": pd.DataFrame(), "error": err}

        if is_true_agg:
            # Merge MNT NAME + Unit into a single display column when grouping by executive,
            # matching the previous architecture's "MNT NAME (Unit)" combined display.
            dims = ir1.get("dimensions") or []
            if (
                "executive" in dims
                and "MNT NAME" in display_df.columns
                and "Unit" in display_df.columns
            ):
                display_df = display_df.copy()
                pos = display_df.columns.get_loc("MNT NAME")
                display_df.insert(pos, "MNT NAME (Unit)",
                                  display_df["MNT NAME"] + " (" + display_df["Unit"] + ")")
                display_df = display_df.drop(columns=["MNT NAME", "Unit"])

            # Grouped results are entity-level (region/branch/executive), not
            # customer-level, so this shouldn't carry PII in practice -- stripped
            # anyway for consistency with the fast-path view construction site
            # above, in case a future dimension ever groups by something
            # customer-identifying.
            kpis     = {"Count": len(display_df), "_agg_rows": _strip_pii_from_agg_rows(display_df.head(5).to_dict(orient="records"))}
            rankings = {}
        else:
            # Apply default curated columns when the planner didn't specify any.
            # Only do this for loan_table rows (not aggregation results, and NOT
            # priority_action -- execute_priority_mode already returns its own
            # curated display columns led by "Priority", which QUERY_DISPLAY_COLS
            # doesn't know about; overriding here silently dropped that column
            # and crashed the UI's later groupby("Priority", ...) with a KeyError).
            if intent != "priority_action" and not ir1.get("display_columns") and not display_df.empty:
                from agents.data_executor import QUERY_DISPLAY_COLS
                rank_col = ["Rank"] if "Rank" in display_df.columns else []
                keep = rank_col + [c for c in QUERY_DISPLAY_COLS if c in display_df.columns]
                if keep:
                    display_df = display_df[keep]

            # compute_result_kpis re-filters df_full by Loan No internally,
            # so it always has the full column set regardless of what select did.
            kpis     = compute_result_kpis(df, display_df)
            rankings = compute_contextual_rankings(df, display_df)

        _trace_metadata({"intent": intent, "result_row_count": len(display_df)})
        return {**state, "result_df": display_df, "result_kpis": kpis,
                "result_rankings": rankings, "error": ""}

    except Exception as e:
        return {**state, "result_df": pd.DataFrame(), "error": f"Data execution failed: {e}"}


# ── Node 5: Insight Generator (Gemini) ───────────────────────────────────────
@traceable(run_type="chain", name="AnalyzeNode", tags=["node"], process_inputs=_trace_process_inputs, process_outputs=_trace_process_outputs)
def analyze_node(state: QueryState) -> QueryState:
    if state.get("skip_insights"):
        # Don't announce the "writing AI observations" step at all when it's
        # being skipped -- showing that label while the Gemini call never
        # actually happens is misleading, implying work that isn't occurring.
        _trace_metadata({"insights_skipped": True})
        return {**state, "insights": ""}
    _announce("analyze")
    ir1 = state.get("ir1") or {}
    plain_english = ir1.get("description") or state.get("query") or ""
    try:
        insights = generate_insights(
            query=state["query"],
            plain_english=plain_english,
            kpis=state.get("result_kpis") or {},
            rankings=state.get("result_rankings") or {},
            insight_focus=state.get("insight_focus") or "",
        )
        return {**state, "insights": insights}
    except Exception as e:
        return {**state, "insights": f"• AI observations unavailable: {e}"}


# ── Error + clarify handlers ──────────────────────────────────────────────────
def error_node(state: QueryState) -> QueryState:
    return state


def clarify_node(state: QueryState) -> QueryState:
    """Terminal: carries the clarification question + options back to the UI."""
    return state


# ── Routing ───────────────────────────────────────────────────────────────────
def _route_planner(state: QueryState) -> str:
    if state.get("error"):
        return "error"
    if state.get("needs_clarification"):
        return "clarify"
    if state.get("priority_mode"):
        return "execute"   # priority mode bypasses compile + validate
    if (state.get("ir1") or {}).get("view"):
        return "view"       # fast-path: try a pre-computed answer first
    return "compile"


def _route_view(state: QueryState) -> str:
    if state.get("error"):
        return "error"
    # view_node clears ir1["view"] on any failure to serve the request -- that's
    # the "fall through to the normal path" signal, not an error.
    if (state.get("ir1") or {}).get("view") is None:
        return "compile"
    return "analyze"


def _route_compile(state: QueryState) -> str:
    return "error" if state.get("error") else "execute"


def _route_execute(state: QueryState) -> str:
    return "error" if state.get("error") else "analyze"


# ── Build graph ───────────────────────────────────────────────────────────────
_graph = StateGraph(QueryState)
_graph.add_node("planner",  logical_planner_node)
_graph.add_node("view",     view_node)
_graph.add_node("compile",  compile_and_validate_node)
_graph.add_node("execute",  execute_node)
_graph.add_node("analyze",  analyze_node)
_graph.add_node("clarify",  clarify_node)
_graph.add_node("error",    error_node)

_graph.add_edge(START, "planner")
_graph.add_conditional_edges(
    "planner", _route_planner,
    {"compile": "compile", "view": "view", "execute": "execute", "clarify": "clarify", "error": "error"},
)
_graph.add_conditional_edges("view",     _route_view,     {"analyze": "analyze", "compile": "compile", "error": "error"})
_graph.add_conditional_edges("compile",  _route_compile,  {"execute": "execute", "error": "error"})
_graph.add_conditional_edges("execute",  _route_execute,  {"analyze":  "analyze",  "error": "error"})
_graph.add_edge("analyze", END)
_graph.add_edge("clarify", END)
_graph.add_edge("error",   END)

_compiled = _graph.compile()


def run_query(
    query: str,
    df: pd.DataFrame,
    on_step: Optional[Callable[[str], None]] = None,
    snapshot_dates: Optional[dict] = None,
    allow_clarification: bool = True,
    df_prev: Optional[pd.DataFrame] = None,
    precomputed_views: Optional[dict] = None,
    alerts_curr: Optional[list] = None,
    alerts_prev: Optional[list] = None,
    rr_meta: Optional[dict] = None,
    skip_insights: bool = False,
) -> QueryState:
    """Run the IR-1 → [fast-path view | compiler] → executor → insights pipeline.

    on_step fires a human-readable label at each node for a live progress UI.
    df_prev/precomputed_views/alerts_curr/alerts_prev/rr_meta feed the fast-path
    view layer (registry/views.py) -- precomputed_views lets it reuse the SAME
    cached analysis/ results app.py already computed for the dashboard tabs,
    instead of recomputing independently.
    skip_insights=True skips the Insight Generator's Gemini call entirely
    (result_df/result_kpis/result_highlights/result_portfolio_kpis are all
    unaffected -- only the written narrative is skipped), for users who only
    want the table/KPIs and don't want to pay for or wait on that 2nd call.
    """
    run_id = str(_uuid.uuid4())

    # Stash the large objects OUTSIDE the traced QueryState (see _stash_large's
    # docstring above) -- initial only ever gets tiny placeholders for these,
    # regardless of how large the real file is. alerts_curr/alerts_prev are
    # included even though no current node reads them via state.get(...) --
    # each alert dict carries its own drilldown DataFrame (smart_alerts.py),
    # and LangGraph traces whatever sits IN the state regardless of whether
    # anything reads it back out.
    _stash_large("result_df_full", df)
    _stash_large("df_prev", df_prev if df_prev is not None else pd.DataFrame())
    _stash_large("precomputed_views", precomputed_views or {})
    _stash_large("alerts_curr", alerts_curr or [])
    _stash_large("alerts_prev", alerts_prev or [])
    # result_df is NOT stashed -- it's the query's actual OUTPUT (view_node/
    # execute_node still return it directly in state, same as before this fix).
    # It's typically a filtered SUBSET of the file (often much smaller than
    # result_df_full), and only present in state for the last 1-2 node
    # transitions rather than unconditionally on every node for every query --
    # a smaller, bounded risk that isn't worth the return-contract break this
    # caused for every direct caller/test of view_node/execute_node.

    initial: QueryState = {
        "query":            query,
        "result_df_full":   None,
        "snapshot_dates":   snapshot_dates or {},
        "df_prev":          None,
        "precomputed_views": {},
        "alerts_curr":      [],
        "alerts_prev":      [],
        "rr_meta":          rr_meta or {},
        "ir1":              {},
        "enriched_query":   "",
        "query_category":   "",
        "query_title":      "",
        "focus_kpis":       [],
        "insight_focus":    "",
        "risk_flag":        "medium",
        "priority_mode":    False,
        "aggregation_mode": False,
        "aggregation_spec": {},
        "plan_mode":        False,
        "plan":             [],
        "result_type":      "loan_table",
        "result_grain":     "loan",
        "view_render":      "",
        "allow_clarification":    allow_clarification,
        "needs_clarification":    False,
        "clarification_question": "",
        "clarification_options":  [],
        "skip_insights":          skip_insights,
        "parsed_filters":   {},
        "result_df":        pd.DataFrame(),
        "result_kpis":      {},
        "result_rankings":  {},
        "result_highlights": [],
        "result_portfolio_kpis": [],
        "insights":         "",
        "error":            "",
        "run_id":           run_id,
        "shadow":           {},
    }

    _tls.step_callback = on_step
    try:
        # LangGraph's OWN built-in tracing (separate from and not controlled by
        # this file's @traceable/process_inputs additions on the node wrappers)
        # serializes the FULL QueryState at every node transition whenever
        # LangSmith env vars are active. Confirmed via direct measurement on a
        # realistic 60k-row file: with the large fields (result_df_full,
        # df_prev, precomputed_views, result_df, alerts_curr, alerts_prev) sitting
        # directly in state, this produced 20-85MB trace payloads -- LangSmith's
        # limit is 20MB, so every one of those calls failed to upload and
        # RETRIED repeatedly, adding 60+ seconds of pure network-retry overhead
        # PER QUERY, entirely invisible in the LangSmith UI (which only ever
        # shows successful traces, never the silent failures). Fixed at the
        # root: those fields are now stashed via _stash_large/_fetch_large
        # (a thread-local side channel) instead of living in the traced state
        # at all, so tracing stays fully enabled -- no tracing_context/enabled
        # flag needed here anymore.
        result = _compiled.invoke(initial, config={"run_id": run_id})
    finally:
        _tls.step_callback = None
        # Without this, _tls.large_data keeps holding THIS query's DataFrames on
        # this thread indefinitely -- _fetch_large's fallback to state.get(...)
        # (documented as the path a direct node call, e.g. a test, relies on)
        # never actually triggers once anything has ever stashed data on this
        # thread, since it only checks "was this key ever stashed here," not
        # "was it stashed by the current call." Confirmed via direct repro:
        # stash a small df once, then call execute_node() with a state dict
        # carrying a DIFFERENT df, and the node silently uses the stale one.
        _tls.large_data = {}

    log_query_outcome(result)
    return result
