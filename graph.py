import threading
import uuid as _uuid
from typing import Any, Callable, Optional, TypedDict

import pandas as pd
from langgraph.graph import END, START, StateGraph

from agents.data_executor import (
    compute_contextual_rankings,
    compute_result_kpis,
    execute_priority_mode,
)
from agents.plan_executor import execute_plan, validate_plan
from agents.logical_planner import plan_logical
from agents.insight_generator import generate_insights
from compiler.core import compile_logical
from registry.semantic_model import resolve_dimension


# ── Per-thread step callback ──────────────────────────────────────────────────
_tls = threading.local()

_STEP_LABELS: dict[str, str] = {
    "planner":  "🧠  Query Planner: understanding your question",
    "compile":  "⚙️   Compiler: building the execution plan",
    "validate": "🔎  Validator: checking the plan against your data",
    "execute":  "⚡  Data Executor: computing the answer",
    "analyze":  "💡  Insight Generator: writing AI observations",
}

_MAX_REPAIRS = 1


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

    # Clarification
    allow_clarification: bool
    needs_clarification: bool
    clarification_question: str
    clarification_options: list

    # Kept for UI backward compat (plain_english display)
    parsed_filters: dict

    # Executor outputs
    result_df: Any
    result_kpis: dict
    result_rankings: dict

    # Insight output
    insights: str

    # Error
    error: str

    # LangSmith trace ID
    run_id: str

    # Shadow-mode comparison slot (kept for backward compat; no longer populated)
    shadow: dict


# ── Node 1: Logical Planner (Gemini) ─────────────────────────────────────────
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
            "needs_clarification":    bool(ir1.get("needs_clarification", False)),
            "clarification_question": ir1.get("clarification_question") or "",
            "clarification_options":  ir1.get("clarification_options") or [],
            "parsed_filters":    {"plain_english": ir1.get("description") or state["query"]},
            "error":             "",
        }
    except Exception as e:
        return {**state, "error": f"Query planning failed: {e}"}


# ── Node 2: Compiler (deterministic) ─────────────────────────────────────────
def compiler_node(state: QueryState) -> QueryState:
    _announce("compile")
    ir1 = state.get("ir1") or {}
    df: pd.DataFrame = state.get("result_df_full")
    cols = list(df.columns) if df is not None and len(df) > 0 else []
    try:
        plan, errs = compile_logical(ir1, cols)
    except Exception as e:
        return {**state, "error": f"Compiler failed: {e}"}
    if errs:
        err_msg = "; ".join(errs)
        if "prev_bucket" in err_msg:
            err_msg += ". Tip: upload a previous-period file to enable snapshot comparisons."
        return {**state, "error": f"Could not compile query: {err_msg}"}
    return {**state, "plan": plan}


# ── Node 3: Validator + one-shot repair ───────────────────────────────────────
def validate_node(state: QueryState) -> QueryState:
    _announce("validate")
    if state.get("priority_mode"):
        return state  # priority executor uses registry directly  -  no plan to validate

    df: pd.DataFrame = state.get("result_df_full")
    if df is None or len(df) == 0:
        return state
    cols = list(df.columns)

    for attempt in range(_MAX_REPAIRS + 1):
        errs = validate_plan(state.get("plan") or [], cols)
        if not errs:
            return state

        if attempt == _MAX_REPAIRS:
            msg = "; ".join(errs)
            if "prev_bucket" in msg:
                msg += ". Tip: upload a previous-period file to enable snapshot comparisons."
            return {**state, "error": f"Could not build a valid query for your data: {msg}"}

        # One repair attempt: re-run the planner with the validation errors as context.
        feedback = (
            f"Your previous output was invalid. "
            f"Errors: {'; '.join(errs)}. "
            f"The ONLY valid column names are: {', '.join(map(str, cols))}. "
            "Correct the IR so every column reference matches an exact column from that list. "
            "Return corrected JSON."
        )
        try:
            fixed_ir1 = plan_logical(
                state["query"], state.get("snapshot_dates"),
                repair_feedback=feedback, allow_clarification=False,
            )
            new_plan, compile_errs = compile_logical(fixed_ir1, cols)
            if compile_errs:
                return {**state, "error": f"Repair compile failed: {'; '.join(compile_errs)}"}
            state = {**state, "ir1": fixed_ir1, "plan": new_plan}
        except Exception as e:
            return {**state, "error": f"Query repair failed: {e}"}

    return state


# ── Node 4: Data Executor (pandas) ───────────────────────────────────────────
def execute_node(state: QueryState) -> QueryState:
    _announce("execute")
    df: pd.DataFrame = state.get("result_df_full")
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

            kpis     = {"Count": len(display_df), "_agg_rows": display_df.head(5).to_dict(orient="records")}
            rankings = {}
        else:
            # Apply default curated columns when the planner didn't specify any.
            # Only do this for loan_table rows (not aggregation results).
            if not ir1.get("display_columns") and not display_df.empty:
                from agents.data_executor import QUERY_DISPLAY_COLS
                rank_col = ["Rank"] if "Rank" in display_df.columns else []
                keep = rank_col + [c for c in QUERY_DISPLAY_COLS if c in display_df.columns]
                if keep:
                    display_df = display_df[keep]

            # compute_result_kpis re-filters df_full by Loan No internally,
            # so it always has the full column set regardless of what select did.
            kpis     = compute_result_kpis(df, display_df)
            rankings = compute_contextual_rankings(df, display_df)

        return {**state, "result_df": display_df, "result_kpis": kpis,
                "result_rankings": rankings, "error": ""}

    except Exception as e:
        return {**state, "result_df": pd.DataFrame(), "error": f"Data execution failed: {e}"}


# ── Node 5: Insight Generator (Gemini) ───────────────────────────────────────
def analyze_node(state: QueryState) -> QueryState:
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
    return "compile"


def _route_compile(state: QueryState) -> str:
    return "error" if state.get("error") else "validate"


def _route_validate(state: QueryState) -> str:
    return "error" if state.get("error") else "execute"


def _route_execute(state: QueryState) -> str:
    return "error" if state.get("error") else "analyze"


# ── Build graph ───────────────────────────────────────────────────────────────
_graph = StateGraph(QueryState)
_graph.add_node("planner",  logical_planner_node)
_graph.add_node("compile",  compiler_node)
_graph.add_node("validate", validate_node)
_graph.add_node("execute",  execute_node)
_graph.add_node("analyze",  analyze_node)
_graph.add_node("clarify",  clarify_node)
_graph.add_node("error",    error_node)

_graph.add_edge(START, "planner")
_graph.add_conditional_edges(
    "planner", _route_planner,
    {"compile": "compile", "execute": "execute", "clarify": "clarify", "error": "error"},
)
_graph.add_conditional_edges("compile",  _route_compile,  {"validate": "validate", "error": "error"})
_graph.add_conditional_edges("validate", _route_validate, {"execute":  "execute",  "error": "error"})
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
) -> QueryState:
    """Run the IR-1 → compiler → executor → insights pipeline.

    on_step fires a human-readable label at each node for a live progress UI.
    """
    run_id = str(_uuid.uuid4())
    initial: QueryState = {
        "query":            query,
        "result_df_full":   df,
        "snapshot_dates":   snapshot_dates or {},
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
        "allow_clarification":    allow_clarification,
        "needs_clarification":    False,
        "clarification_question": "",
        "clarification_options":  [],
        "parsed_filters":   {},
        "result_df":        pd.DataFrame(),
        "result_kpis":      {},
        "result_rankings":  {},
        "insights":         "",
        "error":            "",
        "run_id":           run_id,
        "shadow":           {},
    }

    _tls.step_callback = on_step
    try:
        result = _compiled.invoke(initial, config={"run_id": run_id})
    finally:
        _tls.step_callback = None

    return result
