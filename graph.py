import threading
import uuid as _uuid
from typing import Any, Callable, Optional, TypedDict

import pandas as pd
from langgraph.graph import END, START, StateGraph

from agents.data_executor import (
    compute_contextual_rankings,
    compute_result_kpis,
    distribute_priority_accounts,
    execute_aggregation,
    execute_filters,
    execute_priority_mode,
    validate_aggregation_spec,
    validate_filter_spec,
)
from agents.domain_expert import enrich_query
from agents.insight_generator import generate_insights
from agents.query_parser import parse_query


# ── Per-thread step callback ──────────────────────────────────────────────────
# Each Streamlit user session runs in its own thread, so threading.local()
# isolates callbacks correctly under concurrent load.
_tls = threading.local()

_STEP_LABELS: dict[str, str] = {
    "expert":   "🧠  Domain Expert: enriching query with NBFC context",
    "parse":    "📋  Query Parser: translating to filter conditions",
    "validate": "🔎  Validator: checking the query against your data",
    "execute":  "⚡  Data Executor: applying filters to portfolio",
    "analyze":  "💡  Insight Generator: writing AI observations",
}

# How many times the validator may send a bad spec back to the LLM for repair.
_MAX_REPAIRS = 1


def _announce(node: str) -> None:
    """Fire the step callback, if one is registered for this thread."""
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
    result_df_full: Any         # full unfiltered DataFrame
    snapshot_dates: dict        # {"curr": "YYYY-MM-DD", "prev": "YYYY-MM-DD"} - maps files to dated bucket columns

    # Agent 0 output — Domain Expert
    enriched_query: str
    query_category: str
    query_title: str
    focus_kpis: list
    insight_focus: str
    risk_flag: str
    priority_mode: bool
    aggregation_mode: bool
    aggregation_spec: dict
    result_type: str

    # Agent 1 output — Query Parser
    parsed_filters: dict

    # Agent 2 output — Data Executor
    result_df: Any
    result_kpis: dict
    result_rankings: dict

    # Agent 3 output — Insight Generator
    insights: str

    # Error
    error: str

    # LangSmith trace ID (set by run_query, returned to caller for user feedback)
    run_id: str


# ── Agent 0: Domain Expert (Gemini) ──────────────────────────────────────────
def domain_expert_node(state: QueryState) -> QueryState:
    _announce("expert")
    try:
        enriched = enrich_query(state["query"], state.get("snapshot_dates"))
        return {
            **state,
            "enriched_query":   enriched.get("enriched_query") or state["query"],
            "query_category":   enriched.get("query_category") or "general",
            "query_title":      enriched.get("query_title") or "Custom Query",
            "focus_kpis":       enriched.get("focus_kpis") or [],
            "insight_focus":    enriched.get("insight_focus") or "",
            "risk_flag":        enriched.get("risk_flag") or "medium",
            "priority_mode":    bool(enriched.get("priority_mode", False)),
            "aggregation_mode": bool(enriched.get("aggregation_mode", False)),
            "aggregation_spec": enriched.get("aggregation_spec") or {},
            "result_type":      enriched.get("result_type") or "loan_table",
            "error": "",
        }
    except Exception:
        # Non-fatal — fall back to raw query
        return {
            **state,
            "enriched_query":   state["query"],
            "query_category":   "general",
            "query_title":      "Custom Query",
            "focus_kpis":       [],
            "insight_focus":    "",
            "risk_flag":        "medium",
            "priority_mode":    False,
            "aggregation_mode": False,
            "aggregation_spec": {},
            "result_type":      "loan_table",
            "error": "",
        }


# ── Agent 1: Query Parser (Gemini) ───────────────────────────────────────────
def parse_node(state: QueryState) -> QueryState:
    _announce("parse")
    try:
        parsed = parse_query(state["enriched_query"], state.get("snapshot_dates"))
        return {**state, "parsed_filters": parsed, "error": ""}
    except Exception as e:
        return {**state, "parsed_filters": {}, "error": f"Query parsing failed: {e}"}


# ── Validator + repair (pandas check, then one LLM repair attempt) ────────────
def validate_node(state: QueryState) -> QueryState:
    """Check the generated spec against the ACTUAL columns before executing.
    On a mismatch (hallucinated column / undefined metric alias), send the error
    back to the LLM once for a corrected spec. Priority mode is skipped — its
    rules are static and already guarded inside the executor."""
    _announce("validate")
    if state.get("priority_mode"):
        return state

    df: pd.DataFrame = state.get("result_df_full")
    if df is None or len(df) == 0:
        return state
    cols = list(df.columns)

    for attempt in range(_MAX_REPAIRS + 1):
        if state.get("aggregation_mode"):
            errs = validate_aggregation_spec(state.get("aggregation_spec") or {}, cols)
        else:
            errs = validate_filter_spec(state.get("parsed_filters") or {}, cols)

        if not errs:
            return state

        if attempt == _MAX_REPAIRS:
            msg = "; ".join(errs)
            if "prev_bucket" in msg:
                msg += ". Tip: upload a previous-period file to enable snapshot comparisons."
            return {**state, "error": f"Could not build a valid query for your data: {msg}"}

        feedback = (
            "Your previous output referenced columns/names that are NOT in the dataset. "
            f"Errors: {'; '.join(errs)}. "
            f"The ONLY valid column names are: {', '.join(map(str, cols))}. "
            "Return corrected JSON using ONLY these exact column names."
        )
        try:
            if state.get("aggregation_mode"):
                fixed = enrich_query(state["query"], state.get("snapshot_dates"), repair_feedback=feedback)
                state = {**state, "aggregation_spec": fixed.get("aggregation_spec") or {}}
            else:
                fixed = parse_query(state["enriched_query"], state.get("snapshot_dates"), repair_feedback=feedback)
                state = {**state, "parsed_filters": fixed}
        except Exception as e:
            return {**state, "error": f"Query repair failed: {e}"}

    return state


# ── Agent 2: Data Executor (pandas) ──────────────────────────────────────────
def execute_node(state: QueryState) -> QueryState:
    _announce("execute")
    df: pd.DataFrame = state.get("result_df_full")
    if df is None or len(df) == 0:
        return {**state, "error": "No data loaded."}
    try:
        if state.get("priority_mode"):
            display_df, err = execute_priority_mode(df)
        elif state.get("aggregation_mode"):
            display_df, err = execute_aggregation(df, state.get("aggregation_spec") or {})
        else:
            display_df, err = execute_filters(df, state["parsed_filters"])

        if err:
            return {**state, "result_df": pd.DataFrame(), "error": err}

        if state.get("aggregation_mode"):
            kpis = {"Count": len(display_df), "_agg_rows": display_df.head(5).to_dict(orient="records")}
            rankings = {}
        else:
            kpis     = compute_result_kpis(df, display_df)
            rankings = compute_contextual_rankings(df, display_df)
        return {**state, "result_df": display_df, "result_kpis": kpis, "result_rankings": rankings, "error": ""}
    except Exception as e:
        return {**state, "result_df": pd.DataFrame(), "error": f"Data execution failed: {e}"}


# ── Agent 3: Insight Generator (Gemini) ──────────────────────────────────────
def analyze_node(state: QueryState) -> QueryState:
    _announce("analyze")
    try:
        insights = generate_insights(
            query=state["query"],
            plain_english=(state.get("parsed_filters") or {}).get("plain_english") or state.get("enriched_query") or "",
            kpis=state.get("result_kpis") or {},
            rankings=state.get("result_rankings") or {},
            insight_focus=state.get("insight_focus") or "",
        )
        return {**state, "insights": insights}
    except Exception as e:
        return {**state, "insights": f"• AI observations unavailable: {e}"}


# ── Error handler ─────────────────────────────────────────────────────────────
def error_node(state: QueryState) -> QueryState:
    return state


# ── Routing ───────────────────────────────────────────────────────────────────
def _route_expert(state: QueryState) -> str:
    return "error" if state.get("error") else "parse"

def _route_parse(state: QueryState) -> str:
    return "error" if state.get("error") else "validate"

def _route_validate(state: QueryState) -> str:
    return "error" if state.get("error") else "execute"

def _route_execute(state: QueryState) -> str:
    return "error" if state.get("error") else "analyze"


# ── Build graph ───────────────────────────────────────────────────────────────
_graph = StateGraph(QueryState)
_graph.add_node("expert",   domain_expert_node)
_graph.add_node("parse",    parse_node)
_graph.add_node("validate", validate_node)
_graph.add_node("execute",  execute_node)
_graph.add_node("analyze",  analyze_node)
_graph.add_node("error",    error_node)

_graph.add_edge(START, "expert")
_graph.add_conditional_edges("expert",   _route_expert,   {"parse":    "parse",    "error": "error"})
_graph.add_conditional_edges("parse",    _route_parse,    {"validate": "validate", "error": "error"})
_graph.add_conditional_edges("validate", _route_validate, {"execute":  "execute",  "error": "error"})
_graph.add_conditional_edges("execute",  _route_execute,  {"analyze":  "analyze",  "error": "error"})
_graph.add_edge("analyze", END)
_graph.add_edge("error",   END)

_compiled = _graph.compile()


def run_query(
    query: str,
    df: pd.DataFrame,
    on_step: Optional[Callable[[str], None]] = None,
    snapshot_dates: Optional[dict] = None,
) -> QueryState:
    """Run the 4-agent query pipeline.

    on_step, if provided, is called with a human-readable label at the start of
    each pipeline stage — use it to drive a progress UI (e.g. st.status).
    The callback fires on the same thread that calls run_query, so Streamlit's
    st.status().write() works safely from inside it.
    """
    run_id = str(_uuid.uuid4())
    initial: QueryState = {
        "query":            query,
        "result_df_full":   df,
        "snapshot_dates":   snapshot_dates or {},
        "enriched_query":   "",
        "query_category":   "",
        "query_title":      "",
        "focus_kpis":       [],
        "insight_focus":    "",
        "risk_flag":        "medium",
        "priority_mode":    False,
        "aggregation_mode": False,
        "aggregation_spec": {},
        "result_type":      "loan_table",
        "parsed_filters":   {},
        "result_df":        pd.DataFrame(),
        "result_kpis":      {},
        "result_rankings":  {},
        "insights":         "",
        "error":            "",
        "run_id":           run_id,
    }

    _tls.step_callback = on_step
    try:
        return _compiled.invoke(initial, config={"run_id": run_id})
    finally:
        _tls.step_callback = None
