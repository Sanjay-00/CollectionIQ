from typing import TypedDict, Any
import pandas as pd
from langgraph.graph import StateGraph, START, END

from agents.domain_expert import enrich_query
from agents.query_parser import parse_query
from agents.data_executor import execute_filters, execute_priority_mode, execute_aggregation, distribute_priority_accounts, compute_result_kpis, compute_contextual_rankings
from agents.insight_generator import generate_insights


class QueryState(TypedDict):
    # Input
    query: str
    result_df_full: Any         # full unfiltered DataFrame

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


# ── Agent 0: Domain Expert (Gemini) ──────────────────────────────────────────
def domain_expert_node(state: QueryState) -> QueryState:
    try:
        enriched = enrich_query(state["query"])
        return {
            **state,
            "enriched_query":   enriched.get("enriched_query", state["query"]),
            "query_category":   enriched.get("query_category", "general"),
            "query_title":      enriched.get("query_title", "Custom Query"),
            "focus_kpis":       enriched.get("focus_kpis", []),
            "insight_focus":    enriched.get("insight_focus", ""),
            "risk_flag":        enriched.get("risk_flag", "medium"),
            "priority_mode":    bool(enriched.get("priority_mode", False)),
            "aggregation_mode": bool(enriched.get("aggregation_mode", False)),
            "aggregation_spec": enriched.get("aggregation_spec") or {},
            "result_type":      enriched.get("result_type", "loan_table"),
            "error": "",
        }
    except Exception as e:
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
    try:
        # Use enriched_query so the parser gets domain-expert context
        parsed = parse_query(state["enriched_query"])
        return {**state, "parsed_filters": parsed, "error": ""}
    except Exception as e:
        return {**state, "parsed_filters": {}, "error": f"Query parsing failed: {e}"}


# ── Agent 2: Data Executor (pandas) ──────────────────────────────────────────
def execute_node(state: QueryState) -> QueryState:
    df: pd.DataFrame = state.get("result_df_full")
    if df is None or len(df) == 0:
        return {**state, "error": "No data loaded."}
    try:
        if state.get("priority_mode"):
            # Priority mode: bypass filter parser, apply full priority framework
            display_df, err = execute_priority_mode(df)
        elif state.get("aggregation_mode"):
            # Aggregation mode: GROUP BY + ratio computation — no row-level filter
            display_df, err = execute_aggregation(df, state.get("aggregation_spec", {}))
        else:
            display_df, err = execute_filters(df, state["parsed_filters"])

        if err:
            return {**state, "result_df": pd.DataFrame(), "error": err}

        if state.get("aggregation_mode"):
            # KPIs and rankings are not meaningful for aggregation results
            kpis     = {"Count": len(display_df)}
            rankings = {}
        else:
            kpis     = compute_result_kpis(df, display_df)
            rankings = compute_contextual_rankings(df, display_df)
        return {**state, "result_df": display_df, "result_kpis": kpis, "result_rankings": rankings, "error": ""}
    except Exception as e:
        return {**state, "result_df": pd.DataFrame(), "error": f"Data execution failed: {e}"}


# ── Agent 3: Insight Generator (Gemini) ──────────────────────────────────────
def analyze_node(state: QueryState) -> QueryState:
    try:
        insights = generate_insights(
            query=state["query"],
            plain_english=state["parsed_filters"].get("plain_english", state["enriched_query"]),
            kpis=state["result_kpis"],
            rankings=state["result_rankings"],
            insight_focus=state.get("insight_focus", ""),
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
    return "error" if state.get("error") else "execute"

def _route_execute(state: QueryState) -> str:
    return "error" if state.get("error") else "analyze"


# ── Build graph ───────────────────────────────────────────────────────────────
_graph = StateGraph(QueryState)
_graph.add_node("expert",  domain_expert_node)
_graph.add_node("parse",   parse_node)
_graph.add_node("execute", execute_node)
_graph.add_node("analyze", analyze_node)
_graph.add_node("error",   error_node)

_graph.add_edge(START, "expert")
_graph.add_conditional_edges("expert",  _route_expert,  {"parse":   "parse",   "error": "error"})
_graph.add_conditional_edges("parse",   _route_parse,   {"execute": "execute", "error": "error"})
_graph.add_conditional_edges("execute", _route_execute, {"analyze": "analyze", "error": "error"})
_graph.add_edge("analyze", END)
_graph.add_edge("error",   END)

_compiled = _graph.compile()


def run_query(query: str, df: pd.DataFrame) -> QueryState:
    initial: QueryState = {
        "query":            query,
        "result_df_full":   df,
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
    }
    return _compiled.invoke(initial)
