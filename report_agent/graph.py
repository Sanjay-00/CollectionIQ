import os
import uuid as _uuid
import pandas as pd
from typing import Optional
from langgraph.graph import StateGraph, START, END

from report_agent.state import ReportState
from report_agent.nodes.portfolio_analyzer import portfolio_analyzer_node
from report_agent.nodes.risk_narrator import risk_narrator_node
from report_agent.nodes.report_builder import report_builder_node
from report_agent.nodes.email_dispatcher import email_dispatcher_node


def _route_after_analyze(state: ReportState) -> str:
    return END if state.get("error") else "narrator"


def _route_after_builder(state: ReportState) -> str:
    return "dispatcher" if (os.environ.get("SMTP_HOST", "") and state.get("email_to", "")) else END


_graph = StateGraph(ReportState)
_graph.add_node("analyzer",   portfolio_analyzer_node)
_graph.add_node("narrator",   risk_narrator_node)
_graph.add_node("builder",    report_builder_node)
_graph.add_node("dispatcher", email_dispatcher_node)

_graph.add_edge(START, "analyzer")
_graph.add_conditional_edges("analyzer", _route_after_analyze, {"narrator": "narrator", END: END})
_graph.add_edge("narrator",  "builder")
_graph.add_conditional_edges("builder",  _route_after_builder,  {"dispatcher": "dispatcher", END: END})
_graph.add_edge("dispatcher", END)

_compiled = _graph.compile()

ALL_SECTIONS = [
    "portfolio_health", "risk_flags", "bucket_migration",
    "branch_performance", "executive_rankings",
]


def run_report(
    df_curr: pd.DataFrame,
    df_prev: pd.DataFrame,
    curr_month: str,
    prev_month: Optional[str] = None,
    enabled_sections: Optional[list] = None,
    filters_applied: Optional[dict] = None,
    email_to: str = "",
) -> ReportState:
    run_id = str(_uuid.uuid4())
    initial: ReportState = {
        "df_curr":          df_curr,
        "df_prev":          df_prev,
        "curr_month":       curr_month,
        "prev_month":       prev_month,
        "enabled_sections": enabled_sections or ALL_SECTIONS,
        "filters_applied":  filters_applied or {},
        "section_data":     {},
        "executive_narrative": "",
        "action_plan":      "",
        "html_report":      "",
        "run_id":           run_id,
        "email_to":         email_to,
        "email_sent":       False,
        "email_error":      "",
        "error":            "",
    }
    return _compiled.invoke(initial, config={"run_id": run_id})
