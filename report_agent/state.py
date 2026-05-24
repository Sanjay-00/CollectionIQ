from typing import TypedDict, Any, Optional


class ReportState(TypedDict):
    # Inputs
    df_curr: Any
    df_prev: Any
    curr_month: str
    prev_month: Optional[str]
    enabled_sections: list
    filters_applied: dict

    # Node 1 — portfolio_analyzer
    section_data: dict

    # Node 2 — risk_narrator
    executive_narrative: str
    action_plan: str

    # Node 3 — report_builder
    html_report: str

    # Node 4 — email_dispatcher
    email_to: str
    email_sent: bool
    email_error: str

    # Error
    error: str
