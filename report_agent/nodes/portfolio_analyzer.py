from report_agent.state import ReportState
from report_agent.sections import (
    portfolio_health,
    risk_flags,
    bucket_migration,
    branch_performance,
    executive_rankings,
)

_SECTION_FN = {
    "portfolio_health":   lambda c, p: portfolio_health.compute_portfolio_health(c, p),
    "risk_flags":         lambda c, p: risk_flags.compute_risk_flags(c, p),
    "bucket_migration":   lambda c, p: bucket_migration.compute_bucket_migration_section(c, p),
    "branch_performance": lambda c, p: branch_performance.compute_branch_performance(c, p),
    "executive_rankings": lambda c, p: executive_rankings.compute_executive_rankings(c, p),
}


def portfolio_analyzer_node(state: ReportState) -> ReportState:
    df_curr = state["df_curr"]
    df_prev = state["df_prev"]
    enabled = state.get("enabled_sections", list(_SECTION_FN.keys()))

    section_data = {}
    for name in enabled:
        fn = _SECTION_FN.get(name)
        if fn is None:
            continue
        try:
            result = fn(df_curr, df_prev)
            if result is not None:
                section_data[name] = result
        except Exception as e:
            section_data[f"{name}_error"] = str(e)

    return {**state, "section_data": section_data, "error": ""}
