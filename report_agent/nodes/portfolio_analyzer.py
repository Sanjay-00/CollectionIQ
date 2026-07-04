from report_agent.state import ReportState
from report_agent.sections import (
    portfolio_health,
    verdict,
    risk_flags,
    risk_indicators,
    bucket_migration,
    npa_sma2_movement,
    branch_quadrant,
    concentration,
    region_scorecard,
    product_analysis,
    top_accounts,
    fleet_exposure,
    repossession,
    good_customers,
    branch_performance,
    executive_recovery,
    executive_rankings,
    executive_strike_rankings,
)

_SECTION_FN = {
    "portfolio_health":    lambda c, p: portfolio_health.compute_portfolio_health(c, p),
    "verdict":             lambda c, p: verdict.compute_verdict(c, p),
    "risk_flags":          lambda c, p: risk_flags.compute_risk_flags(c, p),
    "risk_indicators":     lambda c, p: risk_indicators.compute_risk_indicators_section(c, p),
    "bucket_migration":    lambda c, p: bucket_migration.compute_bucket_migration_section(c, p),
    "npa_sma2_movement":   lambda c, p: npa_sma2_movement.compute_npa_sma2_movement(c, p),
    "branch_quadrant":     lambda c, p: branch_quadrant.compute_branch_quadrant_section(c, p),
    "concentration":       lambda c, p: concentration.compute_concentration_section(c, p),
    "region_scorecard":    lambda c, p: region_scorecard.compute_region_scorecard_section(c, p),
    "product_analysis":    lambda c, p: product_analysis.compute_product_analysis_section(c, p),
    "top_accounts":        lambda c, p: top_accounts.compute_top_accounts_section(c, p),
    "fleet_exposure":      lambda c, p: fleet_exposure.compute_fleet_exposure_section(c, p),
    "repossession":        lambda c, p: repossession.compute_repossession_section(c, p),
    "good_customers":      lambda c, p: good_customers.compute_good_customers_section(c, p),
    "branch_performance":  lambda c, p: branch_performance.compute_branch_performance(c, p),
    "executive_recovery":  lambda c, p: executive_recovery.compute_executive_recovery_section(c, p),
    "executive_rankings":  lambda c, p: executive_rankings.compute_executive_rankings(c, p),
    "executive_strike_rankings": lambda c, p: executive_strike_rankings.compute_executive_strike_rankings(c, p),
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
