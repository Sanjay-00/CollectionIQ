import pandas as pd
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from analysis.portfolio_intelligence import (
    compute_region_scorecard, compute_branch_quadrant,
    compute_executive_recovery, compute_risk_indicators, compute_good_bad,
)
from analysis.roll_rate import compute_roll_rate_matrix


def compute_verdict(df_curr: pd.DataFrame, df_prev: pd.DataFrame = None) -> dict | None:
    """Pandas-only good/bad synthesis - same signal source as the dashboard's Good vs Bad card.

    Runs independently of the AI narrative so a lead still gets a data-driven
    verdict even when skip_ai is set or Gemini is unavailable.
    """
    try:
        has_prev = df_prev is not None and len(df_prev) > 0
        empty_prev = df_curr.iloc[0:0]

        region_df = compute_region_scorecard(df_curr, df_prev if has_prev else empty_prev)
        branch_df, _ = compute_branch_quadrant(df_curr)
        exec_df = compute_executive_recovery(df_curr)

        rr_meta = None
        if has_prev:
            _, rr_meta = compute_roll_rate_matrix(df_curr, df_prev)

        risk_indicators = compute_risk_indicators(df_curr, df_prev if has_prev else empty_prev, rr_meta)
        result = compute_good_bad(region_df, branch_df, risk_indicators, exec_df, has_prev)

        if not result["good"] and not result["bad"]:
            return None
        return result
    except Exception:
        return None
