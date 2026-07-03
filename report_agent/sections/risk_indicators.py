import pandas as pd
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from analysis.portfolio_intelligence import compute_risk_indicators
from analysis.roll_rate import compute_roll_rate_matrix


def compute_risk_indicators_section(df_curr: pd.DataFrame, df_prev: pd.DataFrame = None) -> dict | None:
    """Early-warning signal table - same source as the dashboard's Risk Indicators card."""
    try:
        if df_curr is None or len(df_curr) == 0:
            return None

        has_prev = df_prev is not None and len(df_prev) > 0
        rr_meta = None
        if has_prev:
            _, rr_meta = compute_roll_rate_matrix(df_curr, df_prev)

        indicators = compute_risk_indicators(df_curr, df_prev if has_prev else df_curr.iloc[0:0], rr_meta)
        if not indicators:
            return None
        return {"indicators": indicators}
    except Exception:
        return None
