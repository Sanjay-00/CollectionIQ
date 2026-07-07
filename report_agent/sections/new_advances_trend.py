import pandas as pd
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from analysis.portfolio_intelligence import compute_new_advances_trend, compute_new_advances_trend_chart
from report_agent.charts import fig_to_base64
from config import NEW_ADVANCES_REPORT_TREND_MONTHS


def compute_new_advances_trend_section(df_curr: pd.DataFrame, df_prev: pd.DataFrame = None, curr_month: str = None) -> dict | None:
    """New Advances Trend chart, embedded as base64 PNG (kaleido) -- same
    pattern as branch_quadrant/concentration's own report sections. Fixed at
    NEW_ADVANCES_REPORT_TREND_MONTHS (36) months, Monthly granularity: the
    report is a static printed document, not the dashboard's interactive
    window/granularity picker, so there's no toggle to preserve here."""
    try:
        trend_df = compute_new_advances_trend(df_curr, as_of=curr_month, months=NEW_ADVANCES_REPORT_TREND_MONTHS)
        if trend_df.empty:
            return None
        fig = compute_new_advances_trend_chart(trend_df, granularity="Monthly")
        image = fig_to_base64(fig, width=1100, height=420)
        if not image:
            return None
        return {"image": image, "months": len(trend_df)}
    except Exception:
        return None
