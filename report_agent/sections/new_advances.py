import pandas as pd
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from analysis.portfolio_intelligence import compute_new_advances


def compute_new_advances_section(df_curr: pd.DataFrame, df_prev: pd.DataFrame = None, curr_month: str = None) -> dict | None:
    """New business (advances) funded this reporting month, plus segment
    breakdown -- curr_month is threaded through as as_of so "this month" means
    the report's own reporting month, not wall-clock "today" (same reasoning
    as product_analysis/repossession's as_of). df_prev accepted only to match
    portfolio_analyzer.py's uniform (df_curr, df_prev, curr_month) section
    signature -- unused, since compute_new_advances sources last month's own
    advances from df_curr's own Ag_Date history, not a second file."""
    try:
        result = compute_new_advances(df_curr, as_of=curr_month)
        if not result or not result.get("accounts"):
            return None
        seg_df = result.get("segment")
        result = {**result, "segment": seg_df.to_dict("records") if seg_df is not None and not seg_df.empty else []}
        return result
    except Exception:
        return None
