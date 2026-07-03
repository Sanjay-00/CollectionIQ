import pandas as pd
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from analysis.portfolio_intelligence import compute_region_scorecard


def compute_region_scorecard_section(df_curr: pd.DataFrame, df_prev: pd.DataFrame = None) -> dict | None:
    try:
        has_prev = df_prev is not None and len(df_prev) > 0
        df = compute_region_scorecard(df_curr, df_prev if has_prev else df_curr.iloc[0:0])
        if df.empty:
            return None
        return {"rows": df.to_dict("records"), "has_prev": has_prev}
    except Exception:
        return None
