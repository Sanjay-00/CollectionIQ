import pandas as pd
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from analysis.portfolio_intelligence import compute_new_advances_by_dimension
from config import NEW_ADVANCES_REPORT_TOP_N


def compute_new_advances_by_dimension_section(df_curr: pd.DataFrame, df_prev: pd.DataFrame = None, curr_month: str = None) -> dict | None:
    """Top N (NEW_ADVANCES_REPORT_TOP_N, 5) Regions / Branches / Executives by
    New Advances this month -- NOT every row, same "printed document" reasoning
    branch_performance.py's own top5/bottom5 cap already uses. Each grain's own
    DataFrame from compute_new_advances_by_dimension is already sorted by
    Accounts This Month descending, so .head(N) is the top N by construction,
    no bottom-N mirror needed (this is a business volume ranking, not a
    performance/concern one)."""
    try:
        data = compute_new_advances_by_dimension(df_curr, as_of=curr_month)
        result = {}
        for key in ("region", "branch", "executive"):
            df = data.get(key, pd.DataFrame())
            if df.empty:
                continue
            result[key] = df.head(NEW_ADVANCES_REPORT_TOP_N).to_dict("records")
        return result if result else None
    except Exception:
        return None
