import pandas as pd
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from analysis.portfolio_intelligence import compute_product_analysis


def compute_product_analysis_section(df_curr: pd.DataFrame, df_prev: pd.DataFrame = None, curr_month: str = None) -> dict | None:
    """Segment-wise NPA/SMA-2 breakdown only (fuel type, source, and vintage cohort omitted from the report).

    curr_month is still threaded through to compute_product_analysis (as as_of)
    even though this section doesn't use the vintage-cohort rows today -- so the
    report's own reporting month, not wall-clock "now", is what "post-dated
    agreement" exclusion is computed against if vintage is ever re-enabled here.
    """
    try:
        result = compute_product_analysis(df_curr, as_of=curr_month)
        seg_df = result.get("segment")
        if seg_df is None or seg_df.empty:
            return None
        return {"rows": seg_df.to_dict("records")}
    except Exception:
        return None
