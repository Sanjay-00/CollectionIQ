import pandas as pd
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from analysis.portfolio_intelligence import compute_product_analysis


def compute_product_analysis_section(df_curr: pd.DataFrame, df_prev: pd.DataFrame = None) -> dict | None:
    """Segment-wise NPA/SMA-2 breakdown only (fuel type, source, and vintage cohort omitted from the report)."""
    try:
        result = compute_product_analysis(df_curr)
        seg_df = result.get("segment")
        if seg_df is None or seg_df.empty:
            return None
        return {"rows": seg_df.to_dict("records")}
    except Exception:
        return None
