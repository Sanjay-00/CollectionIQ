import pandas as pd
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from analysis.portfolio_intelligence import compute_executive_recovery


def compute_executive_recovery_section(df_curr: pd.DataFrame, df_prev: pd.DataFrame = None) -> dict | None:
    """Who is actually rescuing accounts from risk buckets vs letting them slip -
    a different, behavior-based signal than plain collection%."""
    try:
        df = compute_executive_recovery(df_curr)
        if df.empty:
            return None

        top    = df.head(5).to_dict("records")
        bottom = df[df["Net Recovery"] < 0].tail(5).sort_values("Net Recovery").to_dict("records")
        return {"top": top, "bottom": bottom, "total": len(df)}
    except Exception:
        return None
