import pandas as pd
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from analysis.portfolio_intelligence import compute_repossession_list


def compute_repossession_section(df_curr: pd.DataFrame, df_prev: pd.DataFrame = None, n: int = 12) -> dict | None:
    """Repossession candidates: SMA-2/NPA accounts on recent (still collateral-valuable) loans,
    sorted by SOH so the highest-exposure candidates lead."""
    try:
        df = compute_repossession_list(df_curr)
        if df.empty:
            return None

        df = df.sort_values("SOH", ascending=False) if "SOH" in df.columns else df
        rows = df.head(n).to_dict("records")
        return {"rows": rows, "total": len(df)}
    except Exception:
        return None
