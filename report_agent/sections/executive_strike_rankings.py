import pandas as pd
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from analysis.executive_scorecard import compute_executive_scorecard, rank_by_metric


def compute_executive_strike_rankings(df_curr: pd.DataFrame, df_prev: pd.DataFrame = None) -> dict | None:
    """Same executive pool as executive_rankings, re-ranked by Strike Rate % instead
    of Collection % - who is actually current on their installment obligation this
    month, not just who collected the most (which can be padded by reserve/
    settlement-driven recoveries). Independent of executive_rankings.py's
    Collection%-based ranking; both can be included in a report side by side."""
    try:
        sc = compute_executive_scorecard(df_curr)
        if sc is None or len(sc) == 0:
            return None
        ranked = rank_by_metric(sc, "Strike Rate %")

        def _rows(df_slice):
            return [
                {
                    "name":        row["Executive (Branch)"],
                    "accounts":    int(row["Accounts"]),
                    "strike_rate": float(row["Strike Rate %"]),
                    "coll_pct":    float(row["Collection %"]),
                    "npa_pct":     float(row.get("NPA %", 0)),
                    "tier":        row["Tier"],
                }
                for _, row in df_slice.iterrows()
            ]

        top5 = _rows(ranked[ranked["Tier"] == "top"].head(5))
        bot5 = _rows(ranked[ranked["Tier"] == "bottom"].head(5))
        return {"top5": top5, "bottom5": bot5, "total_executives": len(ranked)}
    except Exception:
        return None
