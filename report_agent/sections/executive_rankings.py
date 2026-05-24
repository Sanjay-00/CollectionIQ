import pandas as pd
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from analysis.executive_scorecard import compute_executive_scorecard


def compute_executive_rankings(df_curr: pd.DataFrame, df_prev: pd.DataFrame = None) -> dict | None:
    try:
        sc = compute_executive_scorecard(df_curr)
        if sc is None or len(sc) == 0:
            return None

        def _rows(df_slice):
            return [
                {
                    "name":        row["MNT NAME"],
                    "accounts":    int(row["Accounts"]),
                    "coll_pct":    float(row["Collection %"]),
                    "strike_rate": float(row["Strike Rate %"]),
                    "npa_pct":     float(row["NPA %"]),
                    "tier":        row["Tier"],
                }
                for _, row in df_slice.iterrows()
            ]

        top5 = _rows(sc[sc["Tier"] == "top"].head(5))
        bot5 = _rows(sc[sc["Tier"] == "bottom"].head(5))
        return {"top5": top5, "bottom5": bot5, "total_executives": len(sc)}
    except Exception:
        return None
