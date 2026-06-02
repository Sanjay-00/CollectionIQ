import pandas as pd
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from utils import _safe_pct


def compute_branch_performance(df_curr: pd.DataFrame, df_prev: pd.DataFrame = None) -> dict | None:
    try:
        if "Unit" not in df_curr.columns:
            return None
        grp = df_curr.groupby("Unit").agg(
            demand=("Net Collection Demand Inst+Exp+BC", "sum"),
            collection=("Month Collection (Excluding Reserve Collection)", "sum"),
            accounts=("Loan No", "nunique"),
        )
        grp["coll_pct"] = grp.apply(lambda r: _safe_pct(r["collection"], r["demand"]), axis=1)
        grp = grp.reset_index().sort_values("coll_pct", ascending=False)

        def _row(r):
            return {
                "branch":     r["Unit"],
                "accounts":   int(r["accounts"]),
                "coll_pct":   round(float(r["coll_pct"]), 1),
                "collection": round(float(r["collection"]) / 100_000, 2),
                "demand":     round(float(r["demand"]) / 100_000, 2),
            }

        top5 = [_row(r) for _, r in grp.head(5).iterrows()]
        bot5 = [_row(r) for _, r in grp.tail(5).sort_values("coll_pct").iterrows()]
        return {"top5": top5, "bottom5": bot5, "total_branches": len(grp)}
    except Exception:
        return None
