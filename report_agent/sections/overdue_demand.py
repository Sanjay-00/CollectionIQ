import pandas as pd
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from analysis.portfolio_intelligence import compute_overdue_demand_scorecard, OVERDUE_DEMAND_IDENTITY_COLS

_DIM_LABEL = {"region": "Region", "branch": "Branch", "executive": "Executive"}
# Extra identity columns each dimension's rows carry -- single source of truth
# in analysis/portfolio_intelligence.py, shared with the dashboard table and
# the report renderer (see that constant's own docstring for why).
_DIM_IDENTITY_COLS = OVERDUE_DEMAND_IDENTITY_COLS


def compute_overdue_demand_section(df_curr: pd.DataFrame, df_prev: pd.DataFrame = None) -> dict | None:
    """Top 5 / bottom 5 by Month Demand Collection %, per dimension -- NOT the
    full region/branch/executive table. The dashboard tab shows every row
    (interactive, scrollable); the report is a static printed document, and
    printing every branch and every executive (often hundreds of rows) was
    responsible for roughly half this report's total row count and a real,
    user-reported slowdown generating it. Same top5/bottom5 cap every other
    league-table-shaped report section already uses (see branch_performance.py).
    """
    try:
        data = compute_overdue_demand_scorecard(df_curr)
        if not data or all(df.empty for df in data.values()):
            return None

        result = {}
        for key, df in data.items():
            if df.empty:
                continue
            label = _DIM_LABEL[key]
            identity_cols = _DIM_IDENTITY_COLS[key]
            ranked = df.sort_values("Month Demand Collection %", ascending=False)

            def _row(r):
                row = {"name": str(r[label])}
                for ic in identity_cols:
                    row[ic.lower()] = str(r.get(ic, "") or "")
                row.update({
                    "accounts":    int(r["Accounts"]),
                    "overdue_cr":  round(float(r["Overdue (Cr)"]), 2),
                    "demand_cr":   round(float(r["Month Demand (Cr)"]), 2),
                    "overdue_pct": round(float(r["Overdue Collection %"]), 2),
                    "demand_pct":  round(float(r["Month Demand Collection %"]), 2),
                })
                return row

            top5 = [_row(r) for _, r in ranked.head(5).iterrows()]
            # .tail(5) then re-sort ascending so "worst first" reads top-to-bottom,
            # same convention branch_performance.py's bottom5 already uses.
            bottom5_df = ranked.tail(5).sort_values("Month Demand Collection %")
            # Fewer than 10 total entities means top5/bottom5 overlap -- drop
            # duplicates (by name) so the same entity never appears in both lists.
            bottom5_df = bottom5_df[~bottom5_df[label].isin([r["name"] for r in top5])]
            bottom5 = [_row(r) for _, r in bottom5_df.iterrows()]

            result[key] = {"top5": top5, "bottom5": bottom5, "total": len(df)}

        return result if result else None
    except Exception:
        return None
