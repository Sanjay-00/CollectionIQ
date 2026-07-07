import pandas as pd
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from analysis.portfolio_intelligence import compute_branch_quadrant
from report_agent.charts import fig_to_base64


def compute_branch_quadrant_section(df_curr: pd.DataFrame, df_prev: pd.DataFrame = None) -> dict | None:
    """Collection% vs NPA% branch scatter (bubble=SOH) plus the top-5 highest-concern branches
    as a text fallback for email clients that block images."""
    try:
        df, fig = compute_branch_quadrant(df_curr)
        if df.empty:
            return None

        image = fig_to_base64(fig, width=1100, height=470)
        # Ranking is still by Concern Score (df is already sorted that way, and
        # Rank reflects it) -- the score itself just isn't shown as a column,
        # per business request: it's an internal composite, not something a
        # report reader needs to interpret on its own.
        top_concern = df.head(5)[[
            "Rank", "Branch", "Region", "Accounts", "SMA-2%", "NPA%",
            "Collection%", "Strike%", "Roll Fwd%", "Chronic (3M+)", "SOH (Cr)",
        ]].to_dict("records")
        return {"image": image, "top_concern": top_concern, "total_branches": len(df)}
    except Exception:
        return None
