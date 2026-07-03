import pandas as pd
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from analysis.portfolio_intelligence import compute_top_accounts


def compute_top_accounts_section(df_curr: pd.DataFrame, df_prev: pd.DataFrame = None, n: int = 10) -> dict | None:
    try:
        top_df, summary = compute_top_accounts(df_curr, n=n)
        if top_df.empty:
            return None

        rows = [
            {
                "loan_no": str(r.get("Loan No", "")),
                "customer": str(r.get("Cust Name", "")),
                "region": str(r.get("RegionName", "")),
                "branch": str(r.get("Unit", "")),
                "bucket": str(r.get("curr_bucket", "")),
                "soh": float(r.get("SOH", 0) or 0),
            }
            for _, r in top_df.iterrows()
        ]
        return {"rows": rows, "summary": summary, "n": len(rows)}
    except Exception:
        return None
