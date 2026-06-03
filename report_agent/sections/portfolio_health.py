import pandas as pd
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from utils import compute_metrics, fmt_value


def compute_portfolio_health(df_curr: pd.DataFrame, df_prev: pd.DataFrame) -> dict | None:
    try:
        metrics = compute_metrics(df_curr, df_prev)
        KINDS = {
            "Month Demand": "money", "Total Collection": "money", "Collection %": "pct",
            "Strike %": "pct", "NPA %": "pct", "Hard Bucket %": "pct",
            "Count": "count", "POS": "money", "LCC%": "pct", "CMD %": "pct",
        }

        def _traffic_light(key, val):
            if key == "Collection %":
                return "green" if val >= 90 else "amber" if val >= 75 else "red"
            if key == "NPA %":
                return "green" if val <= 3 else "amber" if val <= 7 else "red"
            if key == "Hard Bucket %":
                return "green" if val <= 10 else "amber" if val <= 20 else "red"
            if key == "Strike %":
                return "green" if val >= 80 else "amber" if val >= 60 else "red"
            return "neutral"

        prev_metrics = compute_metrics(df_prev, df_prev.iloc[0:0]) if len(df_prev) > 0 else {}

        kpis_out = {}
        for k, (val, mom) in metrics.items():
            kind     = KINDS.get(k, "count")
            prev_val = prev_metrics[k][0] if k in prev_metrics else None
            kpis_out[k] = {
                "value":          val,
                "prev_value":     prev_val,
                "formatted":      fmt_value(val, kind),
                "prev_formatted": fmt_value(prev_val, kind) if prev_val is not None else "—",
                "mom":            mom,
                "traffic":        _traffic_light(k, val),
            }
        return {"kpis": kpis_out, "total_accounts": df_curr["Loan No"].nunique() if "Loan No" in df_curr.columns else len(df_curr)}
    except Exception:
        return None
