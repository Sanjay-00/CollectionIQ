import pandas as pd
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from smart_alerts import run_all_alerts


def compute_risk_flags(df_curr: pd.DataFrame, df_prev: pd.DataFrame = None) -> dict | None:
    try:
        alerts = run_all_alerts(df_curr)
        SEVERITY_RANK = {"critical": 0, "high": 1, "medium": 2}
        # Sort by severity then count; take top 3 non-clear alerts
        active = sorted(
            [a for a in alerts if a["count"] > 0],
            key=lambda a: (SEVERITY_RANK.get(a["severity"], 9), -a["count"])
        )[:3]
        return {
            "flags": [
                {
                    "title":    a["title"],
                    "subtitle": a["subtitle"],
                    "severity": a["severity"],
                    "count":    a["count"],
                    "pos":      a["pos"],
                    "action":   a["action"],
                    "icon":     a["icon"],
                }
                for a in active
            ]
        }
    except Exception:
        return None
