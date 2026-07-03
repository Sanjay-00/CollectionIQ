import pandas as pd
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from analysis.portfolio_intelligence import compute_fleet_exposure


def _mob_location_map(df_curr: pd.DataFrame) -> dict:
    """Cust Mob No -> most common (Region, Branch), for display only.

    compute_fleet_exposure groups purely by mobile number (a customer can span
    branches), so this is a report-layer-only lookup - it doesn't touch analysis/.
    """
    if "Cust Mob No" not in df_curr.columns:
        return {}
    has_region = "RegionName" in df_curr.columns
    has_branch = "Unit" in df_curr.columns

    def _mode(s: pd.Series) -> str:
        m = s.mode()
        return str(m.iat[0] if not m.empty else s.iloc[0])

    result = {}
    for mob, grp in df_curr.groupby("Cust Mob No"):
        result[mob] = (
            _mode(grp["RegionName"]) if has_region else "",
            _mode(grp["Unit"]) if has_branch else "",
        )
    return result


def compute_fleet_exposure_section(df_curr: pd.DataFrame, df_prev: pd.DataFrame = None) -> dict | None:
    try:
        result = compute_fleet_exposure(df_curr)
        if result["count"] == 0:
            return None

        loc_map = _mob_location_map(df_curr)
        top = []
        for _, r in result["top_df"].head(8).iterrows():
            region, branch = loc_map.get(r["Mobile"], ("", ""))
            top.append({
                "customer":  str(r["Customer"]),
                "region":    region,
                "branch":    branch,
                "loans":     int(r["Loans"]),
                "npa_loans": int(r["NPA Loans"]),
                "soh":       float(r["Total SOH (Cr)"]),
            })
        return {
            "count": result["count"],
            "total_soh_cr": result["total_soh_cr"],
            "npa_operators": result["npa_operators"],
            "top": top,
        }
    except Exception:
        return None
