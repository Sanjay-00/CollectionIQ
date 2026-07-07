import pandas as pd
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from analysis.portfolio_intelligence import compute_fleet_exposure


def compute_fleet_exposure_section(df_curr: pd.DataFrame, df_prev: pd.DataFrame = None) -> dict | None:
    try:
        result = compute_fleet_exposure(df_curr)
        if result["count"] == 0:
            return None

        # Region/Unit (most common, since a fleet customer's loans can span
        # branches) now come straight from compute_fleet_exposure's own top_df --
        # this section used to maintain an independent, near-identical lookup
        # (_mob_location_map) purely for report display; that's a second
        # implementation of the same "which branch is this customer in" logic,
        # now unnecessary since analysis/ computes it once, for every consumer.
        top = []
        for _, r in result["top_df"].head(8).iterrows():
            top.append({
                "customer":  str(r["Customer"]),
                "region":    str(r.get("Region", "")),
                "branch":    str(r.get("Unit", "")),
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
