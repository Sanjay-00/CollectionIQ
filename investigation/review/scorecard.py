"""
The region scorecard (spec 5.1): one row per region, every figure from an
engine the rest of the app already uses, so it reconciles with the Dashboard
and Portfolio Intelligence by construction. Rows are sorted by name: ranking
is the lists' job (prioritise.py), not the scorecard's.
"""
from __future__ import annotations

import pandas as pd

from analysis.new_business import compute_new_advances_by_dimension
from analysis.portfolio_intelligence import compute_region_scorecard
from investigation.metrics import get_metric
from utils import unit_metrics

_NEW_BUSINESS = {"Accounts This Month": "New Accounts", "Funded (Cr)": "New Funded (Cr)",
                 "Accounts MoM %": "New Accounts MoM %"}


def _thirty_plus_by_region(df: pd.DataFrame) -> pd.Series:
    """The investigation's 30+ rate (eligible base) per region, in %."""
    m = get_metric("thirty_plus_rate")
    eligible, numerator = m.eligible(df), m.numerator(df)
    g = pd.DataFrame({"r": df["RegionName"], "n": eligible, "k": numerator}).groupby("r").sum()
    return (100 * g["k"] / g["n"].where(g["n"] > 0)).round(2)


def region_scorecard(curr: pd.DataFrame, prev: pd.DataFrame, curr_month: str) -> pd.DataFrame:
    """curr: this month with last month's bucket carried (the app's month
    join); prev: last month's file. Dashboard columns come straight from
    compute_region_scorecard; added: Δ Collection%, 30+% and its change,
    share of book SOH, and new business."""
    base = compute_region_scorecard(curr, prev)
    if base.empty:
        return base
    out = base.copy()
    coll_prev = unit_metrics(prev, ["RegionName"]).set_index("RegionName")["Collection%"] if len(prev) else None
    if coll_prev is not None:
        out["Δ Collection%"] = (out["Collection%"] - out["Region"].map(coll_prev)).round(2)
    tp_curr, tp_prev = _thirty_plus_by_region(curr), _thirty_plus_by_region(prev)
    out["30+%"] = out["Region"].map(tp_curr)
    out["Δ 30+%"] = (out["30+%"] - out["Region"].map(tp_prev)).round(2)
    total_soh = out["SOH (Cr)"].sum()
    out["Share of SOH %"] = (100 * out["SOH (Cr)"] / total_soh).round(2) if total_soh else None
    as_of = pd.Period(curr_month, "M").to_timestamp(how="end").normalize()
    adv = compute_new_advances_by_dimension(curr, as_of=as_of).get("region")
    if adv is not None and not adv.empty:
        adv = adv.set_index("Region")
        for src, dst in _NEW_BUSINESS.items():
            if src in adv.columns:
                out[dst] = out["Region"].map(adv[src])
    return out.sort_values("Region", kind="stable").reset_index(drop=True)
