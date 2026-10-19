"""
Field Executive Performance Scorecard
Groups by MNT NAME and computes per-executive collection metrics.
Performance tiers are quartile-based (relative to the dataset) - not hardcoded thresholds.
"""

import pandas as pd

from config import SCORECARD_MIN_ACCOUNTS
from utils import attach_prev_delinquency, order_unit_columns, unit_metrics


def compute_executive_scorecard(
    df: pd.DataFrame, min_accounts: int = SCORECARD_MIN_ACCOUNTS, df_prev: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Returns a DataFrame ranked by Collection % with a quartile Tier column.

    Columns: Executive (Branch), MNT NAME, Unit, Accounts, Delinquent,
             Delinquency %, [Prev Delinquency %, Δ Delinquency %], SMA-2,
             SMA-2 %, NPA, NPA %, NPA % (SOH), NPA SOH (Cr), Strike Rate %, Collection %,
             Hard Bucket %, [Roll Fwd %, Roll Bwd %], Total POS (L),
             Total SOH (L), Demand (L), Collected (L), Tier
    Every number comes from utils.unit_metrics (the shared definitions), in
    ONE grouped pass. Grouped by MNT NAME + Unit so the same executive in
    different branches appears separately; executives with fewer than
    min_accounts are excluded. Roll rates stay None (not 0.0) for an
    executive with no loan that has both months' bucket, e.g. a new joiner:
    "not enough history" is a different claim from "nothing got worse".
    """
    if "MNT NAME" not in df.columns:
        return pd.DataFrame()

    has_roll = "prev_bucket" in df.columns and "curr_bucket" in df.columns
    group_cols = ["MNT NAME", "Unit"] if "Unit" in df.columns else ["MNT NAME"]
    m = unit_metrics(df, group_cols, min_accounts=min_accounts)
    if m.empty:
        return pd.DataFrame()

    def _r1(col: str) -> pd.Series:
        return m[col].round(1)

    def _ratio1(num: pd.Series, den: pd.Series) -> pd.Series:
        # Straight to 1 decimal (not via unit_metrics' 2-decimal value): the
        # table has always rounded these ratios once.
        return (num / den.where(den > 0) * 100).round(1).fillna(0.0)

    sc = pd.DataFrame({
        "Executive (Branch)": [f"{n} ({b})" if b else str(n) for n, b in
                               zip(m["MNT NAME"], m["Unit"] if "Unit" in m.columns else [""] * len(m))],
        # Raw MNT NAME/Unit for callers that group/filter by branch
        # (investigator/steps.py) without re-parsing the display string.
        "MNT NAME": m["MNT NAME"].astype(str).values,
        "Unit": (m["Unit"].astype(str) if "Unit" in m.columns else pd.Series("", index=m.index)).values,
        "Accounts": m["Accounts"].values,
        "Strike Rate %": _r1("Strike%").values,
        "Collection %": _ratio1(m["Collected"], m["Demand"]).values,
        "Hard Bucket %": _r1("Hard Bucket%").values,
        "Delinquent": m["Delinquent"].values,
        "Delinquency %": _r1("Delinquency%").values,
    })
    if has_roll:
        sc["Roll Fwd %"] = [None if pd.isna(v) else v for v in m["Roll Fwd%"]]
        sc["Roll Bwd %"] = [None if pd.isna(v) else v for v in m["Roll Bwd%"]]
    sc["SMA-2 %"] = _ratio1(m["SMA-2"], m["Accounts"]).values
    sc["NPA %"] = _ratio1(m["NPA"], m["Accounts"]).values
    sc["NPA % (SOH)"] = _r1("NPA% (SOH)").values
    sc["NPA SOH (Cr)"] = m["NPA SOH (Cr)"].values
    sc["NPA"] = m["NPA"].values
    sc["SMA-2"] = m["SMA-2"].values
    sc["Total POS (L)"] = (m["POS"] / 100_000).round(2).values
    sc["Total SOH (L)"] = (m["SOH"] / 100_000).round(2).values
    sc["Demand (L)"] = (m["Demand"] / 100_000).round(2).values
    sc["Collected (L)"] = (m["Collected"] / 100_000).round(2).values

    sc = sc.sort_values("Collection %", ascending=False, kind="stable")
    sc["Tier"] = quartile_tier(sc["Collection %"])
    sc = sc.reset_index(drop=True)
    on = {"MNT NAME": "MNT NAME", "Unit": "Unit"} if "Unit" in df.columns else {"MNT NAME": "MNT NAME"}
    sc = attach_prev_delinquency(sc, df_prev, on, pct_col="Delinquency %", fuzzy="MNT NAME", decimals=1)
    return sc[order_unit_columns(sc.columns, ["Executive (Branch)", "MNT NAME", "Unit"])]


def quartile_tier(series: pd.Series) -> pd.Series:
    """top = >= 75th percentile, bottom = <= 25th percentile, mid = everyone else -
    relative to this dataset, not a hardcoded threshold."""
    q75 = series.quantile(0.75)
    q25 = series.quantile(0.25)

    def _tier(val):
        if val >= q75:
            return "top"
        if val <= q25:
            return "bottom"
        return "mid"

    return series.apply(_tier)


def rank_by_metric(scorecard_df: pd.DataFrame, metric_col: str) -> pd.DataFrame:
    """Re-rank an already-computed scorecard by a different metric column (e.g.
    "Strike Rate %" instead of the default "Collection %"), recomputing quartile
    tiers relative to that metric.

    Returns an independent re-sorted copy - does NOT change compute_executive_scorecard's
    own Collection%-based ranking/Tier, so existing Collection%-ranked consumers
    (report_agent's executive_rankings section, AI Query's executive_rankings
    view) are unaffected unless they explicitly opt in.
    """
    if scorecard_df is None or scorecard_df.empty or metric_col not in scorecard_df.columns:
        return scorecard_df
    ranked = scorecard_df.sort_values(metric_col, ascending=False).copy()
    ranked["Tier"] = quartile_tier(ranked[metric_col])
    return ranked.reset_index(drop=True)


