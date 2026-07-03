"""
Field Executive Performance Scorecard
Groups by MNT NAME and computes per-executive collection metrics.
Performance tiers are quartile-based (relative to the dataset) - not hardcoded thresholds.
"""
import pandas as pd

from config import SCORECARD_MIN_ACCOUNTS
from utils import BUCKET_SCORE, to_num, account_count, compute_strike_pct


def compute_executive_scorecard(df: pd.DataFrame, min_accounts: int = SCORECARD_MIN_ACCOUNTS) -> pd.DataFrame:
    """
    Returns a DataFrame ranked by collection_pct with performance_tier column.

    Columns: Executive (Branch), Accounts, Strike Rate %, Collection %,
             [Roll Fwd %, Roll Bwd %], NPA, SMA-2, Total POS (L), Total SOH (L),
             Demand (L), Collected (L), Tier
    Groups by MNT NAME + Unit so the same executive in different branches appears separately.
    Executives with fewer than min_accounts are excluded.
    """
    if "MNT NAME" not in df.columns:
        return pd.DataFrame()

    has_roll = "prev_bucket" in df.columns and "curr_bucket" in df.columns

    group_cols = ["MNT NAME", "Unit"] if "Unit" in df.columns else ["MNT NAME"]

    rows = []
    for keys, grp in df.groupby(group_cols):
        if isinstance(keys, tuple):
            exec_name, branch = str(keys[0]), str(keys[1])
        else:
            exec_name, branch = str(keys), ""

        n = account_count(grp)
        if n < min_accounts:
            continue

        # Strike rate = % of accounts where full EMI payment was received this month.
        # Uses the shared utils.compute_strike_pct so this doesn't drift from the
        # dashboard/Portfolio Intelligence definition again (this was a 5th
        # independent implementation, missed by the earlier strike_pct/hard_bucket_pct
        # consolidation - it lacked the case/whitespace normalization compute_strike_pct
        # applies, silently undercounting the denominator for lowercase/padded "y"/"n").
        strike_rate = round(compute_strike_pct(grp), 1)

        demand    = to_num(grp, "Net Collection Demand Inst+Exp+BC").sum()
        collected = to_num(grp, "Month Collection (Excluding Reserve Collection)").sum()
        total_soh = to_num(grp, "SOH").sum()
        total_pos = to_num(grp, "POS").sum()
        coll_pct  = round(collected / demand * 100, 1) if demand > 0 else 0.0

        # NPA and SMA-2 counts
        npa_count  = 0
        sma2_count = 0
        if "curr_bucket" in grp.columns and "Loan No" in grp.columns:
            npa_count  = grp[grp["curr_bucket"] == "NPA"]["Loan No"].nunique()
            sma2_count = grp[grp["curr_bucket"] == "SMA-2"]["Loan No"].nunique()

        roll_fwd_pct = roll_bwd_pct = None
        if has_roll:
            curr_score = grp["curr_bucket"].map(BUCKET_SCORE)
            prev_score = grp["prev_bucket"].map(BUCKET_SCORE)
            valid = curr_score.notna() & prev_score.notna()
            total_valid = int(valid.sum())
            if total_valid > 0:
                roll_fwd_pct = round((valid & (curr_score > prev_score)).sum() / total_valid * 100, 1)
                roll_bwd_pct = round((valid & (curr_score < prev_score)).sum() / total_valid * 100, 1)
            else:
                roll_fwd_pct = roll_bwd_pct = 0.0

        display_name = f"{exec_name} ({branch})" if branch else exec_name

        row = {
            "Executive (Branch)": display_name,
            "Accounts":           n,
            "Strike Rate %":      strike_rate,
            "Collection %":       coll_pct,
        }
        if has_roll:
            row["Roll Fwd %"] = roll_fwd_pct
            row["Roll Bwd %"] = roll_bwd_pct
        npa_pct  = round(npa_count  / n * 100, 1) if n > 0 else 0.0
        sma2_pct = round(sma2_count / n * 100, 1) if n > 0 else 0.0
        row.update({
            "SMA-2 %":        sma2_pct,
            "NPA %":          npa_pct,
            "NPA":            npa_count,
            "SMA-2":          sma2_count,
            "Total POS (L)":  round(total_pos / 100_000, 2),
            "Total SOH (L)":  round(total_soh / 100_000, 2),
            "Demand (L)":     round(demand / 100_000, 2),
            "Collected (L)":  round(collected / 100_000, 2),
            "_coll_pct_raw":  coll_pct,
        })
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    sc = pd.DataFrame(rows).sort_values("_coll_pct_raw", ascending=False)
    sc["Tier"] = _quartile_tier(sc["_coll_pct_raw"])
    sc = sc.drop(columns=["_coll_pct_raw"])
    return sc.reset_index(drop=True)


def _quartile_tier(series: pd.Series) -> pd.Series:
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
    (the default Scorecard tab view, report_agent's executive_rankings section,
    AI Query's executive_rankings view) are unaffected unless they explicitly opt in.
    """
    if scorecard_df is None or scorecard_df.empty or metric_col not in scorecard_df.columns:
        return scorecard_df
    ranked = scorecard_df.sort_values(metric_col, ascending=False).copy()
    ranked["Tier"] = _quartile_tier(ranked[metric_col])
    return ranked.reset_index(drop=True)


def build_scorecard_table_html(scorecard_df: pd.DataFrame) -> str:
    """Returns a fully inline-CSS HTML table with color-coded performance tiers."""
    TIER_STYLE = {
        "top":    ("border-left:4px solid #16a34a;background:#f0fdf4;", "#16a34a"),
        "mid":    ("border-left:4px solid #d97706;background:#fff;",    "#d97706"),
        "bottom": ("border-left:4px solid #dc2626;background:#fff5f5;", "#dc2626"),
    }
    TIER_LABEL = {"top": "TOP", "mid": "MID", "bottom": "LOW"}

    headers = [c for c in scorecard_df.columns if c != "Tier"]
    header_html = "".join(
        f'<th style="background:#111;color:#FFC000;padding:8px 12px;'
        f'text-align:left;font-size:12px;white-space:nowrap;">{h}</th>'
        for h in headers
    )

    rows_html = ""
    for _, row in scorecard_df.iterrows():
        tier = row.get("Tier", "mid")
        row_style, tier_color = TIER_STYLE.get(tier, TIER_STYLE["mid"])
        tier_badge = (
            f'<span style="background:{tier_color};color:#fff;font-size:10px;'
            f'font-weight:700;padding:2px 7px;border-radius:10px;">'
            f'{TIER_LABEL.get(tier, tier)}</span>'
        )
        cells = ""
        for col in headers:
            val = row[col]
            if col == "Executive (Branch)":
                cells += (
                    f'<td style="padding:8px 12px;font-size:13px;font-weight:600;">'
                    f'{val} &nbsp;{tier_badge}</td>'
                )
            elif col == "Collection %":
                coll_color = "#16a34a" if val > 100 else "#d97706" if val >= 90 else "#dc2626"
                coll_bg    = "rgba(22,163,74,0.08)" if val > 100 else "rgba(217,119,6,0.08)" if val >= 90 else "rgba(220,38,38,0.08)"
                cells += (
                    f'<td style="padding:8px 12px;font-size:13px;font-weight:800;color:{coll_color};'
                    f'background:{coll_bg};border-radius:4px;">'
                    f'{val}%</td>'
                )
            elif col == "Roll Fwd %":
                color = "#dc2626" if val >= 20 else "#d97706" if val >= 10 else "#16a34a"
                cells += f'<td style="padding:8px 12px;font-size:13px;color:{color};font-weight:600;">{val}%</td>'
            elif col == "Roll Bwd %":
                color = "#16a34a" if val >= 10 else "#d97706" if val >= 5 else "#6b7280"
                cells += f'<td style="padding:8px 12px;font-size:13px;color:{color};font-weight:600;">{val}%</td>'
            elif col in ("Strike Rate %", "NPA %"):
                cells += f'<td style="padding:8px 12px;font-size:13px;">{val}%</td>'
            elif col == "NPA":
                color = "#dc2626" if val > 0 else "#16a34a"
                cells += f'<td style="padding:8px 12px;font-size:13px;font-weight:700;color:{color};">{val}</td>'
            elif col == "SMA-2":
                color = "#d97706" if val > 0 else "#16a34a"
                cells += f'<td style="padding:8px 12px;font-size:13px;font-weight:700;color:{color};">{val}</td>'
            else:
                cells += f'<td style="padding:8px 12px;font-size:13px;">{val}</td>'
        rows_html += (
            f'<tr style="{row_style}border-bottom:1px solid #e5e7eb;">{cells}</tr>'
        )

    return (
        f'<div style="overflow-x:auto;border-radius:10px;border:1px solid #e5e7eb;">'
        f'<table style="width:100%;border-collapse:collapse;font-family:Inter,sans-serif;">'
        f'<thead><tr>{header_html}</tr></thead>'
        f'<tbody>{rows_html}</tbody>'
        f'</table></div>'
    )
