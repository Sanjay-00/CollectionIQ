"""
Field Executive Performance Scorecard
Groups by MNT NAME and computes per-executive collection metrics.
Performance tiers are quartile-based (relative to the dataset) - not hardcoded thresholds.
"""
import html

import pandas as pd

from config import HARD_BUCKET_ARREARS_EMI_MIN, SCORECARD_MIN_ACCOUNTS
from utils import BUCKET_SCORE, to_num, _safe_pct


def compute_executive_scorecard(df: pd.DataFrame, min_accounts: int = SCORECARD_MIN_ACCOUNTS) -> pd.DataFrame:
    """
    Returns a DataFrame ranked by collection_pct with performance_tier column.

    Columns: Executive (Branch), Accounts, Strike Rate %, Collection %,
             [Roll Fwd %, Roll Bwd %], NPA, SMA-2, Total POS (L), Total SOH (L),
             Demand (L), Collected (L), Tier
    Groups by MNT NAME + Unit so the same executive in different branches appears separately.
    Executives with fewer than min_accounts are excluded.

    Every per-executive number below is computed as ONE groupby(group_cols).sum()
    pass over the WHOLE file (vectorized), not a Python-level call per executive
    inside a loop -- confirmed via profiling to be the dominant cost of this
    function on a real ~60k-row file (~287 executives x a full-frame slice +
    a compute_strike_pct() call each, ~2.1s). Only the FINAL per-executive
    arithmetic (ratios, None-handling, display formatting) still runs as a
    plain Python loop, but over the already-aggregated ~287-row summary, not
    the raw file, so that loop's cost is negligible.
    """
    if "MNT NAME" not in df.columns:
        return pd.DataFrame()

    has_roll = "prev_bucket" in df.columns and "curr_bucket" in df.columns
    group_cols = ["MNT NAME", "Unit"] if "Unit" in df.columns else ["MNT NAME"]

    df = df.copy()
    # Strike valid/yes -- the EXACT boolean logic compute_strike_pct/is_yes use
    # (see their own docstrings): valid = normalized Strike in {Y, N, YES, NO}
    # (some monthly LCC extracts spell the flag out instead of abbreviating it
    # -- same reason is_yes() itself accepts both spellings), and within that
    # valid set, yes = normalized in {Y, YES}, matching is_yes() exactly.
    # Precomputed once here instead of calling compute_strike_pct(grp) per
    # executive -- same formula, same result, just not re-run per group.
    if "Strike" in df.columns:
        _strike_norm = df["Strike"].astype(str).str.strip().str.upper()
        df["_strike_valid"] = _strike_norm.isin(["Y", "N", "YES", "NO"])
        df["_strike_yes"]   = _strike_norm.isin(["Y", "YES"])
    else:
        df["_strike_valid"] = False
        df["_strike_yes"]   = False

    df["_demand_num"]    = to_num(df, "Net Collection Demand Inst+Exp+BC")
    df["_collected_num"] = to_num(df, "Month Collection (Excluding Reserve Collection)")
    df["_soh_num"]       = to_num(df, "SOH")
    df["_pos_num"]       = to_num(df, "POS")
    # Hard Bucket % -- same HARD_BUCKET_ARREARS_EMI_MIN threshold
    # utils.py::compute_hard_bucket_pct uses as the single source of truth
    # (CLAUDE.md is explicit this must never be reimplemented locally).
    # Precomputed as a boolean column once, then groupby().sum() below --
    # same vectorized-once-over-the-whole-file convention as every other
    # per-executive number in this function (see this function's own
    # docstring for why a per-executive Python loop was already ruled out).
    df["_hard_bucket_flag"] = to_num(df, "Arrears / EMI") >= HARD_BUCKET_ARREARS_EMI_MIN

    if has_roll:
        curr_score = df["curr_bucket"].map(BUCKET_SCORE)
        prev_score = df["prev_bucket"].map(BUCKET_SCORE)
        df["_roll_valid"] = curr_score.notna() & prev_score.notna()
        df["_roll_fwd"]   = df["_roll_valid"] & (curr_score > prev_score)
        df["_roll_bwd"]   = df["_roll_valid"] & (curr_score < prev_score)

    agg_cols = {
        "_strike_valid": "sum", "_strike_yes": "sum",
        "_demand_num": "sum", "_collected_num": "sum",
        "_soh_num": "sum", "_pos_num": "sum",
        "_hard_bucket_flag": "sum",
    }
    if has_roll:
        agg_cols.update({"_roll_valid": "sum", "_roll_fwd": "sum", "_roll_bwd": "sum"})
    agg = df.groupby(group_cols).agg(agg_cols)
    n_accounts = df.groupby(group_cols)["Loan No"].nunique()

    # NPA/SMA-2 counts: nunique Loan No (matching the original's
    # grp[grp["curr_bucket"]==X]["Loan No"].nunique()), computed as a groupby
    # over each bucket's OWN filtered (smaller) slice -- still 2 vectorized
    # passes total, not one Python call per executive.
    if "curr_bucket" in df.columns and "Loan No" in df.columns:
        npa_counts  = df[df["curr_bucket"] == "NPA"].groupby(group_cols)["Loan No"].nunique()
        sma2_counts = df[df["curr_bucket"] == "SMA-2"].groupby(group_cols)["Loan No"].nunique()
    else:
        npa_counts = sma2_counts = pd.Series(dtype=int)

    rows = []
    for keys, n in n_accounts.items():
        if n < min_accounts:
            continue
        if isinstance(keys, tuple):
            exec_name, branch = str(keys[0]), str(keys[1])
        else:
            exec_name, branch = str(keys), ""

        a = agg.loc[keys]
        strike_rate = round(_safe_pct(a["_strike_yes"], a["_strike_valid"]), 1)
        demand    = a["_demand_num"]
        collected = a["_collected_num"]
        total_soh = a["_soh_num"]
        total_pos = a["_pos_num"]
        coll_pct  = round(collected / demand * 100, 1) if demand > 0 else 0.0

        npa_count  = int(npa_counts.get(keys, 0))
        sma2_count = int(sma2_counts.get(keys, 0))

        roll_fwd_pct = roll_bwd_pct = None
        if has_roll:
            total_valid = int(a["_roll_valid"])
            if total_valid > 0:
                roll_fwd_pct = round(a["_roll_fwd"] / total_valid * 100, 1)
                roll_bwd_pct = round(a["_roll_bwd"] / total_valid * 100, 1)
            # else: leave as None (not 0.0) -- e.g. a newly appointed executive
            # whose entire book is freshly originated this month has NO prior-month
            # bucket to compare against at all. Fabricating 0.0% here would read as
            # "verified: nothing got worse", which is a different, false claim from
            # the true state "not enough history to say". Same None/N-A convention
            # analysis/portfolio_intelligence.py::_roll_rates() already uses for the
            # identical situation at the region/branch grain -- this was a second,
            # independent implementation of the same roll-rate math that had quietly
            # drifted from it at this one edge case.

        display_name = f"{exec_name} ({branch})" if branch else exec_name
        hard_bucket_pct = round(_safe_pct(a["_hard_bucket_flag"], n), 1)

        row = {
            "Executive (Branch)": display_name,
            # Raw, unformatted MNT NAME/Unit -- added for the Investigator
            # feature (investigator/steps.py) so a caller can group/filter by
            # branch without re-parsing "Executive (Branch)"'s display
            # string. Purely additive: existing consumers key off column
            # names, never position, so this can't break them.
            "MNT NAME":           exec_name,
            "Unit":               branch,
            "Accounts":           int(n),
            "Strike Rate %":      strike_rate,
            "Collection %":       coll_pct,
            "Hard Bucket %":      hard_bucket_pct,
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

    # Local import (not module-level): analysis/ is otherwise pure pandas,
    # no UI dependency -- append_total_row lives in ui/components.py since
    # every OTHER table's Total row goes through it too, and ui/components.py
    # itself never imports back into analysis/, so this can't cycle.
    from ui.components import append_total_row

    headers = [c for c in scorecard_df.columns if c != "Tier"]
    header_html = "".join(
        f'<th style="background:#111;color:#FFC000;padding:8px 12px;'
        f'text-align:left;font-size:12px;white-space:nowrap;">{h}</th>'
        for h in headers
    )

    rows_html = ""
    _ratio_cols = {
        "Collection %": ("Collected (L)", "Demand (L)", 100),
        "NPA %":         ("NPA", "Accounts", 100),
        "SMA-2 %":       ("SMA-2", "Accounts", 100),
    }
    df_display = append_total_row(scorecard_df[headers], ratio_cols=_ratio_cols)
    n_data_rows = len(scorecard_df)
    for i, row in df_display.iterrows():
        if i == n_data_rows:
            cells = "".join(
                f'<td style="padding:8px 12px;font-size:13px;font-weight:800;'
                f'border-top:2px solid #FFC000;">{f"{row[c]}%" if "%" in c and row[c] != "" else row[c]}</td>'
                for c in headers
            )
            rows_html += f'<tr style="border-bottom:1px solid #e5e7eb;background:#fffbea;">{cells}</tr>'
            continue
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
                # MNT NAME/Unit are manually-typed LCC fields -- escape so an
                # &, <, > in a real name can't break the table markup (same
                # rule as report_builder.py's _esc and ui/components.py's).
                cells += (
                    f'<td style="padding:8px 12px;font-size:13px;font-weight:600;">'
                    f'{html.escape(str(val))} &nbsp;{tier_badge}</td>'
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
                if val is None or pd.isna(val):
                    cells += '<td style="padding:8px 12px;font-size:13px;color:#9ca3af;"> - </td>'
                else:
                    color = "#dc2626" if val >= 20 else "#d97706" if val >= 10 else "#16a34a"
                    cells += f'<td style="padding:8px 12px;font-size:13px;color:{color};font-weight:600;">{val}%</td>'
            elif col == "Roll Bwd %":
                if val is None or pd.isna(val):
                    cells += '<td style="padding:8px 12px;font-size:13px;color:#9ca3af;"> - </td>'
                else:
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
                cells += f'<td style="padding:8px 12px;font-size:13px;">{html.escape(val) if isinstance(val, str) else val}</td>'
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
