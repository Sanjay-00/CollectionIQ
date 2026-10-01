"""
Field Executive Performance Scorecard
Groups by MNT NAME and computes per-executive collection metrics.
Performance tiers are quartile-based (relative to the dataset) - not hardcoded thresholds.
"""
import html

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
             SMA-2 %, NPA, NPA %, NPA % (SOH), Strike Rate %, Collection %,
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
    sc["NPA"] = m["NPA"].values
    sc["SMA-2"] = m["SMA-2"].values
    sc["Total POS (L)"] = (m["POS"] / 100_000).round(2).values
    sc["Total SOH (L)"] = (m["SOH"] / 100_000).round(2).values
    sc["Demand (L)"] = (m["Demand"] / 100_000).round(2).values
    sc["Collected (L)"] = (m["Collected"] / 100_000).round(2).values

    sc = sc.sort_values("Collection %", ascending=False, kind="stable")
    sc["Tier"] = _quartile_tier(sc["Collection %"])
    sc = sc.reset_index(drop=True)
    on = {"MNT NAME": "MNT NAME", "Unit": "Unit"} if "Unit" in df.columns else {"MNT NAME": "MNT NAME"}
    sc = attach_prev_delinquency(sc, df_prev, on, pct_col="Delinquency %", fuzzy="MNT NAME", decimals=1)
    return sc[order_unit_columns(sc.columns, ["Executive (Branch)", "MNT NAME", "Unit"])]


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
    from ui.components import append_total_row, heat_range, heat_style

    # MNT NAME / Unit are raw copies of what "Executive (Branch)" already shows.
    headers = [c for c in scorecard_df.columns if c not in ("Tier", "MNT NAME", "Unit")]
    header_html = "".join(
        f'<th style="background:#111;color:#FFC000;padding:8px 12px;'
        f'text-align:left;font-size:12px;white-space:nowrap;">{h}</th>'
        for h in headers
    )

    rows_html = ""
    _ratio_cols = {
        "Collection %":   ("Collected (L)", "Demand (L)", 100),
        "NPA %":          ("NPA", "Accounts", 100),
        "SMA-2 %":        ("SMA-2", "Accounts", 100),
        "Delinquency %":  ("Delinquent", "Accounts", 100),
    }
    df_display = append_total_row(scorecard_df[headers], ratio_cols=_ratio_cols)
    n_data_rows = len(scorecard_df)
    # Risk columns: red shading by rank within the column, never green.
    risk_cols = ("Roll Fwd %", "NPA", "NPA %", "NPA % (SOH)", "SMA-2", "SMA-2 %", "Delinquent", "Delinquency %")
    heat = {c: heat_range(scorecard_df[c].tolist()) for c in risk_cols if c in headers}
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
            elif col in heat:
                if val is None or pd.isna(val):
                    cells += '<td style="padding:8px 12px;font-size:13px;color:#9ca3af;"> - </td>'
                else:
                    unit = "%" if col.endswith("%") else ""
                    cells += f'<td style="padding:8px 12px;font-size:13px;{heat_style(val, heat[col])}">{val}{unit}</td>'
            elif col == "Roll Bwd %":
                if val is None or pd.isna(val):
                    cells += '<td style="padding:8px 12px;font-size:13px;color:#9ca3af;"> - </td>'
                else:
                    color = "#16a34a" if val >= 10 else "#d97706" if val >= 5 else "#6b7280"
                    cells += f'<td style="padding:8px 12px;font-size:13px;color:{color};font-weight:600;">{val}%</td>'
            elif col == "Strike Rate %":
                cells += f'<td style="padding:8px 12px;font-size:13px;">{val}%</td>'
            elif col == "Prev Delinquency %":
                shown = " - " if val is None or pd.isna(val) else f"{val}%"
                cells += f'<td style="padding:8px 12px;font-size:13px;color:#6b7280;">{shown}</td>'
            elif col == "Δ Delinquency %":
                if val is None or pd.isna(val):
                    cells += '<td style="padding:8px 12px;font-size:13px;color:#9ca3af;"> - </td>'
                else:
                    # Positive = more of the book behind than last month (worse).
                    color = "#dc2626" if val > 0.1 else ("#16a34a" if val < -0.1 else "#6b7280")
                    arrow = "▲" if val > 0.1 else ("▼" if val < -0.1 else "-")
                    cells += f'<td style="padding:8px 12px;font-size:13px;font-weight:700;color:{color};">{arrow} {abs(val):.2f}pp</td>'
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
