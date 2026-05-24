"""
Field Executive Performance Scorecard
Groups by MNT NAME and computes per-executive collection metrics.
Performance tiers are quartile-based (relative to the dataset) — not hardcoded thresholds.
"""
import pandas as pd


def compute_executive_scorecard(df: pd.DataFrame, min_accounts: int = 5) -> pd.DataFrame:
    """
    Returns a DataFrame ranked by collection_pct with performance_tier column.

    Columns: Executive (Branch), Accounts, Strike Rate %, Collection %,
             NPA %, Total POS (L), Demand (L), Collected (L), Tier
    Groups by MNT NAME + Unit so the same executive in different branches appears separately.
    Executives with fewer than min_accounts are excluded.
    """
    if "MNT NAME" not in df.columns:
        return pd.DataFrame()

    group_cols = ["MNT NAME", "Unit"] if "Unit" in df.columns else ["MNT NAME"]

    rows = []
    for keys, grp in df.groupby(group_cols):
        if isinstance(keys, tuple):
            exec_name, branch = str(keys[0]), str(keys[1])
        else:
            exec_name, branch = str(keys), ""

        n = grp["Loan No"].nunique() if "Loan No" in grp.columns else len(grp)
        if n < min_accounts:
            continue

        # Strike rate — only count rows where Strike is Y or N
        strike_valid = grp[grp["Strike"].isin(["Y", "N"])] if "Strike" in grp.columns else pd.DataFrame()
        strike_rate = round(
            (strike_valid["Strike"] == "Y").sum() / len(strike_valid) * 100, 1
        ) if len(strike_valid) > 0 else 0.0

        demand     = pd.to_numeric(grp.get("NET Collection Demand Inst+Exp", pd.Series(dtype=float)), errors="coerce").sum()
        collected  = pd.to_numeric(grp.get("Month Receipt Amount", pd.Series(dtype=float)), errors="coerce").sum()
        total_pos  = pd.to_numeric(grp.get("POS", pd.Series(dtype=float)), errors="coerce").sum()
        coll_pct   = round(collected / demand * 100, 1) if demand > 0 else 0.0

        npa_count  = 0
        if "curr_bucket" in grp.columns and "Loan No" in grp.columns:
            npa_count = grp[grp["curr_bucket"] == "NPA"]["Loan No"].nunique()
        npa_pct = round(npa_count / n * 100, 1) if n > 0 else 0.0

        display_name = f"{exec_name} ({branch})" if branch else exec_name

        rows.append({
            "Executive (Branch)": display_name,
            "Accounts":           n,
            "Strike Rate %":      strike_rate,
            "Collection %":       coll_pct,
            "NPA %":              npa_pct,
            "Total POS (L)":      round(total_pos / 100_000, 2),
            "Demand (L)":         round(demand / 100_000, 2),
            "Collected (L)":      round(collected / 100_000, 2),
            "_coll_pct_raw":      coll_pct,
        })

    if not rows:
        return pd.DataFrame()

    sc = pd.DataFrame(rows).sort_values("_coll_pct_raw", ascending=False)

    # Quartile-based tiers — relative to this dataset
    q75 = sc["_coll_pct_raw"].quantile(0.75)
    q25 = sc["_coll_pct_raw"].quantile(0.25)

    def _tier(val):
        if val >= q75:
            return "top"
        if val <= q25:
            return "bottom"
        return "mid"

    sc["Tier"] = sc["_coll_pct_raw"].apply(_tier)
    sc = sc.drop(columns=["_coll_pct_raw"])
    return sc.reset_index(drop=True)


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
            elif col in ("Strike Rate %", "NPA %"):
                cells += f'<td style="padding:8px 12px;font-size:13px;">{val}%</td>'
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
