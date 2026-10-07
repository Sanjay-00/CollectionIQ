"""New business: this month's new advances (originations), their trend
over time, and the disbursement-vintage chart. Pure pandas, no LLM."""

import re

import pandas as pd
import plotly.graph_objects as go

from utils import (
    YELLOW, _EXEC_KEY, _exec_name, _with_exec_key, _safe_pct, to_num, account_count, as_of_or_today,
    segment_column,
)
from config import (
    MIN_ACCOUNTS_PRODUCT_SEGMENT, VINTAGE_CHART_CRITICAL_PCT, VINTAGE_CHART_WATCH_PCT,
    NEW_ADVANCES_TREND_DEFAULT_MONTHS,
)


# ── Period rollup (shared by Disbursement Vintage and New Advances Trend) ────
# Single definition of what Monthly/Quarterly/Half-Yearly/Yearly/Financial Year
# MEAN, so the Business tab's two granularity pickers (Disbursement Vintage,
# New Advances Trend) can never silently diverge into two different
# "Quarterly" definitions -- see roll_vintage
# and roll_new_advances_trend below, both callers of these two helpers.


def _period_label(month_str: str, granularity: str) -> str:
    """Financial Year = Apr(Y)-Mar(Y+1) (Indian FY), labeled 'FYyy-yy' -- a
    genuinely different axis from calendar Yearly (Jan-Dec), not a relabeling
    of the same buckets. Quarterly/Half-Yearly/Yearly stay calendar-based,
    unchanged."""
    if granularity == "Monthly":
        return month_str
    try:
        y, m = int(month_str[:4]), int(month_str[5:7])
    except Exception:
        return month_str
    if granularity == "Quarterly":
        q = (m - 1) // 3 + 1
        return f"Q{q}-{y}"
    if granularity == "Half-Yearly":
        h = 1 if m <= 6 else 2
        return f"H{h}-{y}"
    if granularity == "Financial Year":
        fy_start = y if m >= 4 else y - 1
        return f"FY{str(fy_start)[-2:]}-{str(fy_start + 1)[-2:]}"
    return str(y)  # Yearly


def _period_sort_key(label: str) -> int:
    """Chronological sort key for _period_label's output labels."""
    try:
        if label.startswith("Q"):
            q, y = label[1:].split("-")
            return int(y) * 10 + int(q)
        if label.startswith("H"):
            h, y = label[1:].split("-")
            return int(y) * 10 + int(h)
        if label.startswith("FY"):
            fy_start = int(label[2:4])
            return (2000 + fy_start) * 10
        return int(label) * 10
    except Exception:
        return 0


def _add_quarter_end_markers(fig: go.Figure, x_labels: list) -> None:
    for lbl in x_labels:
        s = str(lbl)
        if re.match(r"^\d{4}-(?:03|06|09|12)$", s):
            fig.add_vline(x=s, line_width=1, line_dash="dash", line_color="#9ca3af", opacity=0.6)


def compute_new_advances(df_curr: pd.DataFrame, as_of=None) -> dict:
    """New business funded THIS reporting month -- a loan counts here because
    of when it was ORIGINATED (Ag_Date's own month == the report's own
    reporting month), never gated by curr_bucket, since a fresh advance is
    still new business even if it's already gone delinquent by the time this
    file was uploaded. as_of is the report's own reporting month (app.py's
    Reporting Month picker / report_agent's curr_month), NOT wall-clock
    "today" -- see compute_product_analysis's as_of docstring for why
    (re-analyzing an old file must anchor to the month it reports on, not the
    day the code happens to run).

    Deliberately does NOT take df_prev: a single monthly LCC extract already
    carries every still-open loan regardless of which month it was
    originated in, so last month's own advances are just another Ag_Date
    slice of THIS SAME upload (curr_ym - 1), not a second file. This means
    the MoM comparison works even on a user's very first-ever upload, with
    no previous month file required.
    """
    empty = {
        "month": "", "accounts": 0, "funded_cr": 0.0, "avg_ticket_l": 0.0,
        "has_prev": False, "prev_month": "", "prev_accounts": 0, "prev_funded_cr": 0.0,
        "accounts_mom_pct": None, "funded_mom_pct": None, "segment": pd.DataFrame(),
    }
    if df_curr.empty or "Ag_Date" not in df_curr.columns:
        return empty

    curr_ym = as_of_or_today(as_of).to_period("M")
    prev_ym = curr_ym - 1

    def _month_slice(ym):
        ag = pd.to_datetime(df_curr["Ag_Date"], errors="coerce")
        return df_curr[ag.dt.to_period("M") == ym]

    curr_adv = _month_slice(curr_ym)
    prev_adv = _month_slice(prev_ym)

    def _totals(adv):
        n = account_count(adv)
        amt = to_num(adv, "Loan Amount").sum() if "Loan Amount" in adv.columns else 0.0
        return n, float(amt)

    curr_n, curr_amt = _totals(curr_adv)
    prev_n, prev_amt = _totals(prev_adv)
    has_prev = prev_n > 0

    seg_col = segment_column(curr_adv)
    segment_rows = []
    if seg_col and not curr_adv.empty:
        for val, grp in curr_adv.groupby(seg_col):
            val_str = str(val).strip()
            if not val_str or val_str.lower() in ("nan", "none", ""):
                continue
            n = account_count(grp)
            if n < MIN_ACCOUNTS_PRODUCT_SEGMENT:
                continue
            amt = float(to_num(grp, "Loan Amount").sum()) if "Loan Amount" in grp.columns else 0.0
            segment_rows.append({
                "Segment": val_str,
                "Accounts": n,
                "Funded (Cr)": round(amt / 1e7, 2),
                "Avg Ticket (L)": round((amt / n) / 1e5, 2) if n else 0.0,
                "Share %": _safe_pct(n, curr_n),
            })
    segment_df = (
        pd.DataFrame(segment_rows).sort_values("Funded (Cr)", ascending=False).reset_index(drop=True)
        if segment_rows else pd.DataFrame()
    )

    return {
        "month": str(curr_ym),
        "accounts": curr_n,
        "funded_cr": round(curr_amt / 1e7, 2),
        "avg_ticket_l": round((curr_amt / curr_n) / 1e5, 2) if curr_n else 0.0,
        "has_prev": has_prev,
        "prev_month": str(prev_ym) if has_prev else "",
        "prev_accounts": prev_n,
        "prev_funded_cr": round(prev_amt / 1e7, 2),
        "accounts_mom_pct": (round((curr_n - prev_n) / prev_n * 100, 2) if has_prev and prev_n else None),
        "funded_mom_pct": (round((curr_amt - prev_amt) / prev_amt * 100, 2) if has_prev and prev_amt else None),
        "segment": segment_df,
    }


def compute_new_advances_by_dimension(df_curr: pd.DataFrame, as_of=None) -> dict[str, pd.DataFrame]:
    """New advances this reporting month, one row per Region / Branch (Unit) /
    Executive (MNT NAME) -- Total Accounts (that entity's WHOLE portfolio in
    df_curr, any Ag_Date), Accounts This Month, Funded (Cr), Avg Ticket (L),
    plus MoM vs that SAME entity's own advances last month. Both months
    sourced entirely from df_curr's own Ag_Date history (see
    compute_new_advances's docstring for why no df_prev is needed). Sorted by
    Accounts This Month descending. Same identity-column convention as
    OVERDUE_DEMAND_IDENTITY_COLS: a branch row carries its Region, an
    executive row carries its Branch and Region."""
    empty = {"zone": pd.DataFrame(), "region": pd.DataFrame(), "branch": pd.DataFrame(), "executive": pd.DataFrame()}
    if df_curr.empty or "Ag_Date" not in df_curr.columns:
        return empty

    df_curr = _with_exec_key(df_curr)
    curr_ym = as_of_or_today(as_of).to_period("M")
    prev_ym = curr_ym - 1
    ag = pd.to_datetime(df_curr["Ag_Date"], errors="coerce")
    cohort = ag.dt.to_period("M")
    curr_adv = df_curr[cohort == curr_ym]
    prev_adv = df_curr[cohort == prev_ym]

    def _grouped_totals(adv: pd.DataFrame, col: str, extra_cols: list[tuple[str, str]]):
        """One groupby().agg() pass over `adv` -> {name: (n, amt, uniq_cust)}
        plus a parallel {name: {src: first_value}} for extra_cols. Replaces
        both curr_adv's per-group Python loop AND (the real cost) prev_adv's
        per-group boolean-mask re-filter (`prev_adv[prev_adv[col] == name]`,
        once per group) that used to re-scan the whole prev_adv frame from
        scratch for every single region/branch/executive -- profiled at
        ~1.4s for the 254-executive case alone on a real ~60k-row file,
        almost entirely pandas' own per-filter take/reindex machinery, not
        any real per-group computation. Verified byte-for-byte identical to
        the old implementation via pd.testing.assert_frame_equal.

        uniq_cust is Cust Mob No nunique -- Loan No/Accounts counts loans,
        which overstates how many distinct customers actually took a new
        advance whenever one customer originates more than one loan this
        month. Same customer-identity proxy already used by
        compute_fleet_exposure (Cust Mob No may not be unique across
        branches -- treat as approximate, same caveat as that function)."""
        if col not in adv.columns or adv.empty:
            return {}, {}
        has_loan_no = "Loan No" in adv.columns
        has_cust_mob = "Cust Mob No" in adv.columns
        amt_col = to_num(adv, "Loan Amount") if "Loan Amount" in adv.columns else pd.Series(0.0, index=adv.index)
        agg_df = adv.assign(_amt=amt_col)
        if has_cust_mob:
            # Blank/missing mobile numbers must not collapse into one fake
            # "customer" via nunique -- exclude them from the count the same
            # way compute_fleet_exposure does before its own groupby.
            mob = agg_df["Cust Mob No"].astype(str).str.strip()
            agg_df["_cust_mob"] = mob.where(mob != "", pd.NA)
        spec = {"_amt": "sum"}
        if has_loan_no:
            spec["Loan No"] = "nunique"
        if has_cust_mob:
            spec["_cust_mob"] = "nunique"
        extra_srcs = [src for src, _ in extra_cols if src in agg_df.columns]
        for src in extra_srcs:
            spec[src] = "first"
        grouped = agg_df.groupby(col).agg(spec)
        if has_loan_no:
            n_col = "Loan No"
        else:
            grouped["_n"] = agg_df.groupby(col).size()
            n_col = "_n"
        totals = {
            name: (int(r[n_col]), float(r["_amt"]), int(r["_cust_mob"]) if has_cust_mob else int(r[n_col]))
            for name, r in grouped.iterrows()
        }
        extras = {name: {src: r[src] for src in extra_srcs} for name, r in grouped.iterrows()}
        return totals, extras

    def _rows(col: str, label: str, extra_cols: list[tuple[str, str]] | None = None, min_accounts: int = 0) -> pd.DataFrame:
        extra_cols = extra_cols or []
        if col not in curr_adv.columns or curr_adv.empty:
            return pd.DataFrame()
        # That entity's WHOLE book in this upload, any Ag_Date -- distinct from
        # curr_adv/prev_adv, which are only THIS month's / last month's fresh
        # originations. Grouped over the full df_curr once, not per-row.
        total_counts = (
            df_curr.groupby(col)["Loan No"].nunique()
            if "Loan No" in df_curr.columns else df_curr.groupby(col).size()
        )
        curr_totals, curr_extras = _grouped_totals(curr_adv, col, extra_cols)
        prev_totals, _ = _grouped_totals(prev_adv, col, [])
        rows = []
        for name, (n, amt, uniq_cust) in curr_totals.items():
            if n < min_accounts:
                continue
            prev_n, prev_amt, _ = prev_totals.get(name, (0, 0.0, 0))
            row = {label: _exec_name(name) if col == _EXEC_KEY else name}
            for src, out_label in extra_cols:
                row[out_label] = curr_extras.get(name, {}).get(src)
            row.update({
                "Total Accounts": int(total_counts.get(name, n)),
                "Accounts This Month": n,
                "Unique Customers": uniq_cust,
                "Funded (Cr)": round(amt / 1e7, 2),
                "Avg Ticket (L)": round((amt / n) / 1e5, 2) if n else 0.0,
                "Prev Accounts": prev_n,
                "Prev Funded (Cr)": round(prev_amt / 1e7, 2),
                "Accounts MoM %": (round((n - prev_n) / prev_n * 100, 2) if prev_n else None),
                "Funded MoM %": (round((amt - prev_amt) / prev_amt * 100, 2) if prev_amt else None),
            })
            rows.append(row)
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows).sort_values("Accounts This Month", ascending=False).reset_index(drop=True)

    return {
        "zone":      _rows("Zone", "Zone"),
        "region":    _rows("RegionName", "Region", extra_cols=[("Zone", "Zone")]),
        "branch":    _rows("Unit", "Branch", extra_cols=[("RegionName", "Region")]),
        # Deliberately NO materiality floor here (min_accounts=0, every
        # executive with >=1 new advance shows up) -- unlike the Overdue vs
        # Month Demand league table, this isn't ranking collection
        # performance where a single account can swing a %; it's a business/
        # origination view where even one new advance is real information.
        "executive": _rows(_EXEC_KEY, "Executive", extra_cols=[("Unit", "Branch"), ("RegionName", "Region")],
                            min_accounts=0),
    }


def compute_new_advances_trend(df_curr: pd.DataFrame, as_of=None, months: int | None = NEW_ADVANCES_TREND_DEFAULT_MONTHS) -> pd.DataFrame:
    """Monthly new-advances trend (Accounts, Funded (Cr), Avg Ticket (L)) for
    the last `months` calendar months up to and including the report's own
    reporting month (as_of) -- excludes any post-dated Ag_Date cohort beyond
    as_of, same rule compute_product_analysis's vintage cohort already uses.
    months=None means "All" -- every cohort in df_curr's Ag_Date history, no
    lower bound. Entirely derived from df_curr's own Ag_Date history across
    its full upload, not capped to "this file's reporting month" the way
    compute_new_advances is -- that's the whole point: at least 2 years of
    trend from a single upload, no df_prev needed."""
    if df_curr.empty or "Ag_Date" not in df_curr.columns:
        return pd.DataFrame()

    ref = pd.Timestamp(as_of) if as_of is not None else pd.Timestamp.today()
    curr_ym = ref.to_period("M")

    ag = pd.to_datetime(df_curr["Ag_Date"], errors="coerce")
    df_v = df_curr.assign(_cohort=ag.dt.to_period("M"))
    df_v = df_v.dropna(subset=["_cohort"])
    df_v = df_v[df_v["_cohort"] <= curr_ym]
    if months is not None:
        earliest = curr_ym - (months - 1)
        df_v = df_v[df_v["_cohort"] >= earliest]
    if df_v.empty:
        return pd.DataFrame()

    rows = []
    for cohort, grp in df_v.groupby("_cohort"):
        n = account_count(grp)
        amt = float(to_num(grp, "Loan Amount").sum()) if "Loan Amount" in grp.columns else 0.0
        rows.append({
            "Month": str(cohort),
            "Accounts": n,
            "Funded (Cr)": round(amt / 1e7, 2),
            "Avg Ticket (L)": round((amt / n) / 1e5, 2) if n else 0.0,
        })
    return pd.DataFrame(rows).sort_values("Month").reset_index(drop=True)


def roll_new_advances_trend(trend_df: pd.DataFrame, granularity: str) -> pd.DataFrame:
    """Roll the monthly new-advances trend up to Quarter / Half-Year / Year /
    Financial Year and recompute Avg Ticket -- shares _period_label/
    _period_sort_key with roll_vintage,
    so the two granularity pickers in the Business tab mean the exact same
    thing."""
    if granularity == "Monthly" or trend_df.empty:
        return trend_df

    df = trend_df.copy()
    df["_period"] = df["Month"].apply(lambda m: _period_label(m, granularity))

    agg = (
        df.groupby("_period", sort=False)
        .agg(Accounts=("Accounts", "sum"), **{"Funded (Cr)": ("Funded (Cr)", "sum")})
        .reset_index()
        .rename(columns={"_period": "Month"})
    )
    agg["Avg Ticket (L)"] = ((agg["Funded (Cr)"] * 1e7 / agg["Accounts"]) / 1e5).round(2)
    agg["Avg Ticket (L)"] = agg["Avg Ticket (L)"].where(agg["Accounts"] > 0, 0.0)

    agg["_sk"] = agg["Month"].apply(_period_sort_key)
    return agg.sort_values("_sk").drop(columns=["_sk"]).reset_index(drop=True)


def roll_vintage(df: pd.DataFrame, granularity: str) -> pd.DataFrame:
    """Roll monthly cohorts up to Quarter / Half-Year / Year / Financial Year
    and recompute NPA% + SMA-2%, with the same _period_label/_period_sort_key
    as roll_new_advances_trend, so the Business tab's two granularity pickers
    always mean the same "Quarterly"/"Financial Year"."""
    if granularity == "Monthly" or df.empty:
        return df

    df = df.copy()
    df["_period"] = df["Disbursement Month"].apply(lambda m: _period_label(m, granularity))

    agg = (
        df.groupby("_period", sort=False)
        .agg(
            Accounts=("Accounts", "sum"),
            **{"NPA Count":   ("NPA Count",   "sum")},
            **{"SMA-2 Count": ("SMA-2 Count", "sum")},
            **{"SOH (Cr)":    ("SOH (Cr)",    "sum")},
        )
        .reset_index()
        .rename(columns={"_period": "Disbursement Month"})
    )
    agg["NPA%"]   = (agg["NPA Count"]   / agg["Accounts"] * 100).round(2)
    agg["SMA-2%"] = (agg["SMA-2 Count"] / agg["Accounts"] * 100).round(2)

    agg["_sk"] = agg["Disbursement Month"].apply(_period_sort_key)
    return agg.sort_values("_sk").drop(columns=["_sk"]).reset_index(drop=True)


def compute_new_advances_trend_chart(trend_df: pd.DataFrame, granularity: str = "Monthly") -> go.Figure:
    """Bar (Accounts, left axis) + line (Funded Cr, right axis) by period.
    On Monthly granularity, a dashed vertical marker highlights each Indian
    FY quarter-close month (Mar/Jun/Sep/Dec) -- a no-op once rolled up to
    Quarter/Half-Year/Year/Financial Year, where individual months no longer
    appear on the axis."""
    fig = go.Figure()
    if trend_df.empty or "Month" not in trend_df.columns:
        fig.update_layout(title=dict(text="No data available", font=dict(size=13, color="#111")), height=300)
        return fig

    x = trend_df["Month"].tolist()
    fig.add_trace(go.Bar(
        name="Accounts", x=x, y=trend_df["Accounts"].tolist(),
        marker=dict(color="#1d4ed8"), yaxis="y1",
        hovertemplate="<b>%{x}</b><br>Accounts: %{y:,}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        name="Funded (Cr)", x=x, y=trend_df["Funded (Cr)"].tolist(),
        mode="lines+markers", line=dict(color=YELLOW, width=2.8),
        marker=dict(size=7, color=YELLOW, line=dict(width=1.5, color="#111")),
        yaxis="y2",
        hovertemplate="<b>%{x}</b><br>Funded: &#8377;%{y:.2f}Cr<extra></extra>",
    ))
    if granularity == "Monthly":
        _add_quarter_end_markers(fig, x)
    fig.update_layout(
        title=dict(text="New Advances Trend: Accounts &amp; Funded Amount by Period", font=dict(size=13, color="#111"), x=0),
        plot_bgcolor="white", paper_bgcolor="white",
        xaxis=dict(tickangle=-45, showgrid=False, tickfont=dict(size=10, color="#374151")),
        yaxis=dict(title="Accounts", showgrid=True, gridcolor="#f3f4f6", tickfont=dict(color="#6b7280")),
        yaxis2=dict(title="Funded (Cr)", overlaying="y", side="right", showgrid=False, tickfont=dict(color="#6b7280")),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font=dict(size=11)),
        margin=dict(l=20, r=20, t=60, b=90),
        height=380,
    )
    return fig


def build_vintage_chart(vintage_df: pd.DataFrame) -> go.Figure:
    """Dual-line chart: NPA% + SMA-2% by disbursement month. Colored markers per threshold."""
    if vintage_df.empty or "Disbursement Month" not in vintage_df.columns:
        return go.Figure()

    df = vintage_df.sort_values("Disbursement Month").copy()
    x = df["Disbursement Month"].tolist()

    def _marker_colors(series, thresholds):
        hi, mid = thresholds
        return ["#991b1b" if v >= hi else ("#f97316" if v >= mid else "#16a34a") for v in series]

    npa_vals  = df["NPA%"].tolist()
    sma2_vals = df["SMA-2%"].tolist() if "SMA-2%" in df.columns else [0.0] * len(df)

    _thresholds = (VINTAGE_CHART_CRITICAL_PCT, VINTAGE_CHART_WATCH_PCT)
    npa_colors  = _marker_colors(npa_vals,  _thresholds)
    sma2_colors = _marker_colors(sma2_vals, _thresholds)

    n_points = len(x)
    # Show labels only when not too crowded
    label_mode = "lines+markers+text" if n_points <= 18 else "lines+markers"

    fig = go.Figure()

    # NPA% shaded area fill
    fig.add_trace(go.Scatter(
        x=x, y=npa_vals,
        mode="lines",
        name="_npa_fill",
        line=dict(width=0),
        fill="tozeroy",
        fillcolor="rgba(220,38,38,0.07)",
        showlegend=False,
        hoverinfo="skip",
    ))

    # NPA% line with labels
    fig.add_trace(go.Scatter(
        x=x, y=npa_vals,
        mode=label_mode,
        name="NPA %",
        line=dict(color="#dc2626", width=2.8),
        marker=dict(size=11, color=npa_colors, line=dict(width=2, color="#fff")),
        text=[f"{v:.1f}%" for v in npa_vals],
        textposition="top center",
        textfont=dict(size=9, color="#dc2626", family="Arial Black"),
        hovertemplate="<b>%{x}</b><br>NPA %: %{y:.1f}%<extra></extra>",
    ))

    # SMA-2% line with labels
    fig.add_trace(go.Scatter(
        x=x, y=sma2_vals,
        mode=label_mode,
        name="SMA-2 %",
        line=dict(color="#f97316", width=2.8, dash="dot"),
        marker=dict(size=11, color=sma2_colors, symbol="diamond", line=dict(width=2, color="#fff")),
        text=[f"{v:.1f}%" for v in sma2_vals],
        textposition="bottom center",
        textfont=dict(size=9, color="#f97316", family="Arial Black"),
        hovertemplate="<b>%{x}</b><br>SMA-2 %%: %{y:.1f}%<extra></extra>",
    ))

    # Threshold reference bands
    fig.add_hrect(y0=VINTAGE_CHART_CRITICAL_PCT, y1=max(max(npa_vals + sma2_vals) * 1.1, VINTAGE_CHART_CRITICAL_PCT + 2),
                  fillcolor="rgba(153,27,27,0.04)", line_width=0, layer="below")
    fig.add_hrect(y0=VINTAGE_CHART_WATCH_PCT, y1=VINTAGE_CHART_CRITICAL_PCT,
                  fillcolor="rgba(217,119,6,0.04)", line_width=0, layer="below")
    fig.add_hline(y=VINTAGE_CHART_CRITICAL_PCT, line_dash="dash", line_color="#991b1b", line_width=1,
                  annotation_text=f"Critical  {VINTAGE_CHART_CRITICAL_PCT}%", annotation_position="right",
                  annotation_font=dict(size=10, color="#991b1b"))
    fig.add_hline(y=VINTAGE_CHART_WATCH_PCT, line_dash="dash", line_color="#d97706", line_width=1,
                  annotation_text=f"Watch  {VINTAGE_CHART_WATCH_PCT}%", annotation_position="right",
                  annotation_font=dict(size=10, color="#d97706"))

    _add_quarter_end_markers(fig, x)

    fig.update_layout(
        title=dict(text="NPA % vs SMA-2 % by Disbursement Cohort", font=dict(size=13, color="#111"), x=0),
        xaxis=dict(title="Disbursement Cohort", tickangle=-45, showgrid=False,
                   tickfont=dict(color="#374151")),
        yaxis=dict(title="Delinquency %", showgrid=True, gridcolor="#f3f4f6",
                   tickfont=dict(color="#374151"), ticksuffix="%"),
        plot_bgcolor="white", paper_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                    font=dict(size=11)),
        margin=dict(l=20, r=90, t=60, b=90),
        height=430,
    )
    return fig
