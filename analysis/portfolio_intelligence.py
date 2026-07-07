"""
Portfolio Intelligence Analytics  -  pre-computed, pure pandas, no LLM.
Answers the 5 portfolio questions a collection lead needs every month.
"""
import re

import pandas as pd
import numpy as np
import plotly.graph_objects as go

from utils import (
    BUCKET_ORDER, BUCKET_SCORE, BUCKET_COLORS, to_num, account_count, is_yes,
    compute_strike_pct, compute_hard_bucket_pct, compute_overdue_demand_pct,
)
from config import (
    MIN_ACCOUNTS_DIMENSION_BREAKDOWN,
    MIN_ACCOUNTS_PRODUCT_SEGMENT,
    MIN_ACCOUNTS_SOURCE_VINTAGE,
    MIN_ACCOUNTS_OVERDUE_DEMAND_EXECUTIVE,
    REPOSSESSION_WINDOW_MONTHS,
    GOOD_CUSTOMER_MIN_TENURE_PCT,
    GOOD_CUSTOMER_MIN_LCC_PCT,
    FLEET_MIN_LOANS,
    REGION_STATUS_DELTA_PP,
    GOOD_BAD_REGION_DELTA_PP,
    CONCERN_SCORE_BAD_THRESHOLD,
    CONCERN_SCORE_GOOD_THRESHOLD,
    CONCERN_SCORE_WEIGHTS,
    RISK_INDICATOR_STABLE_PP,
    RISK_INDICATOR_MATERIALITY_PP,
    RISK_INDICATOR_MATERIALITY_COUNT,
    VINTAGE_CHART_CRITICAL_PCT,
    VINTAGE_CHART_WATCH_PCT,
    NEW_ADVANCES_TREND_DEFAULT_MONTHS,
)

YELLOW = "#FFC000"
VALID_BUCKETS = [b for b in BUCKET_ORDER if b != "NA"]


# ── Shared helpers ────────────────────────────────────────────────────────────

def _safe_div(num: float, den: float, scale: float = 100.0) -> float:
    return round(float(num) / float(den) * scale, 2) if den else 0.0


def _npa_pct(df: pd.DataFrame) -> float:
    if df.empty or "Loan No" not in df.columns or "curr_bucket" not in df.columns:
        return 0.0
    total = df["Loan No"].nunique()
    return _safe_div((df["curr_bucket"] == "NPA").sum(), total)


def _coll_pct(df: pd.DataFrame) -> float:
    demand = to_num(df, "Net Collection Demand Inst+Exp+BC").sum()
    coll = to_num(df, "Month Collection (Excluding Reserve Collection)").sum()
    return _safe_div(coll, demand)


def _soh_cr(df: pd.DataFrame) -> float:
    return round(to_num(df, "SOH").sum() / 1e7, 2)


# ── Period rollup (shared by Disbursement Vintage and New Advances Trend) ────
# Single definition of what Monthly/Quarterly/Half-Yearly/Yearly/Financial Year
# MEAN, so the Business tab's two granularity pickers (Disbursement Vintage,
# New Advances Trend) can never silently diverge into two different
# "Quarterly" definitions -- see ui/tabs/portfolio_intelligence.py::_roll_vintage
# and this module's roll_new_advances_trend, both callers of these two helpers.

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


# Indian FY quarter-CLOSE months (Jun/Sep/Dec/Mar close FY Q1-Q4 respectively).
# Coincidentally identical to calendar-quarter-end months -- the FY framing
# only changes which YEAR a Jan/Feb/Mar cohort belongs to, not which months
# are quarter boundaries. Used to mark these months on a Monthly-granularity
# chart's x-axis (raw "YYYY-MM" labels only -- a no-op once rolled up).
_QUARTER_END_MONTHS = {"03", "06", "09", "12"}


def _add_quarter_end_markers(fig: go.Figure, x_labels: list) -> None:
    for lbl in x_labels:
        s = str(lbl)
        if re.match(r"^\d{4}-(?:03|06|09|12)$", s):
            fig.add_vline(x=s, line_width=1, line_dash="dash", line_color="#9ca3af", opacity=0.6)


def _roll_rates(grp: pd.DataFrame) -> tuple[float | None, float | None]:
    if "prev_bucket" not in grp.columns or "curr_bucket" not in grp.columns:
        return None, None
    curr_sc = grp["curr_bucket"].map(BUCKET_SCORE)
    prev_sc = grp["prev_bucket"].map(BUCKET_SCORE)
    valid = curr_sc.notna() & prev_sc.notna()
    n = int(valid.sum())
    if n == 0:
        return None, None
    fwd = round((valid & (curr_sc > prev_sc)).sum() / n * 100, 1)
    bwd = round((valid & (curr_sc < prev_sc)).sum() / n * 100, 1)
    return fwd, bwd


def _bucket_counts(df: pd.DataFrame) -> dict:
    if "curr_bucket" not in df.columns:
        return {b: 0 for b in VALID_BUCKETS}
    counts = df["curr_bucket"].value_counts()
    return {b: int(counts.get(b, 0)) for b in VALID_BUCKETS}


# ── Section 1: Portfolio Pulse ────────────────────────────────────────────────

def compute_bucket_waterfall(df_curr: pd.DataFrame, df_prev: pd.DataFrame) -> go.Figure:
    """
    Grouped bar chart: prev bucket count (gray) vs curr bucket count (colored).
    Labels show count + % of total. Net change annotated above each group.
    """
    curr_counts = _bucket_counts(df_curr)
    prev_counts = _bucket_counts(df_prev) if len(df_prev) > 0 else None
    has_prev    = prev_counts is not None

    buckets   = VALID_BUCKETS
    total_c   = sum(curr_counts.values()) or 1
    total_p   = sum(prev_counts.values()) or 1 if has_prev else 1

    def _label(counts, total, b):
        n   = counts[b]
        pct = n / total * 100
        return f"{n:,} ({pct:.1f}%)"

    fig = go.Figure()

    if has_prev:
        fig.add_trace(go.Bar(
            name="Last Month",
            x=buckets,
            y=[prev_counts[b] for b in buckets],
            marker=dict(color="#d1d5db", line=dict(width=0)),
            text=[_label(prev_counts, total_p, b) for b in buckets],
            textposition="outside",
            cliponaxis=False,
            textfont=dict(size=10, color="#4b5563"),
            hovertemplate="<b>%{x}</b>  -  Last Month<br>Count: %{y:,}<extra></extra>",
        ))

    curr_colors = [BUCKET_COLORS[b] for b in buckets]
    fig.add_trace(go.Bar(
        name="This Month",
        x=buckets,
        y=[curr_counts[b] for b in buckets],
        marker=dict(color=curr_colors, line=dict(width=0)),
        text=[_label(curr_counts, total_c, b) for b in buckets],
        textposition="outside",
        cliponaxis=False,
        textfont=dict(size=10, color="#111"),
        hovertemplate="<b>%{x}</b>  -  This Month<br>Count: %{y:,}<extra></extra>",
    ))

    max_val = max(
        max(curr_counts.values()),
        max(prev_counts.values()) if has_prev else 0,
    ) * 1.45 or 100

    if has_prev:
        # yshift is a fixed PIXEL offset (not a data-coordinate multiple), so the
        # delta annotation always sits a couple of text-lines above its own bar's
        # "outside" value label regardless of that bar's height - unlike scaling
        # off the y-value, which puts far-off annotations for short bars and
        # collides with the label for tall ones.
        for b in buckets:
            delta   = curr_counts[b] - prev_counts[b]
            if delta == 0:
                continue
            arrow   = "▲" if delta > 0 else "▼"
            is_bad  = (delta > 0 and b != "STD") or (delta < 0 and b == "STD")
            color   = "#dc2626" if is_bad else "#16a34a"
            fig.add_annotation(
                x=b,
                y=max(curr_counts[b], prev_counts[b]),
                yshift=28,
                text=f"<b>{arrow} {abs(delta):,}</b>",
                showarrow=False,
                font=dict(size=11, color=color),
                xanchor="center",
                bgcolor="rgba(255,255,255,0.9)",
                borderpad=2,
            )

    title = "Bucket Distribution  -  Last Month vs This Month" if has_prev else "Bucket Distribution (This Month)"
    fig.update_layout(
        title=dict(text=title, font=dict(size=13, color="#111"), x=0),
        barmode="group",
        bargap=0.25, bargroupgap=0.08,
        plot_bgcolor="white", paper_bgcolor="white",
        xaxis=dict(showgrid=False, tickfont=dict(size=12, color="#111")),
        yaxis=dict(showgrid=True, gridcolor="#f3f4f6", tickfont=dict(color="#6b7280")),
        margin=dict(l=10, r=10, t=50, b=10),
        height=360,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                    font=dict(size=11)),
        showlegend=has_prev,
        yaxis_range=[0, max_val],
    )
    return fig


def compute_pulse_kpis(df_curr: pd.DataFrame, df_prev: pd.DataFrame) -> list[dict]:
    """Top-line KPIs for Portfolio Pulse section. Includes SMA-2 alongside NPA."""
    def _calc(df):
        if df.empty:
            return {}
        total = account_count(df)
        soh = _soh_cr(df)
        npa_pct = _npa_pct(df)
        npa_count = int((df["curr_bucket"] == "NPA").sum()) if "curr_bucket" in df.columns else 0
        sma2_count = int((df["curr_bucket"] == "SMA-2").sum()) if "curr_bucket" in df.columns else 0
        sma2_pct = _safe_div(sma2_count, total)
        hard_pct = compute_hard_bucket_pct(df)
        coll = _coll_pct(df)
        strike_pct = compute_strike_pct(df)
        overdue_demand = compute_overdue_demand_pct(df)
        return {
            "accounts": total, "soh": soh,
            "npa_count": npa_count, "npa_pct": npa_pct,
            "sma2_count": sma2_count, "sma2_pct": sma2_pct,
            "hard_pct": hard_pct, "coll_pct": coll, "strike_pct": strike_pct,
            "overdue_coll_pct": overdue_demand["overdue_pct"],
            "demand_coll_pct": overdue_demand["demand_pct"],
        }

    c = _calc(df_curr)
    p = _calc(df_prev)

    def _delta(key, inverse=False):
        cv = c.get(key, 0)
        pv = p.get(key, 0)
        if not p or pv == 0:
            return None
        d = round(cv - pv, 2)
        return d if not inverse else -d

    # curr_raw/prev_raw: the literal unrounded numeric values behind "value" and
    # the (possibly sign-flipped-for-display) "delta" -- exposed additively so a
    # true comparison table (This Month | Previous Month | Delta | % Change) can
    # be built without reverse-engineering a flipped delta back into a raw prior
    # value (which would be fragile/easy to get subtly wrong). Existing consumers
    # (e.g. the Portfolio Pulse dashboard cards) only read label/value/delta/unit/
    # inverse and are unaffected by these extra keys.
    def _card(label, key, value, unit, inverse):
        return {
            "label": label, "value": value, "delta": _delta(key, inverse=inverse),
            "unit": unit, "inverse": inverse,
            "curr_raw": c.get(key, 0), "prev_raw": p.get(key, 0) if p else None,
        }

    return [
        _card("Total Accounts", "accounts",  f"{c.get('accounts',0):,}",      "",   False),
        _card("Total SOH",      "soh",       f"₹{c.get('soh',0):.2f}Cr",      "Cr", True),
        _card("SMA-2 Accounts", "sma2_count", f"{c.get('sma2_count',0):,}",   "",   True),
        _card("SMA-2 %",        "sma2_pct",  f"{c.get('sma2_pct',0):.2f}%",  "%",  True),
        _card("NPA Accounts",   "npa_count", f"{c.get('npa_count',0):,}",    "",   True),
        _card("NPA %",          "npa_pct",   f"{c.get('npa_pct',0):.2f}%",   "%",  True),
        _card("Collection %",   "coll_pct",  f"{c.get('coll_pct',0):.2f}%",  "%",  False),
        _card("Strike %",       "strike_pct", f"{c.get('strike_pct',0):.2f}%", "%", True),
        _card("Overdue Collection %", "overdue_coll_pct", f"{c.get('overdue_coll_pct',0):.2f}%", "%", False),
        _card("Month Demand Collection %", "demand_coll_pct", f"{c.get('demand_coll_pct',0):.2f}%", "%", False),
    ]


# ── Section 2: Region Delinquency Scorecard ────────────────────────────────────

def compute_region_scorecard(df_curr: pd.DataFrame, df_prev: pd.DataFrame) -> pd.DataFrame:
    """One row per region: SMA-2/NPA counts and rates, MoM deltas, Collection%, Strike%, SOH, roll rates, trend status."""
    if "RegionName" not in df_curr.columns or df_curr.empty:
        return pd.DataFrame()

    rows = []
    for region, grp in df_curr.groupby("RegionName"):
        n = account_count(grp)
        curr_npa = _npa_pct(grp)
        curr_coll = _coll_pct(grp)
        soh = _soh_cr(grp)
        strike_pct = compute_strike_pct(grp)
        roll_fwd, roll_bwd = _roll_rates(grp)

        sma2_count = int((grp["curr_bucket"] == "SMA-2").sum()) if "curr_bucket" in grp.columns else 0
        sma2_pct   = _safe_div(sma2_count, n)
        npa_count  = int((grp["curr_bucket"] == "NPA").sum()) if "curr_bucket" in grp.columns else 0

        prev_npa = 0.0
        prev_sma2_pct = 0.0
        has_prev_region = False
        if len(df_prev) > 0 and "RegionName" in df_prev.columns:
            prev_grp = df_prev[df_prev["RegionName"] == region]
            if len(prev_grp) > 0:
                prev_npa = _npa_pct(prev_grp)
                prev_n = account_count(prev_grp)
                prev_sma2_count = int((prev_grp["curr_bucket"] == "SMA-2").sum()) if "curr_bucket" in prev_grp.columns else 0
                prev_sma2_pct = _safe_div(prev_sma2_count, prev_n)
                has_prev_region = True

        delta = round(curr_npa - prev_npa, 2) if has_prev_region else None
        sma2_delta = round(sma2_pct - prev_sma2_pct, 2) if has_prev_region else None
        status = "-"
        if delta is not None:
            status = "Worsening" if delta > REGION_STATUS_DELTA_PP else ("Improving" if delta < -REGION_STATUS_DELTA_PP else "Stable")

        rows.append({
            "Region": region,
            "SMA-2": sma2_count,
            "SMA-2%": sma2_pct,
            "NPA": npa_count,
            "NPA%": curr_npa,
            "Δ SMA-2%": sma2_delta,
            "Δ NPA%": delta,
            "Collection%": curr_coll,
            "Strike%": strike_pct,
            "SOH (Cr)": soh,
            "Roll Fwd%": roll_fwd,
            "Roll Bwd%": roll_bwd,
            "Status": status,
        })

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("NPA%", ascending=False).reset_index(drop=True)


# ── Section 2b: Overdue vs Month Demand Collection (Region / Branch / Executive) ──

# Single source of truth for "a branch carries Region; an executive carries
# Branch+Region" -- compute_overdue_demand_scorecard's own output shape below.
# Its three consumers (the dashboard table in ui/tabs/portfolio_intelligence.py,
# the report section in report_agent/sections/overdue_demand.py, and the report
# renderer in report_agent/nodes/report_builder.py) used to each hardcode their
# own copy of this exact mapping -- correct, but three independently-maintained
# copies (with inconsistent Title-case/lowercase casing between them) that a
# future 4th grain or a rename would need to update in lockstep with nothing
# enforcing that. Import this dict instead of redefining it.
OVERDUE_DEMAND_IDENTITY_COLS: dict[str, list[str]] = {
    "region": [],
    "branch": ["Region"],
    "executive": ["Branch", "Region"],
}


def compute_overdue_demand_scorecard(df_curr: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """One row per Region / Branch (Unit) / Executive (MNT NAME): Overdue and Month
    Demand collection (amount + %), plus Overall Collection (amount + %) -- each
    computed at that entity's own total via utils.compute_overdue_demand_pct,
    never averaged from per-loan percentages, so a branch with one huge loan and
    nine tiny ones isn't skewed by the nine.

    Overall Collection is deliberately NOT (overdue_collected + demand_collected)
    over (Overdue + MonthDemandExclPC) -- that would be a THIRD, independently
    -derived definition of "collection %", disagreeing with the dashboard's own
    Collection % KPI on real data (confirmed: Net Collection Demand Inst+Exp+BC is
    NOT equal to Arrear Opening + Month Due-Inst/Exp/BC -- it's a distinct
    source-system field with its own logic, not a sum of the two). Overall
    Collection here calls the SAME _coll_pct() this file already uses for the
    Pulse KPI and compute_region_scorecard, so "Overall Collection %" in this
    table is always numerically identical to "Collection %" everywhere else in
    the app, by construction, not by coincidence.

    Branch rows carry their Region alongside (a branch belongs to exactly one
    region); Executive rows carry both Branch and Region -- same identity-column
    pattern compute_npa_sma2_comparison already uses (grp[src].iloc[0], since
    every row in a Unit/MNT NAME group shares the same parent Region/Branch)."""
    def _rows(col: str, label: str, extra_cols: list[tuple[str, str]] | None = None, min_accounts: int = 0) -> pd.DataFrame:
        if col not in df_curr.columns or df_curr.empty:
            return pd.DataFrame()
        extra_cols = extra_cols or []
        rows = []
        for name, grp in df_curr.groupby(col):
            n = account_count(grp)
            if n < min_accounts:
                continue
            stats = compute_overdue_demand_pct(grp)
            overall_paid = to_num(grp, "Month Collection (Excluding Reserve Collection)").sum()
            row = {label: name}
            for src, out_label in extra_cols:
                row[out_label] = grp[src].iloc[0] if src in grp.columns and len(grp) else None
            row.update({
                "Accounts": n,
                "Overdue (Cr)": round(stats["overdue_total"] / 1e7, 2),
                "Overdue Collection (Cr)": round(stats["overdue_collected"] / 1e7, 2),
                "Overdue Collection %": stats["overdue_pct"],
                "Month Demand (Cr)": round(stats["demand_total"] / 1e7, 2),
                "Month Demand Collection (Cr)": round(stats["demand_collected"] / 1e7, 2),
                "Month Demand Collection %": stats["demand_pct"],
                "Overall Collection (Cr)": round(overall_paid / 1e7, 2),
                "Overall Collection %": _coll_pct(grp),
            })
            rows.append(row)
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows).sort_values("Overdue Collection %").reset_index(drop=True)

    return {
        "region":    _rows("RegionName", "Region"),
        "branch":    _rows("Unit", "Branch", extra_cols=[("RegionName", "Region")]),
        "executive": _rows("MNT NAME", "Executive", extra_cols=[("Unit", "Branch"), ("RegionName", "Region")],
                            min_accounts=MIN_ACCOUNTS_OVERDUE_DEMAND_EXECUTIVE),
    }


def compute_overdue_demand_chart(df: pd.DataFrame, label_col: str, title_suffix: str, top_n: int = 15) -> go.Figure:
    """Grouped bar: Overdue Collection % vs Month Demand Collection % per entity.
    Capped to the top_n entities by Overdue exposure (Cr) so a branch/executive
    breakdown with a long tail stays readable -- the underlying table (caller's
    own DataFrame from compute_overdue_demand_scorecard) still shows every row."""
    fig = go.Figure()
    if df.empty or label_col not in df.columns:
        fig.update_layout(title=dict(text="No data available", font=dict(size=13, color="#111")), height=300)
        return fig

    plot_df = df.sort_values("Overdue (Cr)", ascending=False).head(top_n)
    names = plot_df[label_col].tolist()

    fig.add_trace(go.Bar(
        name="Overdue Collection %", x=names, y=plot_df["Overdue Collection %"].tolist(),
        marker=dict(color="#991b1b", line=dict(width=0)),
        text=[f"{v:.1f}%" for v in plot_df["Overdue Collection %"]],
        textposition="outside", cliponaxis=False, textfont=dict(size=10, color="#111"),
        hovertemplate="<b>%{x}</b><br>Overdue Collection: %{y:.1f}%<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        name="Month Demand Collection %", x=names, y=plot_df["Month Demand Collection %"].tolist(),
        marker=dict(color="#1d4ed8", line=dict(width=0)),
        text=[f"{v:.1f}%" for v in plot_df["Month Demand Collection %"]],
        textposition="outside", cliponaxis=False, textfont=dict(size=10, color="#111"),
        hovertemplate="<b>%{x}</b><br>Month Demand Collection: %{y:.1f}%<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text=f"Overdue vs Month Demand Collection %  -  {title_suffix}", font=dict(size=13, color="#111"), x=0),
        barmode="group", bargap=0.25, bargroupgap=0.08,
        plot_bgcolor="white", paper_bgcolor="white",
        xaxis=dict(tickangle=-30, showgrid=False, tickfont=dict(size=11, color="#374151")),
        yaxis=dict(title="Collection %", range=[0, 115], showgrid=True, gridcolor="#f3f4f6", tickfont=dict(color="#6b7280")),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font=dict(size=11)),
        margin=dict(l=20, r=20, t=60, b=90),
        height=400,
    )
    return fig


# ── Section 2: NPA & SMA-2 Comparison (Region / Branch / Executive) ──────────

def compute_npa_sma2_comparison(df_curr: pd.DataFrame, df_prev: pd.DataFrame) -> dict:
    """
    Per-dimension comparison: NPA count + SMA-2 count, curr vs prev, with Δ and Δ%.
    Returns dict with keys 'region', 'branch', 'executive'  -  each a DataFrame.
    """
    has_prev = len(df_prev) > 0

    def _dim_rows(df_c, df_p, col, extra_cols: list[tuple[str, str]] | None = None, prefix_fallback: bool = False, precomputed_curr: dict | None = None):
        """extra_cols: [(source_column, output_label), ...] context columns pulled from the group's first row.

        prefix_fallback: for free-text columns like MNT NAME (manually retyped each month, not a
        controlled vocabulary like RegionName/Unit), the source export sometimes truncates the
        same person's name to a different length between months (e.g. "...DNYANESHWAR F" this
        month vs "...DNYANESHWAR FA" last month). When the exact normalized name has no match,
        fall back to a prefix relationship -- but only when exactly one previous-period name is a
        prefix-match candidate, so two different people with similar names never get merged.

        precomputed_curr: {name: {"npa", "sma2"}} -- when given (the "Unit"/branch call site),
        skip recomputing the current-period NPA/SMA-2 counts inline and use these instead. They
        come from _branch_aggregates, which already did this exact groupby+count pass for
        compute_branch_quadrant -- avoids a second, independent full df_curr.groupby("Unit")
        pass over the same rows for the same numbers. Deliberately does NOT cover Roll Fwd%/Bwd%
        -- _branch_aggregates coerces a None roll rate (no prev_bucket data) to 0.0, but this
        function's own _roll_rates(grp) call below preserves None, and that distinction must
        not change here. Prev-period matching (this function's actual unique logic) is
        untouched either way.
        """
        if col not in df_c.columns:
            return []
        extra_cols = extra_cols or []
        rows = []
        # An exact-string join silently drops any row with case/whitespace drift between this
        # month's and last month's spelling. Normalize the join key (not the displayed name) so
        # "Sunil Waghmare" and "SUNIL WAGHMARE " match.
        def _key(v) -> str:
            return str(v).strip().upper()

        prev_map: dict = {}
        if has_prev and col in df_p.columns and "curr_bucket" in df_p.columns:
            for grp_key, grp in df_p.groupby(col):
                prev_map[_key(grp_key)] = {
                    "npa": int((grp["curr_bucket"] == "NPA").sum()),
                    "sma2": int((grp["curr_bucket"] == "SMA-2").sum()),
                }
        prev_keys = list(prev_map.keys())

        def _lookup(key: str) -> dict:
            if key in prev_map:
                return prev_map[key]
            if not prefix_fallback:
                return {}
            candidates = [k for k in prev_keys if k.startswith(key) or key.startswith(k)]
            return prev_map[candidates[0]] if len(candidates) == 1 else {}

        for grp_key, grp in df_c.groupby(col):
            name = str(grp_key)
            n = account_count(grp)
            if n < MIN_ACCOUNTS_DIMENSION_BREAKDOWN:
                continue
            if precomputed_curr is not None:
                pc = precomputed_curr.get(name)
                if pc is None:
                    continue
                npa_c, sma2_c = pc["npa"], pc["sma2"]
            else:
                npa_c  = int((grp["curr_bucket"] == "NPA").sum())  if "curr_bucket" in grp.columns else 0
                sma2_c = int((grp["curr_bucket"] == "SMA-2").sum()) if "curr_bucket" in grp.columns else 0
            prev   = _lookup(_key(grp_key))
            npa_p  = prev.get("npa")
            sma2_p = prev.get("sma2")

            def _delta(c, p):
                if p is None:
                    return None, None
                d = c - p
                pct = round(d / p * 100, 1) if p > 0 else (100.0 if d > 0 else 0.0)
                return d, pct

            npa_d,  npa_dpct  = _delta(npa_c,  npa_p)
            sma2_d, sma2_dpct = _delta(sma2_c, sma2_p)
            row = {col: name}
            for src, label in extra_cols:
                row[label] = grp[src].iloc[0] if src in grp.columns and len(grp) else None
            row.update({
                "Accounts":         n,
                "SMA-2 (Curr)":     sma2_c,
                "SMA-2 (Prev)":     sma2_p,
                "NPA (Curr)":       npa_c,
                "NPA (Prev)":       npa_p,
                "SMA-2 Δ":         sma2_d,
                "SMA-2 Δ%":        sma2_dpct,
                "NPA Δ":            npa_d,
                "NPA Δ%":          npa_dpct,
            })
            roll_fwd, roll_bwd = _roll_rates(grp)
            row["Roll Fwd%"] = roll_fwd
            row["Roll Bwd%"] = roll_bwd
            rows.append(row)
        return rows

    def _executive_rows(df_c: pd.DataFrame, df_p: pd.DataFrame) -> list[dict]:
        """Executives grouped by (MNT NAME, Unit), not name alone -- two different people who
        happen to share a name in different branches (e.g. two "Rahul Sharma"s, one in branch X
        and one in Y) must never be merged into a single row. Display name is disambiguated as
        "Rahul Sharma (X)" / "Rahul Sharma (Y)", matching compute_executive_recovery's convention.

        Unit is a controlled vocabulary (exact match only); MNT NAME still gets the prefix-
        truncation fallback, but candidates are restricted to the same Unit so it can't match a
        same-named person in a different branch.
        """
        if "MNT NAME" not in df_c.columns:
            return []

        def _key(v) -> str:
            return str(v).strip().upper()

        # Unit has the same case-inconsistency as MNT NAME in real extracts (e.g. "AKOLA" vs
        # "akola" for the same branch) -- group on the normalized value, not the raw column, or
        # the same branch silently fragments into duplicate rows/groups.
        has_unit = "Unit" in df_c.columns
        if has_unit:
            df_c = df_c.assign(_unit_key=df_c["Unit"].map(_key))
        group_cols = ["MNT NAME", "_unit_key"] if has_unit else ["MNT NAME"]

        prev_map: dict = {}
        prev_names_by_unit: dict = {}
        if has_prev and "MNT NAME" in df_p.columns and "curr_bucket" in df_p.columns:
            p_has_unit = "_unit_key" in group_cols and "Unit" in df_p.columns
            if p_has_unit:
                df_p = df_p.assign(_unit_key=df_p["Unit"].map(_key))
            p_group_cols = [c for c in group_cols if c == "MNT NAME" or (c == "_unit_key" and p_has_unit)]
            grouped = df_p.groupby(p_group_cols[0]) if len(p_group_cols) == 1 else df_p.groupby(p_group_cols)
            for grp_key, grp in grouped:
                name_k, unit_k = (_key(grp_key[0]), _key(grp_key[1])) if isinstance(grp_key, tuple) else (_key(grp_key), "")
                prev_map[(name_k, unit_k)] = {
                    "npa": int((grp["curr_bucket"] == "NPA").sum()),
                    "sma2": int((grp["curr_bucket"] == "SMA-2").sum()),
                }
                prev_names_by_unit.setdefault(unit_k, []).append(name_k)

        def _lookup(name_k: str, unit_k: str) -> dict:
            if (name_k, unit_k) in prev_map:
                return prev_map[(name_k, unit_k)]
            candidates = [n for n in prev_names_by_unit.get(unit_k, []) if n.startswith(name_k) or name_k.startswith(n)]
            return prev_map[(candidates[0], unit_k)] if len(candidates) == 1 else {}

        def _delta(c, p):
            if p is None:
                return None, None
            d = c - p
            pct = round(d / p * 100, 1) if p > 0 else (100.0 if d > 0 else 0.0)
            return d, pct

        rows = []
        grouped_c = df_c.groupby(group_cols[0]) if len(group_cols) == 1 else df_c.groupby(group_cols)
        for grp_key, grp in grouped_c:
            raw_name, raw_unit = grp_key if isinstance(grp_key, tuple) else (grp_key, None)
            n = account_count(grp)
            if n < MIN_ACCOUNTS_DIMENSION_BREAKDOWN:
                continue
            npa_c  = int((grp["curr_bucket"] == "NPA").sum())  if "curr_bucket" in grp.columns else 0
            sma2_c = int((grp["curr_bucket"] == "SMA-2").sum()) if "curr_bucket" in grp.columns else 0
            prev   = _lookup(_key(raw_name), _key(raw_unit) if raw_unit is not None else "")
            npa_p  = prev.get("npa")
            sma2_p = prev.get("sma2")
            npa_d,  npa_dpct  = _delta(npa_c,  npa_p)
            sma2_d, sma2_dpct = _delta(sma2_c, sma2_p)
            region = str(grp["RegionName"].iloc[0]) if "RegionName" in grp.columns and len(grp) else None
            row = {
                "MNT NAME": f"{raw_name} ({raw_unit})" if raw_unit else str(raw_name),
                "Unit": raw_unit,
                "Region": region,
                "Accounts": n,
                "SMA-2 (Curr)": sma2_c,
                "SMA-2 (Prev)": sma2_p,
                "NPA (Curr)": npa_c,
                "NPA (Prev)": npa_p,
                "SMA-2 Δ": sma2_d,
                "SMA-2 Δ%": sma2_dpct,
                "NPA Δ": npa_d,
                "NPA Δ%": npa_dpct,
            }
            roll_fwd, roll_bwd = _roll_rates(grp)
            row["Roll Fwd%"] = roll_fwd
            row["Roll Bwd%"] = roll_bwd
            rows.append(row)
        return rows

    result = {}

    rows = _dim_rows(df_curr, df_prev, "RegionName")
    if rows:
        result["region"] = pd.DataFrame(rows).sort_values("NPA (Curr)", ascending=False).reset_index(drop=True)

    _branch_agg = _branch_aggregates(df_curr)
    _branch_curr = (
        {r["Branch"]: {"npa": r["NPA"], "sma2": r["SMA-2"]} for r in _branch_agg.to_dict("records")}
        if not _branch_agg.empty else {}
    )
    rows = _dim_rows(df_curr, df_prev, "Unit", extra_cols=[("RegionName", "Region")], precomputed_curr=_branch_curr)
    if rows:
        result["branch"] = pd.DataFrame(rows).sort_values("NPA (Curr)", ascending=False).reset_index(drop=True)

    if "MNT NAME" in df_curr.columns and "curr_bucket" in df_curr.columns:
        rows = _executive_rows(df_curr, df_prev)
        if rows:
            result["executive"] = pd.DataFrame(rows).sort_values("NPA (Curr)", ascending=False).reset_index(drop=True)

    return result


# ── Section 2: Branch Quadrant ────────────────────────────────────────────────

def _branch_aggregates(df_curr: pd.DataFrame) -> pd.DataFrame:
    """One row per branch (Unit): current-period-only stats shared by
    compute_branch_quadrant and compute_npa_sma2_comparison's branch pass,
    which previously each ran their own independent `df_curr.groupby("Unit")`
    Python loop recomputing the same NPA/SMA-2 counts, Collection%, and roll
    rates. Deliberately CURRENT-period only (no df_prev, no name-matching) --
    compute_npa_sma2_comparison's prev-period fuzzy name-matching logic is
    unique to that function and stays there unchanged; this helper only
    replaces the parts with zero cross-function behavioral risk (plain counts
    and percentages, no reimplementation of a shared correctness-critical
    function like compute_strike_pct).

    Columns: Branch, Region, Accounts, NPA, NPA%, SMA-2, SMA-2%, Collection%,
    Strike%, Hard Bucket%, SOH (Cr), Roll Fwd%, Roll Bwd%, Chronic (3M+).
    Rows below MIN_ACCOUNTS_DIMENSION_BREAKDOWN are excluded (both consumers
    already applied this exact same threshold independently).
    """
    if "Unit" not in df_curr.columns or df_curr.empty:
        return pd.DataFrame()

    rows = []
    for branch, grp in df_curr.groupby("Unit"):
        n = account_count(grp)
        if n < MIN_ACCOUNTS_DIMENSION_BREAKDOWN:
            continue
        roll_fwd, roll_bwd = _roll_rates(grp)
        chronic = int(is_yes(grp, "No Coll 3 Months and >6 EMI").sum())
        npa_n  = int((grp["curr_bucket"] == "NPA").sum())   if "curr_bucket" in grp.columns else 0
        sma2_n = int((grp["curr_bucket"] == "SMA-2").sum()) if "curr_bucket" in grp.columns else 0
        region = str(grp["RegionName"].iloc[0]) if "RegionName" in grp.columns and len(grp) else None
        rows.append({
            "Branch": str(branch),
            "Region": region,
            "Accounts": n,
            "NPA": npa_n,
            "NPA%": _safe_div(npa_n, n),
            "SMA-2": sma2_n,
            "SMA-2%": _safe_div(sma2_n, n),
            "Collection%": _coll_pct(grp),
            "Strike%": compute_strike_pct(grp),
            "Hard Bucket%": compute_hard_bucket_pct(grp),
            "SOH (Cr)": _soh_cr(grp),
            "Roll Fwd%": roll_fwd if roll_fwd is not None else 0.0,
            "Roll Bwd%": roll_bwd if roll_bwd is not None else 0.0,
            "Chronic (3M+)": chronic,
        })
    return pd.DataFrame(rows)


def compute_branch_quadrant(df_curr: pd.DataFrame) -> tuple[pd.DataFrame, go.Figure]:
    """Branch scatter (Collection% vs NPA%, bubble=SOH) + concern score table."""
    agg = _branch_aggregates(df_curr)
    if agg.empty:
        return pd.DataFrame(), go.Figure()

    # NPA% here uses the SAME per-branch NPA-count/account-count ratio as
    # _branch_aggregates -- confirmed identical to the original inline
    # _npa_pct(grp) call (both are npa_count/n*100, _safe_div rounds the same way).
    df = agg[["Branch", "Region", "Accounts", "Collection%", "SMA-2%", "NPA%", "Strike%", "Hard Bucket%", "SOH (Cr)", "Roll Fwd%", "Chronic (3M+)"]].copy()
    for col, w in CONCERN_SCORE_WEIGHTS.items():
        df[f"_r_{col}"] = df[col].rank(ascending=True, pct=True) * w
    df["Concern Score"] = df[[c for c in df.columns if c.startswith("_r_")]].sum(axis=1).mul(100).round(0).astype(int)
    df = df.drop(columns=[c for c in df.columns if c.startswith("_r_")])
    df = df.sort_values("Concern Score", ascending=False).reset_index(drop=True)
    df.insert(0, "Rank", range(1, len(df) + 1))
    return df, _build_quadrant_chart(df)


def _build_quadrant_chart(df: pd.DataFrame) -> go.Figure:
    if df.empty:
        return go.Figure()
    med_coll = df["Collection%"].median()
    med_npa  = df["NPA%"].median()

    def _color(row):
        hi = row["NPA%"] >= med_npa
        lo = row["Collection%"] < med_coll
        if lo and hi:     return "#dc2626"
        if not lo and hi: return "#f97316"
        if lo and not hi: return "#d97706"
        return "#16a34a"

    df = df.copy()
    df["_color"] = df.apply(_color, axis=1)
    soh   = df["SOH (Cr)"].clip(lower=0)
    sizes = (soh / (soh.max() or 1) * 44 + 14).tolist()

    x_vals = df["Collection%"].tolist()
    y_vals = df["NPA%"].tolist()
    x0 = df["Collection%"].min() - 5
    x1 = df["Collection%"].max() + 5
    y0 = max(df["NPA%"].min() - 1, 0)
    y1 = df["NPA%"].max() + 2.5

    fig = go.Figure()

    # Quadrant background shading
    for (xs, xe, ys, ye, fill) in [
        (x0, med_coll, med_npa, y1, "rgba(220,38,38,0.04)"),   # Intervene Now
        (med_coll, x1, med_npa, y1, "rgba(249,115,22,0.04)"),  # Watch
        (x0, med_coll, y0, med_npa, "rgba(217,119,6,0.04)"),   # Underperforming
        (med_coll, x1, y0, med_npa, "rgba(22,163,74,0.04)"),   # Healthy
    ]:
        fig.add_shape(type="rect", x0=xs, x1=xe, y0=ys, y1=ye,
                      fillcolor=fill, line_width=0, layer="below")

    fig.add_vline(x=med_coll, line_dash="dash", line_color="#9ca3af", line_width=1.2)
    fig.add_hline(y=med_npa,  line_dash="dash", line_color="#9ca3af", line_width=1.2)

    for qx, qy, ql, qc in [
        (x0 + 0.5, y1 - 0.4, "🔴 Intervene Now",    "#dc2626"),
        (med_coll + 0.5, y1 - 0.4, "🟠 Watch",       "#f97316"),
        (x0 + 0.5, y0 + 0.2, "🟡 Underperforming",  "#d97706"),
        (med_coll + 0.5, y0 + 0.2, "🟢 Healthy",     "#16a34a"),
    ]:
        fig.add_annotation(x=qx, y=qy, text=f"<b>{ql}</b>", showarrow=False,
                           font=dict(size=11, color=qc), opacity=0.8, xanchor="left")

    # Bubble labels: "BranchName · NPA X.X%"
    bubble_text = [
        f"{row['Branch']}<br><b>{row['NPA%']:.1f}% NPA</b>"
        for _, row in df.iterrows()
    ]
    hover_text = [
        f"<b>{row['Branch']}</b><br>"
        f"Collection: {row['Collection%']:.1f}%<br>"
        f"SMA-2: {row['SMA-2%']:.1f}%<br>"
        f"NPA: {row['NPA%']:.1f}%<br>"
        f"SOH: ₹{row['SOH (Cr)']:.2f}Cr<br>"
        f"Concern Score: {row['Concern Score']}"
        for _, row in df.iterrows()
    ]

    fig.add_trace(go.Scatter(
        x=x_vals, y=y_vals,
        mode="markers+text",
        marker=dict(
            size=sizes, color=df["_color"].tolist(),
            opacity=0.88, line=dict(width=1.5, color="#fff"),
        ),
        text=bubble_text,
        textposition="top center",
        textfont=dict(size=9, color="#111"),
        hovertext=hover_text,
        hoverinfo="text",
    ))

    fig.update_layout(
        title=dict(text="Branch Quadrant  -  Collection% vs NPA%  (bubble size = SOH)", font=dict(size=13, color="#111"), x=0),
        xaxis=dict(title="Collection %", showgrid=True, gridcolor="#f0f0f0",
                   tickfont=dict(color="#374151"), range=[x0, x1]),
        yaxis=dict(title="NPA %", showgrid=True, gridcolor="#f0f0f0",
                   tickfont=dict(color="#374151"), range=[y0, y1]),
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(l=30, r=30, t=55, b=50),
        height=470,
        showlegend=False,
    )
    return fig


# ── Section 2: Executive Recovery Leaderboard ─────────────────────────────────

def compute_executive_recovery(df_curr: pd.DataFrame) -> pd.DataFrame:
    """Executives ranked by net accounts rescued from high-risk buckets."""
    if (
        "MNT NAME" not in df_curr.columns
        or "prev_bucket" not in df_curr.columns
        or "curr_bucket" not in df_curr.columns
    ):
        return pd.DataFrame()

    group_cols = ["MNT NAME", "Unit"] if "Unit" in df_curr.columns else ["MNT NAME"]
    rows = []
    for keys, grp in df_curr.groupby(group_cols):
        exec_name = str(keys[0]) if isinstance(keys, tuple) else str(keys)
        branch    = str(keys[1]) if isinstance(keys, tuple) and len(keys) > 1 else ""
        n = account_count(grp)
        if n < MIN_ACCOUNTS_DIMENSION_BREAKDOWN:
            continue
        curr_sc = grp["curr_bucket"].map(BUCKET_SCORE)
        prev_sc = grp["prev_bucket"].map(BUCKET_SCORE)
        valid   = curr_sc.notna() & prev_sc.notna()
        if not valid.any():
            continue
        was_risk = grp["prev_bucket"].isin(["NPA", "SMA-2", "SMA-1"])
        rescued  = int((valid & was_risk & (curr_sc < prev_sc)).sum())
        slipped  = int((valid & (curr_sc > prev_sc)).sum())
        region   = str(grp["RegionName"].iloc[0]) if "RegionName" in grp.columns and len(grp) else ""
        rows.append({
            "Executive": f"{exec_name} ({branch})" if branch else exec_name,
            "Branch": branch,
            "Region": region,
            "Accounts": n,
            "Collection%": _coll_pct(grp),
            "Strike%": compute_strike_pct(grp),
            "Rescued": rescued,
            "Slipped": slipped,
            "Net Recovery": rescued - slipped,
        })

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("Net Recovery", ascending=False).reset_index(drop=True)


# ── Section 3: Good vs Bad ────────────────────────────────────────────────────

def compute_good_bad(
    region_df: pd.DataFrame,
    branch_df: pd.DataFrame,
    risk_indicators: list[dict],
    exec_df: pd.DataFrame,
    has_prev: bool,
) -> dict:
    good: list[str] = []
    bad:  list[str] = []

    if has_prev and not region_df.empty and "Δ NPA%" in region_df.columns:
        imp = region_df.dropna(subset=["Δ NPA%"]).query(f"`Δ NPA%` < {-GOOD_BAD_REGION_DELTA_PP}").sort_values("Δ NPA%")
        wor = region_df.dropna(subset=["Δ NPA%"]).query(f"`Δ NPA%` > {GOOD_BAD_REGION_DELTA_PP}").sort_values("Δ NPA%", ascending=False)
        # Use .iterrows() (not itertuples) - "Δ NPA%"/"SMA-2%" etc. aren't valid
        # Python identifiers, so itertuples() silently renames them to positional
        # _N attrs and _4 does NOT reliably point at "Δ NPA%".
        for _, r in imp.head(2).iterrows():
            good.append(f"{r['Region']}: NPA% fell {abs(r['Δ NPA%']):.1f}pp  -  delinquency improving")
        for _, r in wor.head(2).iterrows():
            bad.append(f"{r['Region']}: NPA% rose {r['Δ NPA%']:.1f}pp  -  escalate field visits")

    if not branch_df.empty and "Concern Score" in branch_df.columns:
        worst = branch_df.iloc[0]
        best  = branch_df.iloc[-1]
        if worst["Concern Score"] >= CONCERN_SCORE_BAD_THRESHOLD:
            bad.append(f"{worst['Branch']}: Highest concern ({worst['Concern Score']})  -  SMA-2 {worst.get('SMA-2%',0):.1f}%, NPA {worst['NPA%']:.1f}%, Coll {worst['Collection%']:.1f}%")
        if best["Concern Score"] <= CONCERN_SCORE_GOOD_THRESHOLD:
            good.append(f"{best['Branch']}: Healthiest branch  -  SMA-2 {best.get('SMA-2%',0):.1f}%, NPA {best['NPA%']:.1f}%, Coll {best['Collection%']:.1f}%")

    if not exec_df.empty and has_prev:
        top_exec  = exec_df[exec_df["Net Recovery"] > 0].head(1)
        bad_exec  = exec_df[exec_df["Net Recovery"] < 0].tail(1)
        if len(top_exec) > 0:
            r = top_exec.iloc[0]
            good.append(f"{r['Executive']}: rescued {r['Rescued']} accounts from NPA/SMA")
        if len(bad_exec) > 0:
            r = bad_exec.iloc[0]
            bad.append(f"{r['Executive']}: net {abs(r['Net Recovery'])} accounts slipped vs rescued  -  portfolio deteriorating")

    for ind in risk_indicators:
        d = ind["_direction"]
        delta_abs = abs(ind.get("_delta", 0))
        threshold = RISK_INDICATOR_MATERIALITY_COUNT if ind.get("_is_count") else RISK_INDICATOR_MATERIALITY_PP
        if d == "Improving" and delta_abs >= threshold:
            # Strip leading sign  -  the word "improvement" already implies positive change
            delta_display = str(ind["Δ"]).lstrip("+-").lstrip()
            good.append(f"{ind['Signal']}: {delta_display} improvement")
        elif d == "Worsening" and delta_abs >= threshold:
            bad.append(f"{ind['Signal']}: {ind['Δ']}  -  {ind['Note']}")

    return {"good": good[:6], "bad": bad[:6]}


# ── Section 4: Risk Flag Comparison ──────────────────────────────────────────

def compute_risk_flag_comparison(alerts_curr: list, alerts_prev: list) -> pd.DataFrame:
    """Merge curr and prev alert counts into a comparison table."""
    if not alerts_curr:
        return pd.DataFrame()

    prev_map = {a["title"]: a for a in alerts_prev} if alerts_prev else {}

    rows = []
    for a in alerts_curr:
        title = a["title"]
        cnt_c = a["count"]
        soh_c = round(a.get("pos", 0) / 1e7, 2)
        prev  = prev_map.get(title)
        cnt_p = prev["count"] if prev else None
        delta = (cnt_c - cnt_p) if cnt_p is not None else None
        rows.append({
            "Risk Type": title,
            "Accounts": cnt_c,
            "SOH (Cr)": soh_c,
            "Last Month": cnt_p,
            "Δ": delta,
            "Severity": a.get("severity", "medium"),
            "Action": a.get("action", ""),
            "_df_key": title,
        })
    return pd.DataFrame(rows)


# ── Section 5: Product / Segment Analysis ─────────────────────────────────────

def compute_product_analysis(df_curr: pd.DataFrame, as_of=None) -> dict:
    """as_of: the report's OWN reporting month/date (e.g. app.py's Reporting
    Month picker, or report_agent's curr_month) -- NOT necessarily today's real
    date. Defaults to the real wall-clock date only when the caller doesn't
    have a reporting date to hand (e.g. a script or test calling this
    directly). Without this, "exclude post-dated agreement dates" silently
    compared against whatever day the code happened to RUN rather than the
    month the upload is actually reporting on -- re-analyzing an old file
    (e.g. a March extract opened in July) would wrongly exclude/include
    cohorts relative to July, not March."""
    results: dict[str, pd.DataFrame] = {}

    seg_col = next((c for c in ["SegmentName", "Segment"] if c in df_curr.columns), None)
    if seg_col:
        rows = _group_npa_table(df_curr, seg_col, "Segment", min_n=MIN_ACCOUNTS_PRODUCT_SEGMENT)
        if rows:
            results["segment"] = pd.DataFrame(rows).sort_values("NPA%", ascending=False).reset_index(drop=True)

    if "FUEL_TYPE" in df_curr.columns:
        rows = _group_npa_table(df_curr, "FUEL_TYPE", "Fuel Type", min_n=MIN_ACCOUNTS_PRODUCT_SEGMENT)
        if rows:
            results["fuel"] = pd.DataFrame(rows).sort_values("NPA%", ascending=False).reset_index(drop=True)

    if "Ag_Date" in df_curr.columns:
        df_v = df_curr.copy()
        df_v["_cohort"] = pd.to_datetime(df_v["Ag_Date"], errors="coerce").dt.to_period("M")
        _ref_date = pd.Timestamp(as_of) if as_of is not None else pd.Timestamp.now()
        _today_period = _ref_date.to_period("M")
        cohort_rows = []
        for cohort, grp in df_v.dropna(subset=["_cohort"]).groupby("_cohort"):
            if cohort > _today_period:
                continue  # exclude post-dated / future agreement dates
            n = account_count(grp)
            if n < MIN_ACCOUNTS_SOURCE_VINTAGE:
                continue
            soh = _soh_cr(grp)
            avg_loan = to_num(grp, "Loan Amount").mean()
            npa_n  = int((grp["curr_bucket"] == "NPA").sum())   if "curr_bucket" in grp.columns else 0
            sma2_n = int((grp["curr_bucket"] == "SMA-2").sum()) if "curr_bucket" in grp.columns else 0
            cohort_rows.append({
                "Disbursement Month": str(cohort),
                "Accounts":   n,
                "NPA Count":  npa_n,
                "SMA-2 Count": sma2_n,
                "NPA%":        _safe_div(npa_n, n),
                "SMA-2%":      _safe_div(sma2_n, n),
                "Collection%": _coll_pct(grp),
                "SOH (Cr)":    soh,
                "Avg Loan (L)": round(avg_loan / 1e5, 2) if not pd.isna(avg_loan) else 0.0,
            })
        if cohort_rows:
            results["vintage"] = (
                pd.DataFrame(cohort_rows)
                .sort_values("Disbursement Month", ascending=False)
                .reset_index(drop=True)
            )

    if "SRC Name" in df_curr.columns:
        rows = _group_npa_table(df_curr, "SRC Name", "Source", min_n=MIN_ACCOUNTS_SOURCE_VINTAGE)
        if rows:
            results["source"] = pd.DataFrame(rows).sort_values("NPA%", ascending=False).reset_index(drop=True)

    return results


# ── New Advances (Business/Originations) ──────────────────────────────────────

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

    ref = pd.Timestamp(as_of) if as_of is not None else pd.Timestamp.today()
    curr_ym = ref.to_period("M")
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

    seg_col = next((c for c in ["SegmentName", "Segment"] if c in curr_adv.columns), None)
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
                "Share %": _safe_div(n, curr_n),
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


NEW_ADVANCES_IDENTITY_COLS: dict[str, list[str]] = {
    "region": [],
    "branch": ["Region"],
    "executive": ["Branch", "Region"],
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
    empty = {"region": pd.DataFrame(), "branch": pd.DataFrame(), "executive": pd.DataFrame()}
    if df_curr.empty or "Ag_Date" not in df_curr.columns:
        return empty

    ref = pd.Timestamp(as_of) if as_of is not None else pd.Timestamp.today()
    curr_ym = ref.to_period("M")
    prev_ym = curr_ym - 1
    ag = pd.to_datetime(df_curr["Ag_Date"], errors="coerce")
    cohort = ag.dt.to_period("M")
    curr_adv = df_curr[cohort == curr_ym]
    prev_adv = df_curr[cohort == prev_ym]

    def _totals(adv):
        n = account_count(adv)
        amt = float(to_num(adv, "Loan Amount").sum()) if "Loan Amount" in adv.columns else 0.0
        return n, amt

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
        rows = []
        for name, grp in curr_adv.groupby(col):
            n, amt = _totals(grp)
            if n < min_accounts:
                continue
            prev_grp = prev_adv[prev_adv[col] == name] if col in prev_adv.columns and not prev_adv.empty else prev_adv.iloc[0:0]
            prev_n, prev_amt = _totals(prev_grp)
            row = {label: name}
            for src, out_label in extra_cols:
                row[out_label] = grp[src].iloc[0] if src in grp.columns and len(grp) else None
            row.update({
                "Total Accounts": int(total_counts.get(name, n)),
                "Accounts This Month": n,
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
        "region":    _rows("RegionName", "Region"),
        "branch":    _rows("Unit", "Branch", extra_cols=[("RegionName", "Region")]),
        # Deliberately NO materiality floor here (min_accounts=0, every
        # executive with >=1 new advance shows up) -- unlike the Overdue vs
        # Month Demand league table, this isn't ranking collection
        # performance where a single account can swing a %; it's a business/
        # origination view where even one new advance is real information.
        "executive": _rows("MNT NAME", "Executive", extra_cols=[("Unit", "Branch"), ("RegionName", "Region")],
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
    _period_sort_key with ui/tabs/portfolio_intelligence.py's _roll_vintage,
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
        title=dict(text="New Advances Trend  -  Accounts &amp; Funded Amount by Period", font=dict(size=13, color="#111"), x=0),
        plot_bgcolor="white", paper_bgcolor="white",
        xaxis=dict(tickangle=-45, showgrid=False, tickfont=dict(size=10, color="#374151")),
        yaxis=dict(title="Accounts", showgrid=True, gridcolor="#f3f4f6", tickfont=dict(color="#6b7280")),
        yaxis2=dict(title="Funded (Cr)", overlaying="y", side="right", showgrid=False, tickfont=dict(color="#6b7280")),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font=dict(size=11)),
        margin=dict(l=20, r=20, t=60, b=90),
        height=380,
    )
    return fig


def _group_npa_table(df: pd.DataFrame, group_col: str, label: str, min_n: int) -> list:
    """Vectorized: one groupby(group_col).agg(...) pass over the WHOLE df,
    not a Python-level call (account_count/_npa_pct/_coll_pct/_soh_cr, each
    re-slicing the group) per distinct value -- confirmed via profiling to be
    the dominant cost of compute_product_analysis on a real ~60k-row file
    (called 3x for segment/fuel/source, ~4.3s combined). has_loan_no mirrors
    _npa_pct's own "Loan No" column-presence gate exactly (returns 0.0 NPA%
    if absent, regardless of account count), rather than silently reusing
    whatever denominator account_count's own len(df) fallback would give."""
    if group_col not in df.columns:
        return []
    has_loan_no = "Loan No" in df.columns
    has_bucket  = "curr_bucket" in df.columns

    d = df[[group_col]].copy()
    d["_is_sma2"]       = (df["curr_bucket"] == "SMA-2") if has_bucket else False
    d["_is_npa"]        = (df["curr_bucket"] == "NPA") if has_bucket else False
    d["_demand_num"]    = to_num(df, "Net Collection Demand Inst+Exp+BC")
    d["_collected_num"] = to_num(df, "Month Collection (Excluding Reserve Collection)")
    d["_soh_num"]       = to_num(df, "SOH")
    if has_loan_no:
        d["Loan No"] = df["Loan No"]

    n_series = d.groupby(group_col)["Loan No"].nunique() if has_loan_no else d.groupby(group_col).size()
    agg = d.groupby(group_col).agg(
        sma2_n=("_is_sma2", "sum"), npa_n=("_is_npa", "sum"),
        demand=("_demand_num", "sum"), collected=("_collected_num", "sum"), soh=("_soh_num", "sum"),
    )

    rows = []
    for val, n in n_series.items():
        val_str = str(val).strip()
        if not val_str or val_str.lower() in ("nan", "none", ""):
            continue
        if n < min_n:
            continue
        a = agg.loc[val]
        rows.append({
            label: val_str,
            "Accounts": int(n),
            "SMA-2%": _safe_div(a["sma2_n"], n),
            "NPA%": _safe_div(a["npa_n"], n) if has_loan_no else 0.0,
            "Collection%": _safe_div(a["collected"], a["demand"]),
            "SOH (Cr)": round(a["soh"] / 1e7, 2),
        })
    return rows


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


# ── Section 5: Risk Indicators ────────────────────────────────────────────────

def compute_risk_indicators(
    df_curr: pd.DataFrame,
    df_prev: pd.DataFrame,
    rr_meta: dict | None,
) -> list[dict]:
    indicators: list[dict] = []
    total_prev = account_count(df_prev) if len(df_prev) > 0 else 0
    has_prev = total_prev > 0

    def _add(label, curr_val, prev_val, unit, good_dir, note, is_count=False):
        # prev_val=None means "no real prior-period value exists for this metric"
        # (e.g. Fresh NPA Formation, a same-period roll-rate figure with nothing
        # genuine to compare against) -- distinct from has_prev=False (no prev file
        # uploaded at all). Treating None as if prev were 0 would make delta equal
        # curr_val itself, so every non-zero reading falsely renders as "Worsening".
        has_cmp = has_prev and prev_val is not None
        delta = round(curr_val - prev_val, 2) if has_cmp else 0.0
        if not has_cmp:
            direction = " - "
        elif is_count:
            direction = ("Improving" if delta < 0 else ("Worsening" if delta > 0 else "Stable")) if good_dir == "down" else ("Improving" if delta > 0 else ("Worsening" if delta < 0 else "Stable"))
        else:
            direction = "Stable" if abs(delta) < RISK_INDICATOR_STABLE_PP else (("Improving" if delta < 0 else "Worsening") if good_dir == "down" else ("Improving" if delta > 0 else "Worsening"))
        fmt      = f"{int(curr_val):,}{unit}" if is_count else f"{curr_val:.1f}{unit}"
        prev_fmt = (f"{int(prev_val):,}{unit}" if is_count else f"{prev_val:.1f}{unit}") if has_cmp else " - "
        sign     = "+" if delta >= 0 else ""
        delta_s  = (f"{sign}{int(delta)}{unit}" if is_count else f"{sign}{delta:.1f}{unit}") if has_cmp else " - "
        indicators.append({
            "Signal": label, "This Month": fmt, "Last Month": prev_fmt,
            "Δ": delta_s, "Direction": direction, "Note": note,
            "_delta": delta, "_direction": direction, "_good": good_dir,
            "_is_count": is_count,
        })

    if "curr_bucket" in df_curr.columns:
        def _pct(df, bucket):
            n = (df["curr_bucket"] == bucket).sum()
            return _safe_div(n, account_count(df))

        _add("SMA-1 Pool (Early Warning)", _pct(df_curr, "SMA-1"),
             _pct(df_prev, "SMA-1") if has_prev else 0.0,
             "%", "down", "Rising SMA-1 predicts NPA formation 1-2 months out")
        _add("SMA-2 Pool (Potential NPA)", _pct(df_curr, "SMA-2"),
             _pct(df_prev, "SMA-2") if has_prev else 0.0,
             "%", "down", "Handle SMA-2 now to prevent NPA  -  2+ EMI overdue, last intervention window")
        _add("NPA Pool", _pct(df_curr, "NPA"),
             _pct(df_prev, "NPA") if has_prev else 0.0,
             "%", "down", "Current NPA accounts as % of total portfolio")

    if rr_meta and rr_meta.get("matched_count", 0) > 0:
        # No real prior-period formation rate exists to compare against (it would
        # require a THIRD month's data) -- pass prev_val=None so this renders as a
        # standalone reading, not a fabricated "Worsening" trend every month.
        _add("Fresh NPA Formation", rr_meta["npa_formation_rate"], None, "%", "down",
             "Non-NPA accounts that became NPA this month  -  more important than total NPA count")

    col3m = "No Coll 3 Months and >6 EMI"
    if col3m in df_curr.columns:
        c = int(is_yes(df_curr, col3m).sum())
        p = int(is_yes(df_prev, col3m).sum()) if has_prev and col3m in df_prev.columns else 0
        _add("Chronic Defaulters (3M+)", c, p, "", "down", "Zero payment ≥3 months AND >6 EMI arrears", is_count=True)

    if "Non Starter" in df_curr.columns:
        c = int(is_yes(df_curr, "Non Starter").sum())
        p = int(is_yes(df_prev, "Non Starter").sum()) if has_prev and "Non Starter" in df_prev.columns else 0
        _add("Non-Starters", c, p, "", "down", "Never paid first EMI  -  highest NPA risk", is_count=True)

    if "CoLending_Loans" in df_curr.columns and "Arrears / EMI" in df_curr.columns:
        c = int((is_yes(df_curr, "CoLending_Loans") & (to_num(df_curr, "Arrears / EMI") > 0)).sum())
        p_val = 0
        if has_prev and "CoLending_Loans" in df_prev.columns and "Arrears / EMI" in df_prev.columns:
            p_val = int((is_yes(df_prev, "CoLending_Loans") & (to_num(df_prev, "Arrears / EMI") > 0)).sum())
        _add("Co-Lending At Risk", c, p_val, "", "down", "Partner-bank loans showing delinquency", is_count=True)

    return indicators


# ── Section 6: Concentration & Exposure ───────────────────────────────────────

def compute_concentration_treemap(df_curr: pd.DataFrame) -> go.Figure:
    """
    Two-level treemap: Region → Branch.
    Size = SOH (Cr). Color = NPA% (green→red).

    Values are built leaf-up (branch → region → root) so each parent value
    equals the exact sum of its children  -  avoiding the branchvalues="total"
    blank-render bug caused by rounding drift when _soh_cr rounds independently
    at each level.
    """
    if "RegionName" not in df_curr.columns or "Unit" not in df_curr.columns:
        return go.Figure()

    def _raw_soh(df: pd.DataFrame) -> float:
        return float(to_num(df, "SOH").sum() / 1e7)

    ids, labels, parents, values, colors, texts = ["portfolio"], ["Portfolio"], [""], [0.0], [0.0], [""]

    portfolio_total = 0.0
    for region, rgrp in df_curr.groupby("RegionName"):
        r_id  = f"r_{region}"
        r_npa = _npa_pct(rgrp)

        # Compute branches first, accumulate their sum for the region value
        branch_sum = 0.0
        branch_buf: list[tuple] = []
        for branch, bgrp in rgrp.groupby("Unit"):
            b_soh = _raw_soh(bgrp)
            b_npa = _npa_pct(bgrp)
            b_id  = f"b_{region}_{branch}"
            branch_buf.append((b_id, str(branch), r_id, b_soh, b_npa,
                                f"{branch}<br>SOH ₹{b_soh:.1f}Cr<br>NPA {b_npa:.1f}%"))
            branch_sum += b_soh

        # Region value = exact branch sum (no independent rounding)
        ids.append(r_id); labels.append(str(region)); parents.append("portfolio")
        values.append(branch_sum); colors.append(r_npa)
        texts.append(f"{region}<br>SOH ₹{branch_sum:.1f}Cr<br>NPA {r_npa:.1f}%")
        portfolio_total += branch_sum

        for (bid, bl, bp, bsoh, bnpa, btxt) in branch_buf:
            ids.append(bid); labels.append(bl); parents.append(bp)
            values.append(bsoh); colors.append(bnpa); texts.append(btxt)

    # Root value = exact sum of region values
    values[0] = portfolio_total

    if len(ids) < 3:
        return go.Figure()

    leaf_npas = [c for c in colors[1:] if isinstance(c, (int, float))]
    cmax = max(max(leaf_npas, default=0), 1.0)

    fig = go.Figure(go.Treemap(
        ids=ids, labels=labels, parents=parents, values=values,
        customdata=texts,
        hovertemplate="%{customdata}<extra></extra>",
        texttemplate="%{label}",
        textfont=dict(size=12),
        marker=dict(
            colors=colors,
            colorscale=[[0, "#16a34a"], [0.1, "#86efac"], [0.3, "#fef08a"], [0.6, "#f97316"], [1.0, "#991b1b"]],
            cmin=0, cmax=cmax,
            showscale=True,
            colorbar=dict(title="NPA %", thickness=12, len=0.6),
        ),
        branchvalues="total",
        pathbar=dict(visible=True),
    ))
    fig.update_layout(
        title=dict(text="Concentration Map: Region → Branch  (size = SOH, color = NPA%)", font=dict(size=13, color="#000")),
        margin=dict(l=10, r=10, t=50, b=10),
        height=420,
        paper_bgcolor="white",
    )
    return fig


def _grouped_mode(df: pd.DataFrame, group_col: str, value_col: str) -> pd.Series:
    """Vectorized per-group mode, matching Series.mode().iat[0]'s own tie-break
    (pandas' mode() returns every tied value sorted ascending; .iat[0] takes
    the smallest) -- one groupby+sort+drop_duplicates pass over the WHOLE
    column, replacing a Python-level .mode() call paid once per group (each
    itself an O(n log n) sort) -- confirmed via profiling to be the dominant
    cost of compute_fleet_exposure on a real ~60k-row file (~1,600+ fleet
    customers x 2 mode() calls each, ~2.7s). Groups with no non-null
    value_col rows are simply absent from the result -- callers already
    handle a missing key via .get(...)/reindex + fillna("")."""
    sub = df[[group_col, value_col]].dropna(subset=[value_col])
    if sub.empty:
        return pd.Series(dtype=object)
    counts = sub.groupby([group_col, value_col], observed=True).size().rename("_n").reset_index()
    counts = counts.sort_values([group_col, "_n", value_col], ascending=[True, False, True])
    winners = counts.drop_duplicates(subset=[group_col], keep="first")
    return winners.set_index(group_col)[value_col]


def compute_fleet_exposure(df_curr: pd.DataFrame) -> dict:
    """
    Customers (identified by Cust Mob No) with 3+ loans  -  fleet operators.
    Returns summary dict + top_df (fleet customers ranked by total SOH).
    NOTE: Cust Mob No may not be unique across branches  -  treat counts as approximate.
    """
    if "Cust Mob No" not in df_curr.columns or "Loan No" not in df_curr.columns:
        return {"count": 0, "total_soh_cr": 0.0, "npa_operators": 0, "top_df": pd.DataFrame()}

    cust_loan_counts = df_curr.groupby("Cust Mob No")["Loan No"].nunique()
    fleet_customers  = cust_loan_counts[cust_loan_counts >= FLEET_MIN_LOANS].index

    fleet_df = df_curr[df_curr["Cust Mob No"].isin(fleet_customers)]
    if fleet_df.empty:
        return {"count": 0, "total_soh_cr": 0.0, "npa_operators": 0, "top_df": pd.DataFrame()}

    n_operators = fleet_customers.nunique()
    total_soh   = _soh_cr(fleet_df)

    # Fleet operators with at least 1 NPA loan
    npa_ops = 0
    if "curr_bucket" in fleet_df.columns:
        has_npa = fleet_df[fleet_df["curr_bucket"] == "NPA"].groupby("Cust Mob No").size()
        npa_ops = len(has_npa)

    # Top fleet customers by SOH -- vectorized (see _grouped_mode's docstring
    # for why this replaced a per-customer Python loop). first_rows uses
    # .head(1) per group (NOT .first(), which silently skips NaN and would
    # disagree with the original .iloc[0] on a customer whose first row's
    # Cust Name happens to be blank).
    fleet_df = fleet_df.copy()
    fleet_df["_soh_num"] = to_num(fleet_df, "SOH")
    fleet_df["_is_npa"]  = (fleet_df["curr_bucket"] == "NPA") if "curr_bucket" in fleet_df.columns else False

    grp_loans  = fleet_df.groupby("Cust Mob No")["Loan No"].nunique()
    grp_soh    = (fleet_df.groupby("Cust Mob No")["_soh_num"].sum() / 1e7).round(2)
    grp_npa    = fleet_df.groupby("Cust Mob No")["_is_npa"].sum().astype(int)
    first_rows = fleet_df.groupby("Cust Mob No", sort=False).head(1).set_index("Cust Mob No")
    grp_name   = first_rows["Cust Name"] if "Cust Name" in first_rows.columns else pd.Series(dtype=object)
    # A fleet customer's loans CAN span multiple regions/branches (Cust Mob
    # No isn't guaranteed unique per branch, per this function's own
    # docstring) -- the most common (mode) region/branch is shown as a
    # single representative value, not necessarily every branch this
    # customer touches. Same convention report_agent's own fleet section
    # used to compute independently; now the single source of truth.
    grp_region = _grouped_mode(fleet_df, "Cust Mob No", "RegionName") if "RegionName" in fleet_df.columns else pd.Series(dtype=object)
    grp_unit   = _grouped_mode(fleet_df, "Cust Mob No", "Unit") if "Unit" in fleet_df.columns else pd.Series(dtype=object)

    # "Cust Name" missing entirely -> str(mob) for every customer (matches the
    # original per-row `... if "Cust Name" in grp.columns else str(mob)`).
    # "Cust Name" present but blank for a given customer's first row -> keep
    # that blank as-is, same as the original `.iloc[0]` (no per-row fallback).
    customer_col = grp_name.reindex(fleet_customers) if "Cust Name" in fleet_df.columns else pd.Series(fleet_customers, index=fleet_customers)

    top_df = pd.DataFrame({
        "Customer":       customer_col,
        "Mobile":         pd.Series(fleet_customers, index=fleet_customers).astype(str),
        "Region":         grp_region.reindex(fleet_customers).fillna(""),
        "Unit":           grp_unit.reindex(fleet_customers).fillna(""),
        "Loans":          grp_loans.reindex(fleet_customers),
        "NPA Loans":      grp_npa.reindex(fleet_customers).fillna(0).astype(int),
        "Total SOH (Cr)": grp_soh.reindex(fleet_customers),
    })
    top_df = (
        top_df
        .sort_values("Total SOH (Cr)", ascending=False)
        .head(20)
        .reset_index(drop=True)
    )
    # Python's str(), not pandas' .astype(str) -- under pandas 3.x's default
    # string dtype, .astype(str) preserves NaN as a null instead of the
    # literal string "nan" that the original code's `str(cust_name)` (a
    # per-row Python builtin call) always produced. Only 20 rows by now
    # (after head(20)), so a per-row .apply(str) here is negligible cost --
    # this is about matching Python's str() semantics exactly, not performance.
    top_df["Customer"] = top_df["Customer"].apply(str)

    return {
        "count": n_operators,
        "total_soh_cr": total_soh,
        "npa_operators": npa_ops,
        "top_df": top_df,
    }


def compute_top_accounts(df_curr: pd.DataFrame, n: int = 20) -> tuple[pd.DataFrame, dict]:
    """
    Top N accounts by SOH among DELINQUENT accounts (any non-STD bucket)  -  the
    largest single exposures actually at risk, not just the largest loans overall
    (a big healthy STD loan isn't a collections priority).

    Returns (top_df, summary): total SOH of these N accounts, what % of the
    whole portfolio's SOH they represent (single-borrower concentration), and
    how many are already NPA  -  the most severe subset of an already-delinquent
    list.
    """
    empty_summary = {"total_soh_cr": 0.0, "pct_of_portfolio": 0.0, "npa_count": 0}
    if df_curr.empty or "SOH" not in df_curr.columns or "curr_bucket" not in df_curr.columns:
        return pd.DataFrame(), empty_summary

    # NA = missing Arrears/EMI data (not a real delinquency state) - exclude it
    # alongside STD, same as VALID_BUCKETS elsewhere in this module.
    delinquent = df_curr[~df_curr["curr_bucket"].isin(["STD", "NA"])]
    if delinquent.empty:
        return pd.DataFrame(), empty_summary

    seg_col = next((c for c in ["SegmentName", "Segment"] if c in delinquent.columns), None)
    cols = [c for c in [
        "Loan No", "Cust Name", "Cust Mob No", "RegionName", "Unit", seg_col, "MNT NAME",
        "curr_bucket", "SOH", "Closing Arrears", "Arrears / EMI", "VehEMI Accrued", "LCC%",
        "Loan Status", "Ag_Date", "Non Starter", "CoLending_Loans",
    ] if c and c in delinquent.columns]

    top_df = (
        delinquent[cols]
        .assign(SOH=lambda d: to_num(d, "SOH"))
        .sort_values("SOH", ascending=False)
        .head(n)
        .reset_index(drop=True)
    )

    total_portfolio_soh = to_num(df_curr, "SOH").sum()
    top_soh = top_df["SOH"].sum()
    npa_count = int((top_df["curr_bucket"] == "NPA").sum())
    summary = {
        "total_soh_cr": round(top_soh / 1e7, 2),
        "pct_of_portfolio": round(top_soh / total_portfolio_soh * 100, 1) if total_portfolio_soh > 0 else 0.0,
        "npa_count": npa_count,
    }
    return top_df, summary


# ── Repossession Analysis ─────────────────────────────────────────────────────

_REPO_DISPLAY_COLS = [
    "Loan No", "Cust Name", "Cust Mob No", "RegionName", "Unit", "MNT NAME",
    "curr_bucket", "Ag_Date", "SOH", "LCC%", "Arrears / EMI",
    "Closing Arrears", "Loan Amount", "Vehicle Description", "FUEL_TYPE",
    "Veh ID", "Last Receipt Date", "Last Receipt Amount",
]


def compute_repossession_list(df_curr: pd.DataFrame, as_of=None) -> pd.DataFrame:
    """
    Accounts eligible for repossession:
      - curr_bucket in ["SMA-2", "NPA"]  (2+ EMI overdue, deep delinquent)
      - Ag_Date within last 18 months    (recent loans  -  still have collateral value)

    as_of: the report's OWN reporting month/date, same reasoning as
    compute_product_analysis's as_of -- defaults to the real wall-clock date
    only when the caller has no reporting date to hand. Without this, the
    "still within the repossession window" cutoff silently measured backward
    from whatever day the code happened to RUN, not the month the upload is
    reporting on -- wrong for any retroactive/historical analysis.

    Returns cleaned DataFrame. Caller sorts for the 3 views.
    """
    if df_curr.empty:
        return pd.DataFrame()

    _ref_date = pd.Timestamp(as_of) if as_of is not None else pd.Timestamp.today()
    cutoff = _ref_date - pd.DateOffset(months=REPOSSESSION_WINDOW_MONTHS)

    bucket_mask = pd.Series(False, index=df_curr.index)
    if "curr_bucket" in df_curr.columns:
        bucket_mask = df_curr["curr_bucket"].isin(["SMA-2", "NPA"])

    date_mask = pd.Series(True, index=df_curr.index)
    if "Ag_Date" in df_curr.columns:
        ag = pd.to_datetime(df_curr["Ag_Date"], errors="coerce")
        date_mask = ag >= cutoff

    repo_df = df_curr[bucket_mask & date_mask].copy()
    if repo_df.empty:
        return pd.DataFrame()

    for col in ["SOH", "LCC%", "Arrears / EMI"]:
        if col in repo_df.columns:
            repo_df[col] = to_num(repo_df, col)
    if "Ag_Date" in repo_df.columns:
        repo_df["Ag_Date"] = pd.to_datetime(repo_df["Ag_Date"], errors="coerce").dt.date

    cols = [c for c in _REPO_DISPLAY_COLS if c in repo_df.columns]
    return repo_df[cols].reset_index(drop=True)


# ── Good Customers ────────────────────────────────────────────────────────────

_GOOD_CUST_DISPLAY_COLS = [
    "Loan No", "Cust Name", "RegionName", "Unit",
    "Ag_Date", "Tenure", "VehEMI Accrued",
    "Arrears against Inst+Exp", "SOH", "Arrears / EMI",
    "LCC%", "Tenure Completed %",
]


def compute_good_customers(df_curr: pd.DataFrame) -> pd.DataFrame:
    """
    Loyal / high-quality customers eligible for refinance or relationship management.
    Criteria:
      - VehEMI Accrued / Tenure >= 70%  (completed at least 70% of loan term)
      - LCC% >= 100%                     (collected everything ever due, no lifetime shortfall)
    Sorted by SOH ascending (lowest exposure first - easiest refinance targets).
    """
    if df_curr.empty:
        return pd.DataFrame()

    df = df_curr.copy()

    # Tenure completed %
    if "VehEMI Accrued" in df.columns and "Tenure" in df.columns:
        emi_paid = to_num(df, "VehEMI Accrued")
        tenure   = to_num(df, "Tenure")
        df["Tenure Completed %"] = (emi_paid / tenure * 100).round(1)
        tenure_mask = (df["Tenure Completed %"] >= GOOD_CUSTOMER_MIN_TENURE_PCT) & tenure.notna() & (tenure > 0)
    else:
        df["Tenure Completed %"] = None
        tenure_mask = pd.Series(False, index=df.index)

    # LCC% hard cutoff
    lcc_mask = to_num(df, "LCC%") >= GOOD_CUSTOMER_MIN_LCC_PCT

    good_df = df[tenure_mask & lcc_mask].copy()
    if good_df.empty:
        return pd.DataFrame()

    # Numeric clean-up for display
    for col in ["SOH", "Arrears / EMI", "LCC%", "Arrears against Inst+Exp"]:
        if col in good_df.columns:
            good_df[col] = to_num(good_df, col)

    good_df = good_df.sort_values("SOH", ascending=True)

    cols = [c for c in _GOOD_CUST_DISPLAY_COLS if c in good_df.columns]
    return good_df[cols].reset_index(drop=True)
