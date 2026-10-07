"""
Portfolio Intelligence Analytics  -  pre-computed, pure pandas, no LLM.
Answers the 5 portfolio questions a collection lead needs every month.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go

from utils import (
    _EXEC_KEY, _exec_name, _with_exec_key, _safe_pct, BUCKET_ORDER, BUCKET_COLORS, to_num, account_count,
    is_yes, compute_overdue_demand_pct, attach_prev_delinquency, _unit_key, order_unit_columns, unit_metrics,
    first_value, as_of_or_today, segment_column, match_prev, loan_flags, portfolio_metrics,
)
from config import (
    MIN_ACCOUNTS_DIMENSION_BREAKDOWN, MIN_ACCOUNTS_EXECUTIVE, MIN_ACCOUNTS_PRODUCT_SEGMENT,
    MIN_ACCOUNTS_SOURCE_VINTAGE, MIN_ACCOUNTS_OVERDUE_DEMAND_EXECUTIVE, REGION_STATUS_DELTA_PP,
    GOOD_BAD_REGION_DELTA_PP, CONCERN_SCORE_BAD_THRESHOLD, CONCERN_SCORE_GOOD_THRESHOLD,
    CONCERN_SCORE_WEIGHTS, RISK_INDICATOR_STABLE_PP, RISK_INDICATOR_MATERIALITY_PP,
    RISK_INDICATOR_MATERIALITY_COUNT, RECENT_ADVANCES_COHORT_START, NOT_PAYING_3M_FLAG_COL,
    HARD_BUCKET_ARREARS_EMI_MIN,
)

VALID_BUCKETS = [b for b in BUCKET_ORDER if b != "NA"]


# ── Shared helpers ────────────────────────────────────────────────────────────

# The one "x as % of y" rule (0.0 when y is 0), shared with utils.
_safe_div = _safe_pct


def _none_if_nan(v):
    return None if v is None or pd.isna(v) else float(v)


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
            hovertemplate="<b>%{x}</b>: Last Month<br>Count: %{y:,}<extra></extra>",
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
        hovertemplate="<b>%{x}</b>: This Month<br>Count: %{y:,}<extra></extra>",
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

    title = "Bucket Distribution: Last Month vs This Month" if has_prev else "Bucket Distribution (This Month)"
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
        m = portfolio_metrics(df)   # the shared definitions, whole portfolio
        total = int(m["Accounts"])
        soh = m["SOH (Cr)"]
        pos_cr = round(to_num(df, "POS").sum() / 1e7, 2)
        arrears_cr = round(to_num(df, "Closing Arrears").sum() / 1e7, 2)
        npa_pct, npa_count = m["NPA%"], int(m["NPA"])
        sma2_count, sma2_pct = int(m["SMA-2"]), m["SMA-2%"]
        hard_pct, coll, strike_pct = m["Hard Bucket%"], m["Collection%"], m["Strike%"]
        overdue_demand = compute_overdue_demand_pct(df)

        # Insurance Debit Cases: same definition as smart_alerts.py's
        # "Insurance-Driven Delinquency" alert (EMI-current, but unpaid
        # insurance charge alone is creating delinquency) -- reuses the same
        # INSURANCE_EXP_ARREARS_MIN threshold so the KPI card and the alert
        # can never quietly disagree on what counts.
        arrears_emi = to_num(df, "Arrears / EMI")
        insurance_debit_count = int(m["Insurance-Only"])
        npa_soh_pct = m["NPA% (SOH)"]

        # NOV'25 Onward Delinquency: delinquent accounts (Arrears/EMI > 0)
        # originated on/after RECENT_ADVANCES_COHORT_START -- a FIXED cohort-start
        # date (not a rolling window like REPOSSESSION_WINDOW_MONTHS), since
        # this deliberately tracks originations since a specific management
        # change, not "recent" loans in general.
        if "Ag_Date" in df.columns:
            cohort_mask = (df["Ag_Date"] >= pd.Timestamp(RECENT_ADVANCES_COHORT_START)) & (arrears_emi > 0)
            nov25_delinquency_count = account_count(df[cohort_mask])
        else:
            nov25_delinquency_count = 0

        return {
            "accounts": total, "soh": soh, "pos_cr": pos_cr, "arrears_cr": arrears_cr,
            "npa_count": npa_count, "npa_pct": npa_pct, "npa_soh_pct": npa_soh_pct,
            "sma2_count": sma2_count, "sma2_pct": sma2_pct,
            "hard_pct": hard_pct, "coll_pct": coll, "strike_pct": strike_pct,
            "overdue_coll_pct": overdue_demand["overdue_pct"],
            "demand_coll_pct": overdue_demand["demand_pct"],
            "insurance_debit_count": insurance_debit_count,
            "nov25_delinquency_count": nov25_delinquency_count,
        }

    c = _calc(df_curr)
    p = _calc(df_prev)

    def _delta(key):
        # Always the RAW (cv - pv) movement, regardless of `inverse` -- this
        # used to sign-flip for inverse=True metrics under the assumption
        # that _kpi_card_html only reads a pre-flipped number, but
        # _kpi_card_html *also* independently applies its own inverse-aware
        # sign logic to whatever delta it receives (that's its actual,
        # documented contract: raw delta in, inverse flag flips the COLOR,
        # not the delta itself). Combining both was a double sign-flip: for
        # every inverse=True metric (SOH, SMA-2, NPA, Strike %), the
        # rendered arrow pointed the WRONG way (opposite of the real
        # movement) and the color was ALSO backwards (green shown for a
        # metric that had actually worsened, red for one that had actually
        # improved) -- confirmed against real data: NPA Accounts +431 (got
        # worse) rendered as "▼ green" (implying an improvement).
        cv = c.get(key, 0)
        pv = p.get(key, 0)
        if not p or pv == 0:
            return None
        return round(cv - pv, 2)

    # curr_raw/prev_raw: the literal unrounded numeric values behind "value"
    # and the raw "delta" -- exposed additively so a true comparison table
    # (This Month | Previous Month | Delta | % Change) can be built directly.
    def _card(label, key, value, unit, inverse, good_override=None):
        return {
            "label": label, "value": value, "delta": _delta(key),
            "unit": unit, "inverse": inverse, "good_override": good_override,
            "curr_raw": c.get(key, 0), "prev_raw": p.get(key, 0) if p else None,
        }

    # SOH rising isn't uniformly bad the way NPA%/SMA-2 rising is: SOH = POS +
    # Closing Arrears, so an increase driven by POS growth (book growing,
    # legitimate new/rolled-forward business) is a GOOD sign, while an
    # increase driven by Closing Arrears growth (customers falling further
    # behind) is BAD -- a flat sign-of-delta "inverse" flag can't tell those
    # apart, it only knows SOH went up, not why. A falling/flat SOH keeps the
    # existing "good" (inverse=True) treatment unconditionally -- less total
    # exposure is favorable regardless of which component drove it down.
    def _soh_good():
        if not p:
            return None
        soh_delta = c.get("soh", 0) - p.get("soh", 0)
        if soh_delta <= 0:
            return True
        pos_delta = c.get("pos_cr", 0) - p.get("pos_cr", 0)
        arrears_delta = c.get("arrears_cr", 0) - p.get("arrears_cr", 0)
        return bool(pos_delta >= arrears_delta)

    return [
        _card("Total Accounts", "accounts",  f"{c.get('accounts',0):,}",      "",   False),
        _card("Total SOH",      "soh",       f"₹{c.get('soh',0):.2f}Cr",      "Cr", True, good_override=_soh_good()),
        _card("SMA-2 Accounts", "sma2_count", f"{c.get('sma2_count',0):,}",   "",   True),
        _card("SMA-2 %",        "sma2_pct",  f"{c.get('sma2_pct',0):.2f}%",  "%",  True),
        _card("NPA Accounts",   "npa_count", f"{c.get('npa_count',0):,}",    "",   True),
        _card("NPA %",          "npa_pct",   f"{c.get('npa_pct',0):.2f}%",   "%",  True),
        _card("NPA % (SOH)",    "npa_soh_pct", f"{c.get('npa_soh_pct',0):.2f}%", "%", True),
        _card("Collection %",   "coll_pct",  f"{c.get('coll_pct',0):.2f}%",  "%",  False),
        # inverse=False: Strike% = % of accounts current on their installment
        # obligation (compute_strike_pct's own docstring) -- rising is GOOD
        # (more accounts current), matching the Dashboard tab's own Strike%
        # card (not in its _INVERSE_MOM set). This card previously had
        # inverse=True, backwards relative to both the metric's documented
        # meaning and the Dashboard tab's own convention for the same metric.
        _card("Strike %",       "strike_pct", f"{c.get('strike_pct',0):.2f}%", "%", False),
        _card("Overdue Collection %", "overdue_coll_pct", f"{c.get('overdue_coll_pct',0):.2f}%", "%", False),
        _card("Month Demand Collection %", "demand_coll_pct", f"{c.get('demand_coll_pct',0):.2f}%", "%", False),
        _card("Insurance Debit Cases", "insurance_debit_count", f"{c.get('insurance_debit_count',0):,}", "", True),
        _card("NOV'25 Onward Delinquency", "nov25_delinquency_count", f"{c.get('nov25_delinquency_count',0):,}", "", True),
    ]


# ── Section 2: Region Delinquency Scorecard ────────────────────────────────────

def compute_region_scorecard(df_curr: pd.DataFrame, df_prev: pd.DataFrame) -> pd.DataFrame:
    """One row per region: account/delinquency/SMA-2/NPA counts and rates,
    MoM deltas, Collection%, Strike%, SOH, roll rates, trend status. Every
    number comes from utils.unit_metrics (the shared definitions)."""
    if "RegionName" not in df_curr.columns or df_curr.empty:
        return pd.DataFrame()
    m = unit_metrics(df_curr, ["RegionName"])
    prev = (unit_metrics(df_prev, ["RegionName"]).set_index("RegionName")
            if len(df_prev) > 0 and "RegionName" in df_prev.columns else pd.DataFrame())

    rows = []
    for _, r in m.iterrows():
        region = r["RegionName"]
        has_prev_region = not prev.empty and region in prev.index
        delta = round(r["NPA%"] - prev.at[region, "NPA%"], 2) if has_prev_region else None
        sma2_delta = round(r["SMA-2%"] - prev.at[region, "SMA-2%"], 2) if has_prev_region else None
        status = "-"
        if delta is not None:
            status = "Worsening" if delta > REGION_STATUS_DELTA_PP else ("Improving" if delta < -REGION_STATUS_DELTA_PP else "Stable")
        rows.append({
            "Region": region,
            "Accounts": int(r["Accounts"]),
            "SMA-2": int(r["SMA-2"]),
            "SMA-2%": r["SMA-2%"],
            "NPA": int(r["NPA"]),
            "NPA%": r["NPA%"],
            "NPA% (SOH)": r["NPA% (SOH)"],
            "Delinquent": int(r["Delinquent"]),
            "Delinquency%": r["Delinquency%"],
            "Δ SMA-2%": sma2_delta,
            "Δ NPA%": delta,
            "Collection%": r["Collection%"],
            "Strike%": r["Strike%"],
            "SOH (Cr)": r["SOH (Cr)"],
            "Roll Fwd%": _none_if_nan(r.get("Roll Fwd%")),
            "Roll Bwd%": _none_if_nan(r.get("Roll Bwd%")),
            "Status": status,
        })

    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows).sort_values("NPA%", ascending=False).reset_index(drop=True)
    out = attach_prev_delinquency(out, df_prev, {"Region": "RegionName"})
    return out[order_unit_columns(out.columns, ["Region"])]


# ── Section 2b: Overdue vs Month Demand Collection (Region / Branch / Executive) ──

# Single source of truth for "a branch carries Region; an executive carries
# Branch+Region" -- compute_overdue_demand_scorecard's own output shape below.
# Its three consumers (the dashboard table in ui/tabs/portfolio_intelligence.py,
# and the monthly report (report_agent/story.py)) used to each hardcode their
# own copy of this exact mapping -- correct, but three independently-maintained
# copies (with inconsistent Title-case/lowercase casing between them) that a
# future 4th grain or a rename would need to update in lockstep with nothing
# enforcing that. Import this dict instead of redefining it.
OVERDUE_DEMAND_IDENTITY_COLS: dict[str, list[str]] = {
    "zone": [],
    "region": ["Zone"],
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
    Collection here uses the SAME shared Collection% (utils.unit_metrics) as the
    Pulse KPI and compute_region_scorecard, so "Overall Collection %" in this
    table is always numerically identical to "Collection %" everywhere else in
    the app, by construction, not by coincidence.

    Branch rows carry their Region alongside (a branch belongs to exactly one
    region); Executive rows carry both Branch and Region -- same identity-column
    pattern compute_npa_sma2_comparison already uses (grp[src].iloc[0], since
    every row in a Unit/MNT NAME group shares the same parent Region/Branch)."""
    df_curr = _with_exec_key(df_curr)
    flags = loan_flags(df_curr) if not df_curr.empty else None

    def _rows(col: str, label: str, extra_cols: list[tuple[str, str]] | None = None, min_accounts: int = 0) -> pd.DataFrame:
        if col not in df_curr.columns or df_curr.empty:
            return pd.DataFrame()
        extra_cols = extra_cols or []
        # Overall Collection % for every unit in one pass (shared definition).
        coll = unit_metrics(df_curr, [col], flags=flags).set_index(col)["Collection%"]
        rows = []
        # Only the columns read below: grouping all ~100 columns per unit was the cost.
        keep = [col, *[src for src, _ in extra_cols], "Loan No", "Overdue", "MonthDemandExclPC", "OverdueCollected",
                "DemandCollected", "Month Collection (Excluding Reserve Collection)"]
        narrow = df_curr[list(dict.fromkeys(k for k in keep if k in df_curr.columns))]
        for key, grp in narrow.groupby(col):
            name = _exec_name(key) if col == _EXEC_KEY else key
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
                "Overall Collection %": float(coll.get(key, 0.0)),
            })
            rows.append(row)
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows).sort_values("Overdue Collection %").reset_index(drop=True)

    return {
        "zone":      _rows("Zone", "Zone"),
        "region":    _rows("RegionName", "Region", extra_cols=[("Zone", "Zone")]),
        "branch":    _rows("Unit", "Branch", extra_cols=[("RegionName", "Region")]),
        "executive": _rows(_EXEC_KEY, "Executive", extra_cols=[("Unit", "Branch"), ("RegionName", "Region")],
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
        title=dict(text=f"Overdue vs Month Demand Collection %: {title_suffix}", font=dict(size=13, color="#111"), x=0),
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
    Per-dimension comparison: accounts, delinquency (with last month's %),
    NPA + SMA-2 counts curr vs prev with Δ and Δ%, and roll rates.
    Returns dict with keys 'region', 'branch', 'executive' - each a DataFrame.

    All numbers come from utils.unit_metrics (one grouped pass per month).
    Labels are matched to last month case/space-insensitively; an executive
    is (name, branch), never name alone, so two same-named people in
    different branches stay separate, and a name the source truncates to a
    different length next month still matches (utils.match_prev).
    """
    has_prev = len(df_prev) > 0
    if df_curr.empty:
        return {}
    # Per-loan checks once per month, reused by all three groupings.
    curr_flags = loan_flags(df_curr)
    prev_flags = loan_flags(df_prev) if has_prev else None

    def _norm(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        return df.assign(**{f"_k_{c}": df[c].map(_unit_key) for c in cols if c in df.columns})

    def _delta(c, p):
        if p is None:
            return None, None
        d = c - p
        pct = round(d / p * 100, 1) if p > 0 else (100.0 if d > 0 else 0.0)
        return d, pct

    def _table(label_cols: list[str], min_accounts: int, fuzzy_pos, display, extras: dict) -> pd.DataFrame:
        """label_cols: raw columns to group by; display(row) -> the identity
        columns; extras: {output column: source column} from the unit's first row."""
        if not set(label_cols) <= set(df_curr.columns):
            return pd.DataFrame()
        key_cols = [f"_k_{c}" for c in label_cols]
        # An executive groups on the NORMALIZED branch (the same branch typed
        # two ways is one branch) but keeps the raw name for display.
        group_cols = [label_cols[0]] + key_cols[1:]
        curr = _norm(df_curr, label_cols)
        m = unit_metrics(curr, group_cols, min_accounts=min_accounts, flags=curr_flags)
        if m.empty:
            return pd.DataFrame()
        prev: dict = {}
        if has_prev and set(label_cols) <= set(df_prev.columns):
            pm = unit_metrics(_norm(df_prev, label_cols), key_cols, flags=prev_flags)
            prev = {tuple(r[k] for k in key_cols): r for _, r in pm.iterrows()}
        firsts = {out: first_value(curr, group_cols, src) for out, src in extras.items()}

        rows = []
        for _, r in m.iterrows():
            gkey = tuple(r[c] for c in group_cols)
            p = match_prev(prev, (_unit_key(gkey[0]),) + gkey[1:], fuzzy_pos) if prev else None
            npa_p = int(p["NPA"]) if p is not None else None
            sma2_p = int(p["SMA-2"]) if p is not None else None
            npa_d, npa_dpct = _delta(int(r["NPA"]), npa_p)
            sma2_d, sma2_dpct = _delta(int(r["SMA-2"]), sma2_p)
            row = display(r)
            for out, ser in firsts.items():
                v = ser.get(gkey if len(gkey) > 1 else gkey[0])
                row[out] = None if v is None or pd.isna(v) else str(v)
            row.update({
                "Accounts": int(r["Accounts"]),
                "Delinquent": int(r["Delinquent"]),
                "Delinquency%": r["Delinquency%"],
            })
            if prev:
                prev_pct = float(p["Delinquency%"]) if p is not None else None
                row["Prev Delinquency%"] = prev_pct
                row["Δ Delinquency%"] = round(r["Delinquency%"] - prev_pct, 2) if prev_pct is not None else None
            row.update({
                "SMA-2 (Curr)": int(r["SMA-2"]), "SMA-2 (Prev)": sma2_p,
                "NPA (Curr)": int(r["NPA"]), "NPA (Prev)": npa_p,
                "SMA-2 Δ": sma2_d, "SMA-2 Δ%": sma2_dpct, "NPA Δ": npa_d, "NPA Δ%": npa_dpct,
                "Roll Fwd%": _none_if_nan(r.get("Roll Fwd%")), "Roll Bwd%": _none_if_nan(r.get("Roll Bwd%")),
            })
            rows.append(row)
        return pd.DataFrame(rows).sort_values("NPA (Curr)", ascending=False, kind="stable").reset_index(drop=True)

    result = {}
    region = _table(["RegionName"], MIN_ACCOUNTS_DIMENSION_BREAKDOWN, None,
                    lambda r: {"RegionName": str(r["RegionName"])}, {})
    if not region.empty:
        result["region"] = region
    branch = _table(["Unit"], MIN_ACCOUNTS_DIMENSION_BREAKDOWN, None,
                    lambda r: {"Unit": str(r["Unit"])}, {"Region": "RegionName"})
    if not branch.empty:
        result["branch"] = branch
    if "MNT NAME" in df_curr.columns and "curr_bucket" in df_curr.columns:
        if "Unit" in df_curr.columns:
            exec_ = _table(["MNT NAME", "Unit"], MIN_ACCOUNTS_EXECUTIVE, 0,
                           lambda r: {"MNT NAME": f"{r['MNT NAME']} ({r['_k_Unit']})", "Unit": r["_k_Unit"]},
                           {"Region": "RegionName"})
        else:
            exec_ = _table(["MNT NAME"], MIN_ACCOUNTS_EXECUTIVE, 0,
                           lambda r: {"MNT NAME": str(r["MNT NAME"]), "Unit": None}, {"Region": "RegionName"})
        if not exec_.empty:
            result["executive"] = exec_
    return result


# ── Section 2: Branch Quadrant ────────────────────────────────────────────────

def _branch_aggregates(df_curr: pd.DataFrame) -> pd.DataFrame:
    """One row per branch (Unit), from utils.unit_metrics: shared by
    compute_branch_quadrant and compute_npa_sma2_comparison's branch pass.
    Roll Fwd%/Roll Bwd% stay NaN for a branch with no previous-month bucket
    (not 0: "no history" isn't "nothing got worse"). Branches below
    MIN_ACCOUNTS_DIMENSION_BREAKDOWN are excluded."""
    if "Unit" not in df_curr.columns or df_curr.empty:
        return pd.DataFrame()
    m = unit_metrics(df_curr, ["Unit"], min_accounts=MIN_ACCOUNTS_DIMENSION_BREAKDOWN)
    if m.empty:
        return pd.DataFrame()
    region = first_value(df_curr, ["Unit"], "RegionName")
    m.insert(1, "Region", m["Unit"].map(region).astype(str) if not region.empty else None)
    if "Roll Fwd%" not in m.columns:
        m["Roll Fwd%"], m["Roll Bwd%"] = np.nan, np.nan
    return m.rename(columns={"Unit": "Branch"}).assign(Branch=lambda d: d["Branch"].astype(str))


def _concern_score(df: pd.DataFrame) -> pd.Series:
    """0-100 percentile rank blend (config.CONCERN_SCORE_WEIGHTS, higher =
    worse). A component a branch has no data for (e.g. Roll Fwd% with no
    previous month) is left out and the remaining weights re-scaled, instead
    of scoring it as the best possible value."""
    parts, weights = [], []
    for col, w in CONCERN_SCORE_WEIGHTS.items():
        if col not in df.columns:
            continue
        rank = df[col].rank(ascending=True, pct=True)
        parts.append(rank * w)
        weights.append(rank.notna() * w)
    if not parts:
        return pd.Series(0, index=df.index)
    score = pd.concat(parts, axis=1).sum(axis=1) / pd.concat(weights, axis=1).sum(axis=1).replace(0, np.nan)
    return score.mul(100).round(0).fillna(0).astype(int)


def compute_branch_quadrant(df_curr: pd.DataFrame, df_prev: pd.DataFrame | None = None) -> tuple[pd.DataFrame, go.Figure]:
    """Branch scatter (Collection% vs NPA%, bubble=SOH) + concern score table.
    With df_prev, the table also carries Prev Delinquency% and Δ Delinquency%."""
    agg = _branch_aggregates(df_curr)
    if agg.empty:
        return pd.DataFrame(), go.Figure()

    df = agg[["Branch", "Region", "Accounts", "Collection%", "SMA-2%", "NPA%", "NPA% (SOH)", "Delinquent",
              "Delinquency%", "Strike%", "Hard Bucket%", "SOH (Cr)", "Roll Fwd%",
              "Not Paying 3M+", "Not Paying 3M+%"]].copy()
    df["Concern Score"] = _concern_score(df)
    df = df.sort_values("Concern Score", ascending=False).reset_index(drop=True)
    df.insert(0, "Rank", range(1, len(df) + 1))
    df = attach_prev_delinquency(df, df_prev, {"Branch": "Unit"})
    df = df[order_unit_columns(df.columns, ["Rank", "Branch", "Region"])]
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
        title=dict(text="Branch Quadrant: Collection% vs NPA%  (bubble size = SOH)", font=dict(size=13, color="#111"), x=0),
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
    """Executives ranked by net accounts rescued from high-risk buckets
    (Rescued = moved from SMA-1/SMA-2/NPA to a better bucket; Slipped = any
    worse bucket). From utils.unit_metrics; executives are (name, branch)."""
    if (
        "MNT NAME" not in df_curr.columns
        or "prev_bucket" not in df_curr.columns
        or "curr_bucket" not in df_curr.columns
    ):
        return pd.DataFrame()

    group_cols = ["MNT NAME", "Unit"] if "Unit" in df_curr.columns else ["MNT NAME"]
    m = unit_metrics(df_curr, group_cols, min_accounts=MIN_ACCOUNTS_EXECUTIVE)
    if m.empty or "_roll_valid" not in m.columns:
        return pd.DataFrame()
    m = m[m["_roll_valid"] > 0]
    if m.empty:
        return pd.DataFrame()
    region = first_value(df_curr, group_cols, "RegionName")
    branch = m["Unit"].astype(str) if "Unit" in m.columns else pd.Series("", index=m.index)
    keys = list(zip(*[m[c] for c in group_cols])) if len(group_cols) > 1 else list(m[group_cols[0]])
    out = pd.DataFrame({
        "Executive": [f"{n} ({b})" if b else str(n) for n, b in zip(m["MNT NAME"], branch)],
        "Branch": branch.values,
        "Region": [str(region.get(k, "")) if not region.empty else "" for k in keys],
        "Accounts": m["Accounts"].values,
        "Collection%": m["Collection%"].values,
        "Strike%": m["Strike%"].values,
        "Rescued": m["Rescued"].values,
        "Slipped": m["Slipped"].values,
    })
    out["Net Recovery"] = out["Rescued"] - out["Slipped"]
    return out.sort_values("Net Recovery", ascending=False, kind="stable").reset_index(drop=True)


# ── Section 3: Good vs Bad ────────────────────────────────────────────────────

def good_bad_summary(df_curr: pd.DataFrame, df_prev: pd.DataFrame, rr_meta: dict | None = None) -> dict:
    """compute_good_bad straight from the data: builds the region, branch,
    executive and risk-indicator tables it reads (the same functions and
    inputs the dashboard used), so AI Query can answer it on demand."""
    has_prev = df_prev is not None and len(df_prev) > 0
    rr = rr_meta if rr_meta and rr_meta.get("matched_count", 0) > 0 else None
    return compute_good_bad(
        compute_region_scorecard(df_curr, df_prev), compute_branch_quadrant(df_curr, df_prev)[0],
        compute_risk_indicators(df_curr, df_prev, rr), compute_executive_recovery(df_curr), has_prev,
    )


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
            good.append(f"{r['Region']}: NPA% fell {abs(r['Δ NPA%']):.1f}pp, delinquency improving")
        for _, r in wor.head(2).iterrows():
            bad.append(f"{r['Region']}: NPA% rose {r['Δ NPA%']:.1f}pp, escalate field visits")

    if not branch_df.empty and "Concern Score" in branch_df.columns:
        worst = branch_df.iloc[0]
        best  = branch_df.iloc[-1]
        if worst["Concern Score"] >= CONCERN_SCORE_BAD_THRESHOLD:
            bad.append(f"{worst['Branch']}: Highest concern ({worst['Concern Score']}), SMA-2 {worst.get('SMA-2%',0):.1f}%, NPA {worst['NPA%']:.1f}%, Coll {worst['Collection%']:.1f}%")
        if best["Concern Score"] <= CONCERN_SCORE_GOOD_THRESHOLD:
            good.append(f"{best['Branch']}: Healthiest branch, SMA-2 {best.get('SMA-2%',0):.1f}%, NPA {best['NPA%']:.1f}%, Coll {best['Collection%']:.1f}%")

    if not exec_df.empty and has_prev:
        top_exec  = exec_df[exec_df["Net Recovery"] > 0].head(1)
        bad_exec  = exec_df[exec_df["Net Recovery"] < 0].tail(1)
        if len(top_exec) > 0:
            r = top_exec.iloc[0]
            good.append(f"{r['Executive']}: rescued {r['Rescued']} accounts from NPA/SMA")
        if len(bad_exec) > 0:
            r = bad_exec.iloc[0]
            bad.append(f"{r['Executive']}: net {abs(r['Net Recovery'])} accounts slipped vs rescued, portfolio deteriorating")

    for ind in risk_indicators:
        d = ind["_direction"]
        delta_abs = abs(ind.get("_delta", 0))
        threshold = RISK_INDICATOR_MATERIALITY_COUNT if ind.get("_is_count") else RISK_INDICATOR_MATERIALITY_PP
        if d == "Improving" and delta_abs >= threshold:
            # Strip leading sign  -  the word "improvement" already implies positive change
            delta_display = str(ind["Δ"]).lstrip("+-").lstrip()
            good.append(f"{ind['Signal']}: {delta_display} improvement")
        elif d == "Worsening" and delta_abs >= threshold:
            bad.append(f"{ind['Signal']}: {ind['Δ']} ({ind['Note']})")

    return {"good": good[:6], "bad": bad[:6]}


# ── Section 4: Risk Flag Comparison ──────────────────────────────────────────

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
    if df_curr.empty:
        return results
    flags = loan_flags(df_curr)   # once, reused by every table below

    seg_col = segment_column(df_curr)
    if seg_col:
        rows = _group_npa_table(df_curr, seg_col, "Segment", min_n=MIN_ACCOUNTS_PRODUCT_SEGMENT, flags=flags)
        if rows:
            results["segment"] = pd.DataFrame(rows).sort_values("NPA%", ascending=False).reset_index(drop=True)

    if "FUEL_TYPE" in df_curr.columns:
        rows = _group_npa_table(df_curr, "FUEL_TYPE", "Fuel Type", min_n=MIN_ACCOUNTS_PRODUCT_SEGMENT, flags=flags)
        if rows:
            results["fuel"] = pd.DataFrame(rows).sort_values("NPA%", ascending=False).reset_index(drop=True)

    if "Ag_Date" in df_curr.columns:
        cohort = pd.to_datetime(df_curr["Ag_Date"], errors="coerce").dt.to_period("M")
        # Post-dated / future agreement dates are left out.
        in_range = cohort.notna() & (cohort <= as_of_or_today(as_of).to_period("M"))
        df_v = df_curr[in_range].assign(_cohort=cohort[in_range].astype(str))
        m = unit_metrics(df_v, ["_cohort"], min_accounts=MIN_ACCOUNTS_SOURCE_VINTAGE, flags=flags)
        avg_loan = to_num(df_v, "Loan Amount").groupby(df_v["_cohort"]).mean()
        cohort_rows = [{
            "Disbursement Month": r["_cohort"],
            "Accounts":   int(r["Accounts"]),
            "NPA Count":  int(r["NPA"]),
            "SMA-2 Count": int(r["SMA-2"]),
            "NPA%":        r["NPA%"],
            "SMA-2%":      r["SMA-2%"],
            "Collection%": r["Collection%"],
            "SOH (Cr)":    r["SOH (Cr)"],
            "Avg Loan (L)": 0.0 if pd.isna(avg_loan.get(r["_cohort"])) else round(avg_loan[r["_cohort"]] / 1e5, 2),
        } for _, r in m.iterrows()] if not m.empty else []
        if cohort_rows:
            results["vintage"] = (
                pd.DataFrame(cohort_rows)
                .sort_values("Disbursement Month", ascending=False)
                .reset_index(drop=True)
            )

    if "SRC Name" in df_curr.columns:
        rows = _group_npa_table(df_curr, "SRC Name", "Source", min_n=MIN_ACCOUNTS_SOURCE_VINTAGE, flags=flags)
        if rows:
            results["source"] = pd.DataFrame(rows).sort_values("NPA%", ascending=False).reset_index(drop=True)

    return results


# ── New Advances (Business/Originations) ──────────────────────────────────────


NEW_ADVANCES_IDENTITY_COLS: dict[str, list[str]] = {
    "region": [],
    "branch": ["Region"],
    "executive": ["Branch", "Region"],
}


def _group_npa_table(df: pd.DataFrame, group_col: str, label: str, min_n: int,
                     flags: pd.DataFrame | None = None) -> list:
    """Segment/fuel/source table rows from utils.unit_metrics (one grouped pass)."""
    if group_col not in df.columns:
        return []
    m = unit_metrics(df, [group_col], min_accounts=min_n, flags=flags)
    rows = []
    for _, r in m.iterrows():
        val_str = str(r[group_col]).strip()
        if not val_str or val_str.lower() in ("nan", "none", ""):
            continue
        rows.append({
            label: val_str,
            "Accounts": int(r["Accounts"]),
            "SMA-2%": r["SMA-2%"],
            "NPA%": r["NPA%"],
            "NPA% (SOH)": r["NPA% (SOH)"],
            "Collection%": r["Collection%"],
            "SOH (Cr)": r["SOH (Cr)"],
        })
    return rows


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
             "%", "down", "Handle SMA-2 now to prevent NPA: 2+ EMI overdue, last intervention window")
        _add("NPA Pool", _pct(df_curr, "NPA"),
             _pct(df_prev, "NPA") if has_prev else 0.0,
             "%", "down", "Current NPA accounts as % of total portfolio")

    if rr_meta and rr_meta.get("matched_count", 0) > 0:
        # No real prior-period formation rate exists to compare against (it would
        # require a THIRD month's data) -- pass prev_val=None so this renders as a
        # standalone reading, not a fabricated "Worsening" trend every month.
        _add("Fresh NPA Formation", rr_meta["npa_formation_rate"], None, "%", "down",
             "Non-NPA accounts that became NPA this month: more important than total NPA count")

    col3m = NOT_PAYING_3M_FLAG_COL
    if col3m in df_curr.columns:
        c = int(is_yes(df_curr, col3m).sum())
        p = int(is_yes(df_prev, col3m).sum()) if has_prev and col3m in df_prev.columns else 0
        _add("Not Paying 3M+", c, p, "", "down",
             f"No payment in 3 months AND more than {HARD_BUCKET_ARREARS_EMI_MIN} EMIs overdue", is_count=True)

    if "Non Starter" in df_curr.columns:
        c = int(is_yes(df_curr, "Non Starter").sum())
        p = int(is_yes(df_prev, "Non Starter").sum()) if has_prev and "Non Starter" in df_prev.columns else 0
        _add("Non-Starters", c, p, "", "down", "Never paid first EMI: highest NPA risk", is_count=True)

    if "CoLending_Loans" in df_curr.columns and "Arrears / EMI" in df_curr.columns:
        c = int((is_yes(df_curr, "CoLending_Loans") & (to_num(df_curr, "Arrears / EMI") > 0)).sum())
        p_val = 0
        if has_prev and "CoLending_Loans" in df_prev.columns and "Arrears / EMI" in df_prev.columns:
            p_val = int((is_yes(df_prev, "CoLending_Loans") & (to_num(df_prev, "Arrears / EMI") > 0)).sum())
        _add("Co-Lending At Risk", c, p_val, "", "down", "Partner-bank loans showing delinquency", is_count=True)

    return indicators


