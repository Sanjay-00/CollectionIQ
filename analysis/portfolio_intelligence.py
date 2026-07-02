"""
Portfolio Intelligence Analytics  -  pre-computed, pure pandas, no LLM.
Answers the 5 portfolio questions a collection lead needs every month.
"""
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from utils import BUCKET_ORDER, BUCKET_SCORE, BUCKET_COLORS, to_num, account_count, is_yes
from config import (
    MIN_ACCOUNTS_DIMENSION_BREAKDOWN,
    MIN_ACCOUNTS_PRODUCT_SEGMENT,
    MIN_ACCOUNTS_SOURCE_VINTAGE,
    HARD_BUCKET_ARREARS_EMI_MIN,
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
        for b in buckets:
            delta   = curr_counts[b] - prev_counts[b]
            if delta == 0:
                continue
            arrow   = "▲" if delta > 0 else "▼"
            is_bad  = (delta > 0 and b != "STD") or (delta < 0 and b == "STD")
            color   = "#dc2626" if is_bad else "#16a34a"
            fig.add_annotation(
                x=b,
                y=max(curr_counts[b], prev_counts[b]) * 1.30,
                text=f"<b>{arrow} {abs(delta):,}</b>",
                showarrow=False,
                font=dict(size=11, color=color),
                xanchor="center",
                bgcolor="rgba(255,255,255,0.85)",
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
        hard_pct = _safe_div((to_num(df, "Arrears / EMI") >= HARD_BUCKET_ARREARS_EMI_MIN).sum(), total)
        coll = _coll_pct(df)
        strike_valid = df[df["Strike"].astype(str).str.strip().str.upper().isin(["Y", "N"])] if "Strike" in df.columns else pd.DataFrame()
        strike_pct = _safe_div(is_yes(strike_valid, "Strike").sum(), len(strike_valid)) if not strike_valid.empty else 0.0
        return {
            "accounts": total, "soh": soh,
            "npa_count": npa_count, "npa_pct": npa_pct,
            "sma2_count": sma2_count, "sma2_pct": sma2_pct,
            "hard_pct": hard_pct, "coll_pct": coll, "strike_pct": strike_pct,
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

    return [
        {"label": "Total Accounts",       "value": f"{c.get('accounts',0):,}",       "delta": _delta("accounts"),            "unit": "",  "inverse": False},
        {"label": "Total SOH",            "value": f"₹{c.get('soh',0):.2f}Cr",       "delta": _delta("soh", inverse=True),   "unit": "Cr","inverse": True},
        {"label": "SMA-2 Accounts",       "value": f"{c.get('sma2_count',0):,}",      "delta": _delta("sma2_count", inverse=True), "unit": "", "inverse": True},
        {"label": "SMA-2 %",              "value": f"{c.get('sma2_pct',0):.2f}%",    "delta": _delta("sma2_pct", inverse=True),  "unit": "%","inverse": True},
        {"label": "NPA Accounts",         "value": f"{c.get('npa_count',0):,}",       "delta": _delta("npa_count", inverse=True), "unit": "", "inverse": True},
        {"label": "NPA %",                "value": f"{c.get('npa_pct',0):.2f}%",     "delta": _delta("npa_pct", inverse=True),   "unit": "%","inverse": True},
        {"label": "Collection %",         "value": f"{c.get('coll_pct',0):.2f}%",    "delta": _delta("coll_pct"),                "unit": "%","inverse": False},
        {"label": "Strike %",             "value": f"{c.get('strike_pct',0):.2f}%",  "delta": _delta("strike_pct", inverse=True), "unit": "%","inverse": True},
    ]


# ── Section 2: Region Delinquency Scorecard ────────────────────────────────────

def compute_region_scorecard(df_curr: pd.DataFrame, df_prev: pd.DataFrame) -> pd.DataFrame:
    """One row per region: curr vs prev NPA%, delta, Collection%, Hard Bucket%, SOH, roll rates, trend status."""
    if "RegionName" not in df_curr.columns or df_curr.empty:
        return pd.DataFrame()

    rows = []
    for region, grp in df_curr.groupby("RegionName"):
        n = account_count(grp)
        curr_npa = _npa_pct(grp)
        curr_coll = _coll_pct(grp)
        soh = _soh_cr(grp)
        hard_pct = _safe_div((to_num(grp, "Arrears / EMI") >= HARD_BUCKET_ARREARS_EMI_MIN).sum(), n)
        roll_fwd, roll_bwd = _roll_rates(grp)

        sma2_count = int((grp["curr_bucket"] == "SMA-2").sum()) if "curr_bucket" in grp.columns else 0
        sma2_pct   = _safe_div(sma2_count, n)

        prev_npa = 0.0
        has_prev_region = False
        if len(df_prev) > 0 and "RegionName" in df_prev.columns:
            prev_grp = df_prev[df_prev["RegionName"] == region]
            if len(prev_grp) > 0:
                prev_npa = _npa_pct(prev_grp)
                has_prev_region = True

        delta = round(curr_npa - prev_npa, 2) if has_prev_region else None
        status = "-"
        if delta is not None:
            status = "Worsening" if delta > REGION_STATUS_DELTA_PP else ("Improving" if delta < -REGION_STATUS_DELTA_PP else "Stable")

        rows.append({
            "Region": region,
            "Accounts": n,
            "SMA-2": sma2_count,
            "SMA-2%": sma2_pct,
            "NPA% (Curr)": curr_npa,
            "NPA% (Prev)": prev_npa if has_prev_region else None,
            "Δ NPA%": delta,
            "Collection%": curr_coll,
            "Hard Bucket%": hard_pct,
            "SOH (Cr)": soh,
            "Roll Fwd%": roll_fwd,
            "Roll Bwd%": roll_bwd,
            "Status": status,
        })

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("NPA% (Curr)", ascending=False).reset_index(drop=True)


# ── Section 2: NPA & SMA-2 Comparison (Region / Branch / Executive) ──────────

def compute_npa_sma2_comparison(df_curr: pd.DataFrame, df_prev: pd.DataFrame) -> dict:
    """
    Per-dimension comparison: NPA count + SMA-2 count, curr vs prev, with Δ and Δ%.
    Returns dict with keys 'region', 'branch', 'executive'  -  each a DataFrame.
    """
    has_prev = len(df_prev) > 0

    def _dim_rows(df_c, df_p, col):
        if col not in df_c.columns:
            return []
        rows = []
        prev_map: dict = {}
        if has_prev and col in df_p.columns and "curr_bucket" in df_p.columns:
            for grp_key, grp in df_p.groupby(col):
                prev_map[str(grp_key)] = {
                    "npa": int((grp["curr_bucket"] == "NPA").sum()),
                    "sma2": int((grp["curr_bucket"] == "SMA-2").sum()),
                }

        for grp_key, grp in df_c.groupby(col):
            name = str(grp_key)
            n = account_count(grp)
            if n < MIN_ACCOUNTS_DIMENSION_BREAKDOWN:
                continue
            npa_c  = int((grp["curr_bucket"] == "NPA").sum())  if "curr_bucket" in grp.columns else 0
            sma2_c = int((grp["curr_bucket"] == "SMA-2").sum()) if "curr_bucket" in grp.columns else 0
            prev   = prev_map.get(name, {})
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
            rows.append({
                col:                name,
                "Accounts":         n,
                "NPA (Curr)":       npa_c,
                "NPA (Prev)":       npa_p,
                "NPA Δ":            npa_d,
                "NPA Δ%":          npa_dpct,
                "SMA-2 (Curr)":     sma2_c,
                "SMA-2 (Prev)":     sma2_p,
                "SMA-2 Δ":         sma2_d,
                "SMA-2 Δ%":        sma2_dpct,
            })
        return rows

    result = {}

    rows = _dim_rows(df_curr, df_prev, "RegionName")
    if rows:
        result["region"] = pd.DataFrame(rows).sort_values("NPA (Curr)", ascending=False).reset_index(drop=True)

    rows = _dim_rows(df_curr, df_prev, "Unit")
    if rows:
        result["branch"] = pd.DataFrame(rows).sort_values("NPA (Curr)", ascending=False).reset_index(drop=True)

    if "MNT NAME" in df_curr.columns and "curr_bucket" in df_curr.columns:
        rows = _dim_rows(df_curr, df_prev, "MNT NAME")
        if rows:
            result["executive"] = pd.DataFrame(rows).sort_values("NPA (Curr)", ascending=False).reset_index(drop=True)

    return result


# ── Section 2: Branch Quadrant ────────────────────────────────────────────────

def compute_branch_quadrant(df_curr: pd.DataFrame) -> tuple[pd.DataFrame, go.Figure]:
    """Branch scatter (Collection% vs NPA%, bubble=SOH) + concern score table."""
    if "Unit" not in df_curr.columns or df_curr.empty:
        return pd.DataFrame(), go.Figure()

    rows = []
    for branch, grp in df_curr.groupby("Unit"):
        n = account_count(grp)
        if n < MIN_ACCOUNTS_DIMENSION_BREAKDOWN:
            continue
        roll_fwd, _ = _roll_rates(grp)
        chronic = int(is_yes(grp, "No Coll 3 Months and >6 EMI").sum())
        sma2_n = int((grp["curr_bucket"] == "SMA-2").sum()) if "curr_bucket" in grp.columns else 0
        rows.append({
            "Branch": str(branch),
            "Accounts": n,
            "Collection%": _coll_pct(grp),
            "SMA-2%": _safe_div(sma2_n, n),
            "NPA%": _npa_pct(grp),
            "Hard Bucket%": _safe_div((to_num(grp, "Arrears / EMI") >= HARD_BUCKET_ARREARS_EMI_MIN).sum(), n),
            "SOH (Cr)": _soh_cr(grp),
            "Roll Fwd%": roll_fwd if roll_fwd is not None else 0.0,
            "Chronic (3M+)": chronic,
        })

    if not rows:
        return pd.DataFrame(), go.Figure()

    df = pd.DataFrame(rows)
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
        rows.append({
            "Executive": f"{exec_name} ({branch})" if branch else exec_name,
            "Branch": branch,
            "Accounts": n,
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

def compute_product_analysis(df_curr: pd.DataFrame) -> dict:
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
        _today_period = pd.Period.now("M")
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


def _group_npa_table(df: pd.DataFrame, group_col: str, label: str, min_n: int) -> list:
    rows = []
    for val, grp in df.groupby(group_col):
        val_str = str(val).strip()
        if not val_str or val_str.lower() in ("nan", "none", ""):
            continue
        n = account_count(grp)
        if n < min_n:
            continue
        sma2_n = int((grp["curr_bucket"] == "SMA-2").sum()) if "curr_bucket" in grp.columns else 0
        rows.append({
            label: val_str,
            "Accounts": n,
            "SMA-2%": _safe_div(sma2_n, n),
            "NPA%": _npa_pct(grp),
            "Collection%": _coll_pct(grp),
            "SOH (Cr)": _soh_cr(grp),
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

    npa_colors  = _marker_colors(npa_vals,  (10, 5))
    sma2_colors = _marker_colors(sma2_vals, (10, 5))

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
    fig.add_hrect(y0=10, y1=max(max(npa_vals + sma2_vals) * 1.1, 12),
                  fillcolor="rgba(153,27,27,0.04)", line_width=0, layer="below")
    fig.add_hrect(y0=5, y1=10,
                  fillcolor="rgba(217,119,6,0.04)", line_width=0, layer="below")
    fig.add_hline(y=10, line_dash="dash", line_color="#991b1b", line_width=1,
                  annotation_text="Critical  10%", annotation_position="right",
                  annotation_font=dict(size=10, color="#991b1b"))
    fig.add_hline(y=5, line_dash="dash", line_color="#d97706", line_width=1,
                  annotation_text="Watch  5%", annotation_position="right",
                  annotation_font=dict(size=10, color="#d97706"))

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
        delta = round(curr_val - prev_val, 2) if has_prev else 0.0
        if not has_prev:
            direction = " - "
        elif is_count:
            direction = ("Improving" if delta < 0 else ("Worsening" if delta > 0 else "Stable")) if good_dir == "down" else ("Improving" if delta > 0 else ("Worsening" if delta < 0 else "Stable"))
        else:
            direction = "Stable" if abs(delta) < RISK_INDICATOR_STABLE_PP else (("Improving" if delta < 0 else "Worsening") if good_dir == "down" else ("Improving" if delta > 0 else "Worsening"))
        fmt      = f"{int(curr_val):,}{unit}" if is_count else f"{curr_val:.1f}{unit}"
        prev_fmt = (f"{int(prev_val):,}{unit}" if is_count else f"{prev_val:.1f}{unit}") if has_prev else " - "
        sign     = "+" if delta >= 0 else ""
        delta_s  = (f"{sign}{int(delta)}{unit}" if is_count else f"{sign}{delta:.1f}{unit}") if has_prev else " - "
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
        _add("Fresh NPA Formation", rr_meta["npa_formation_rate"], 0.0, "%", "down",
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

    # Top fleet customers by SOH
    top_rows = []
    for mob, grp in fleet_df.groupby("Cust Mob No"):
        cust_name = grp["Cust Name"].iloc[0] if "Cust Name" in grp.columns else str(mob)
        n_loans   = grp["Loan No"].nunique()
        soh       = _soh_cr(grp)
        npa_count = int((grp["curr_bucket"] == "NPA").sum()) if "curr_bucket" in grp.columns else 0
        top_rows.append({
            "Customer": str(cust_name),
            "Mobile": str(mob),
            "Loans": n_loans,
            "NPA Loans": npa_count,
            "Total SOH (Cr)": soh,
        })

    top_df = (
        pd.DataFrame(top_rows)
        .sort_values("Total SOH (Cr)", ascending=False)
        .head(20)
        .reset_index(drop=True)
    )

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


def compute_repossession_list(df_curr: pd.DataFrame) -> pd.DataFrame:
    """
    Accounts eligible for repossession:
      - curr_bucket in ["SMA-2", "NPA"]  (2+ EMI overdue, deep delinquent)
      - Ag_Date within last 18 months    (recent loans  -  still have collateral value)

    Returns cleaned DataFrame. Caller sorts for the 3 views.
    """
    if df_curr.empty:
        return pd.DataFrame()

    cutoff = pd.Timestamp.today() - pd.DateOffset(months=REPOSSESSION_WINDOW_MONTHS)

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
