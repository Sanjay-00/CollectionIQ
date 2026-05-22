import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

YELLOW = "#FFC000"

# Columns that MUST exist for calculations to work
CRITICAL_COLS = [
    "Loan No", "RegionName", "Unit", "Loan Status", "Ag_Date",
    "Arrears / EMI", "Month Receipt Amount", "NET Collection Demand Inst+Exp",
    "POS", "LCC%", "Strike", "Closing Arrears", "ClosingPC",
    "Month Due-Inst", "Month Due-Exp", "Total Cum Collection",
]

# Known column name variations across different LCC extracts
COL_ALIASES = {
    "UN-CLEARED CHEQUE FOR THE MONTH/Amount Not remitted by R":
        "UN-CLEARED CHEQUE FOR THE MONTH/Amount Not remitted by RE",
    "Cum Coll (Inst+Exp+BC)": "Cum Coll (Inst+Exp)",
}

# All expected columns (used for reference only — missing ones show a warning, not error)
REQUIRED_COLS = [
    "SNo", "Loan No", "CHANNEL", "BU", "StateName", "Zone", "RegionName", "Unit",
    "Ag_Date", "SRC Code", "SRC Name", "MNT CODE", "MNT NAME", "Due Dt", "Tenure",
    "Loan Status", "Loan Amount", "Veh ID", "Cust Name", "Guar Name", "Cust Mob No",
    "Guar Mob No", "Segment", "SegmentName", "Make", "Vehicle Description",
    "Year Of Manufacture", "Arrear Opening", "ARREARS OPEN AGAINST INST",
    "ARREARS OPEN AGAINST EXP", "ARREARS OPEN AGAINST BC", "ARREARS OPEN AGAINST PC",
    "OPENING RESERVE COLLECTION", "Month Due-Inst", "Month Due-Exp", "MONTH DUE (BC)",
    "MONTH DUE PC", "Month Receipt Amount", "MONTHCOLL INST", "MONTHCOLL EXP",
    "MONTHCOLL BC", "MONTHCOLL PC", "Month Collection (Excluding Reserve Collection)",
    "Closing Arrears", "UN-CLEARED CHEQUE FOR THE MONTH/Amount Not remitted by RE",
    "Cum Due-Inst", "Cum Due-Exp", "CUM DUE (BC)", "Cum Due PC", "Cum Coll (Inst+Exp)",
    "Total Cum Collection", "ARREARS AGAINST INST", "ARREARS AGAINST EXP",
    "ARREARS AGAINST BC", "ARREARS AGAINST PC", "CLOSING RESERVE COLLECTION",
    "Arrears against Inst+Exp", "Uncleared Cheque/Amount Not remitted by RE", "LCC%",
    "Arrears / EMI", "DelinquencyDays", "VehEMI Accrued", "ClosingPC", "POS", "scheme",
    "Non Starter", "Strike", "RCEndors(>90Days)", "RTO / INSURANCE",
    "NET Collection Demand Inst+Exp", "Net Collection Demand Inst+Exp+BC",
    "NET COLLECTION", "NET COLLECTION EXCLUDING RESERVE COLL", "Last Receipt Date",
    "Last Receipt Amount", "ParentLDueDate", "No Coll 3 Months and >6 EMI",
    "NACHStatus", "SaleType", "CoLending_Loans", "CUSTOMER_STATUS", "LGL_FLAG",
    "LGL_DESCRIPTION", "TyreFlag", "FUEL_TYPE",
]

BUCKET_ORDER = ["STD", "1-30 DPD", "SMA-1", "SMA-2", "NPA", "NA"]
BUCKET_SCORE = {"STD": 0, "1-30 DPD": 1, "SMA-1": 2, "SMA-2": 3, "NPA": 4, "NA": -1}


def _assign_bucket(val):
    try:
        v = float(val)
    except (TypeError, ValueError):
        return "NA"
    if pd.isna(v):
        return "NA"
    if v == 0:
        return "STD"
    if v < 1:
        return "1-30 DPD"
    if v < 2:
        return "SMA-1"
    if v < 3:
        return "SMA-2"
    return "NPA"


def assign_buckets(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["curr_bucket"] = df["Arrears / EMI"].apply(_assign_bucket)
    df["curr_score"] = df["curr_bucket"].map(BUCKET_SCORE)
    return df


def load_and_validate(file) -> tuple[pd.DataFrame, list[str]]:
    try:
        df = pd.read_excel(file, engine="openpyxl")
    except Exception as e:
        return None, [f"Could not read file: {e}"]

    # Rename known column variations to standard names
    df.rename(columns={k: v for k, v in COL_ALIASES.items() if k in df.columns}, inplace=True)

    # Hard-fail only on critical columns
    missing_critical = [c for c in CRITICAL_COLS if c not in df.columns]
    if missing_critical:
        return None, [
            f"Missing critical column(s): {', '.join(missing_critical)}"
        ]

    for date_col in ["Ag_Date", "Last Receipt Date", "ParentLDueDate"]:
        if date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    # Due Dt is a numeric EMI due day (5, 10, 15, 20) — keep as number
    df["Due Dt"] = pd.to_numeric(df["Due Dt"], errors="coerce")

    for col in [
        "NET Collection Demand Inst+Exp", "Net Collection Demand Inst+Exp+BC",
        "Month Receipt Amount", "NET COLLECTION", "POS",
        "Cum Coll (Inst+Exp)", "Total Cum Collection", "Arrears / EMI",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    df["LCC%"] = pd.to_numeric(df["LCC%"], errors="coerce")
    df["Strike"] = df["Strike"].astype(str).str.strip().str.upper()
    if "Unit" in df.columns:
        df["Unit"] = df["Unit"].astype(str).str.strip().str.upper()

    df = assign_buckets(df)
    return df, []


def apply_filters(df: pd.DataFrame, region: str, branch: str, status: str) -> pd.DataFrame:
    if region != "All":
        df = df[df["RegionName"] == region]
    if branch != "All":
        df = df[df["Unit"] == branch]
    if status != "All":
        df = df[df["Loan Status"] == status]
    return df


def _safe_pct(num, den):
    if den == 0:
        return 0.0
    return round(num / den * 100, 2)


def _mom_pct(curr, prev):
    if prev == 0:
        return 0.0
    return round((curr - prev) / abs(prev) * 100, 2)


def compute_metrics(df_curr: pd.DataFrame, df_prev: pd.DataFrame) -> dict:
    def _calc(df):
        n_accounts = df["Loan No"].nunique()
        demand = df["NET Collection Demand Inst+Exp"].sum()
        collection = df["Month Receipt Amount"].sum()
        pos = df["POS"].sum()
        cum_coll_total = df["Total Cum Collection"].sum()
        cum_coll_inst_exp = df["Cum Coll (Inst+Exp)"].sum()

        strike_valid = df[df["Strike"].isin(["Y", "N"])]
        strike_pct = _safe_pct(
            (strike_valid["Strike"] == "Y").sum(),
            len(strike_valid),
        )

        npa_pct = _safe_pct(
            df[df["curr_bucket"] == "NPA"]["Loan No"].nunique(),
            n_accounts,
        )
        hard_pct = _safe_pct(
            df[df["curr_score"] >= 2]["Loan No"].nunique(),
            n_accounts,
        )
        lcc_avg = df["LCC%"].mean()
        lcc_avg = round(lcc_avg, 2) if not pd.isna(lcc_avg) else 0.0
        cmd_pct = _safe_pct(cum_coll_total, cum_coll_inst_exp)

        return {
            "Month Demand": demand,
            "Total Collection": collection,
            "Collection %": _safe_pct(collection, demand),
            "Strike %": strike_pct,
            "NPA %": npa_pct,
            "Hard Bucket %": hard_pct,
            "Count": n_accounts,
            "POS": pos,
            "LCC%": lcc_avg,
            "CMD %": cmd_pct,
        }

    curr = _calc(df_curr)
    prev = _calc(df_prev)
    result = {}
    for k in curr:
        result[k] = (curr[k], _mom_pct(curr[k], prev[k]))
    return result


def fmt_value(val, kind: str) -> str:
    if kind == "money":
        if abs(val) >= 1_000_000:
            return f"{val / 1_000_000:.1f}M"
        if abs(val) >= 1_000:
            return f"{val / 1_000:.1f}K"
        return f"{val:.0f}"
    if kind == "pct":
        return f"{val:.2f}"
    if kind == "count":
        if abs(val) >= 1_000_000:
            return f"{val / 1_000_000:.2f}M"
        if abs(val) >= 1_000:
            return f"{val / 1_000:.1f}K"
        return f"{int(val)}"
    return str(val)


def build_status_bar_chart(df: pd.DataFrame) -> go.Figure:
    counts = (
        df.groupby("curr_bucket")["Loan No"]
        .nunique()
        .reindex(BUCKET_ORDER, fill_value=0)
    )
    fig = go.Figure(go.Bar(
        x=counts.index.tolist(),
        y=counts.values.tolist(),
        marker_color=YELLOW,
        text=counts.values.tolist(),
        textposition="outside",
        textfont=dict(size=12, color="#000000"),
    ))
    fig.update_layout(
        title=dict(text="Loan Count by DPD Bucket", font=dict(size=14, color="#000000")),
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis=dict(showgrid=False, tickfont=dict(color="#000000")),
        yaxis=dict(showgrid=False, visible=False),
        margin=dict(l=20, r=20, t=40, b=20),
        height=300,
    )
    return fig


def build_branch_bar_chart(df: pd.DataFrame) -> go.Figure:
    if len(df) == 0:
        return go.Figure()

    grp = df.groupby("Unit").agg(
        demand=("NET Collection Demand Inst+Exp", "sum"),
        collection=("Month Receipt Amount", "sum"),
    )
    grp["coll_pct"] = grp.apply(
        lambda r: _safe_pct(r["collection"], r["demand"]), axis=1
    )
    grp = grp.sort_values("coll_pct", ascending=True).tail(12)

    fig = go.Figure(go.Bar(
        x=grp["coll_pct"].tolist(),
        y=grp.index.tolist(),
        orientation="h",
        marker_color=YELLOW,
        text=[f"{v:.0f}" for v in grp["coll_pct"]],
        textposition="outside",
        textfont=dict(size=11, color="#000000"),
    ))
    fig.update_layout(
        title=dict(text="Collection % by Branch", font=dict(size=14, color="#000000")),
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis=dict(
            showgrid=False,
            title=dict(text="Collection %", font=dict(color="#000000")),
        ),
        yaxis=dict(showgrid=False, tickfont=dict(color="#000000")),
        margin=dict(l=20, r=60, t=40, b=20),
        height=300,
    )
    return fig


def build_closing_pc_chart(df: pd.DataFrame) -> go.Figure:
    """Arrears exposure by DPD bucket — SUM(ClosingPC) per bucket.
    Shows how much money is stuck at each risk level."""
    if len(df) == 0 or "ClosingPC" not in df.columns:
        return go.Figure()

    df = df.copy()
    df["ClosingPC"] = pd.to_numeric(df["ClosingPC"], errors="coerce").fillna(0)

    exposure = (
        df.groupby("curr_bucket")["ClosingPC"]
        .sum()
        .reindex(BUCKET_ORDER, fill_value=0)
    )

    # Format labels as M/K for readability
    def _fmt(v):
        if v >= 1_000_000:
            return f"{v/1_000_000:.1f}M"
        if v >= 1_000:
            return f"{v/1_000:.0f}K"
        return f"{v:.0f}"

    labels = [_fmt(v) for v in exposure.values]

    # Colour-code buckets: STD=green, 1-30=yellow, SMA-1=orange, SMA-2=orangered, NPA=red, NA=grey
    bucket_colors = {
        "STD":      "#16a34a",
        "1-30 DPD": YELLOW,
        "SMA-1":    "#f97316",
        "SMA-2":    "#ef4444",
        "NPA":      "#991b1b",
        "NA":       "#9ca3af",
    }
    colors = [bucket_colors.get(b, YELLOW) for b in exposure.index]

    fig = go.Figure(go.Bar(
        x=exposure.index.tolist(),
        y=exposure.values.tolist(),
        marker_color=colors,
        text=labels,
        textposition="outside",
        textfont=dict(size=12, color="#000000"),
    ))
    fig.update_layout(
        title=dict(text="Arrears Exposure (ClosingPC) by DPD Bucket", font=dict(size=14, color="#000000")),
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis=dict(showgrid=False, tickfont=dict(color="#000000")),
        yaxis=dict(showgrid=False, visible=False),
        margin=dict(l=20, r=20, t=40, b=20),
        height=300,
    )
    return fig


def _kpi_card_html(label: str, value: str, mom: float, unit: str = "") -> str:
    arrow = "&#9650;" if mom >= 0 else "&#9660;"
    color = "#00A651" if mom >= 0 else "#CC0000"
    mom_str = f"{abs(mom):.2f}%"
    return (
        f'<div style="border:2px solid {YELLOW};border-radius:8px;padding:16px 12px;'
        f'background:#fff;text-align:center;min-width:130px;flex:1;">'
        f'<div style="font-size:13px;font-weight:600;color:#333;margin-bottom:6px;">{label}</div>'
        f'<div style="font-size:28px;font-weight:700;color:#000;line-height:1.1;">{value}{unit}</div>'
        f'<div style="font-size:12px;margin-top:6px;">'
        f'MoM %: <span style="color:{color};font-weight:600;">{arrow} {mom_str}</span>'
        f'</div></div>'
    )


def build_html_export(
    df_curr: pd.DataFrame,
    df_prev: pd.DataFrame,
    metrics: dict,
    fig_status: go.Figure,
    fig_branch: go.Figure,
    fig_trend: go.Figure,
    filters: dict,
) -> str:
    def _fig_html(fig):
        return pio.to_html(fig, full_html=False, include_plotlyjs=False)

    KPI_TOP = ["Month Demand", "Total Collection", "Collection %", "Strike %", "NPA %", "Hard Bucket %"]
    KPI_BOT = ["Count", "POS", "CMD %"]
    KINDS = {
        "Month Demand": "money", "Total Collection": "money", "Collection %": "pct",
        "Strike %": "pct", "NPA %": "pct", "Hard Bucket %": "pct",
        "Count": "count", "POS": "money", "LCC%": "pct", "CMD %": "pct",
    }

    def _cards(keys, style=""):
        cards = "".join(
            _kpi_card_html(k, fmt_value(metrics[k][0], KINDS[k]), metrics[k][1])
            for k in keys
        )
        return f'<div style="display:flex;gap:12px;flex-wrap:wrap;{style}">{cards}</div>'

    filter_info = " | ".join(f"{k}: {v}" for k, v in filters.items() if v != "All") or "All data"

    lcc_card = _kpi_card_html("LCC%", fmt_value(metrics["LCC%"][0], "pct"), metrics["LCC%"][1])

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Shriram Finance – Regional Collection Dashboard</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
  body {{font-family:Arial,sans-serif;background:#f5f5f5;margin:0;padding:0;}}
  .header {{background:#fff;border-bottom:4px solid {YELLOW};padding:16px 32px;
            display:flex;align-items:center;gap:24px;}}
  .header h1 {{font-size:22px;color:#000;margin:0;}}
  .logo {{font-size:26px;font-weight:900;color:{YELLOW};letter-spacing:-1px;}}
  .content {{padding:24px 32px;}}
  .filter-bar {{font-size:12px;color:#666;margin-bottom:16px;}}
  .mid-row {{display:grid;grid-template-columns:1fr 1fr auto;gap:16px;margin-top:16px;}}
  .bot-row {{display:grid;grid-template-columns:3fr 2fr;gap:16px;margin-top:16px;}}
  .chart-box {{background:#fff;border:1px solid #e0e0e0;border-radius:8px;padding:8px;}}
  .lcc-box {{display:flex;align-items:center;justify-content:center;min-width:160px;}}
</style>
</head>
<body>
<div class="header">
  <div class="logo">SHRIRAM<br><span style="font-size:14px;font-weight:400;">Finance</span></div>
  <h1>Shriram Finance &ndash; Regional Collection Dashboard</h1>
</div>
<div class="content">
  <div class="filter-bar">Filters: {filter_info}</div>
  {_cards(KPI_TOP, "margin-bottom:16px;")}
  <div class="mid-row">
    <div class="chart-box">{_fig_html(fig_status)}</div>
    <div class="chart-box">{_fig_html(fig_branch)}</div>
    <div class="lcc-box">{lcc_card}</div>
  </div>
  <div class="bot-row">
    <div class="chart-box">{_fig_html(fig_trend)}</div>
    <div style="display:flex;flex-direction:column;justify-content:center;">
      {_cards(KPI_BOT)}
    </div>
  </div>
</div>
</body>
</html>"""
    return html
