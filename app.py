import os
from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import pandas as pd
from utils import (
    load_and_validate,
    apply_filters,
    compute_metrics,
    fmt_value,
    build_status_bar_chart,
    build_branch_bar_chart,
    build_closing_pc_chart,
    build_html_export,
    _kpi_card_html,
)
from graph import run_query
from smart_alerts import run_all_alerts

st.set_page_config(
    page_title="Shriram Finance Dashboard",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

/* ── Global reset ── */
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 2rem 2rem 2rem !important; max-width: 100% !important; }

/* ── Page background ── */
.stApp { background: #f0f2f5; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #111111 0%, #1e1e1e 100%) !important;
    border-right: 3px solid #FFC000;
}
[data-testid="stSidebar"] * { color: #e0e0e0 !important; }
[data-testid="stSidebar"] .stSelectbox > label { color: #FFC000 !important; font-weight: 600; font-size: 13px; }
[data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] > div {
    background: #2a2a2a !important; border-color: #444 !important; color: #fff !important;
}
[data-testid="stSidebar"] hr { border-color: #333 !important; }

/* ── Top header banner ── */
.top-banner {
    background: linear-gradient(90deg, #FFC000 0%, #FFD740 100%);
    height: 6px; margin: 0 -2rem 0 -2rem; margin-bottom: 0;
}
.dash-header {
    background: #fff;
    display: flex; align-items: center; gap: 20px;
    padding: 14px 24px;
    border-bottom: 1px solid #e8e8e8;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    margin: 0 -2rem 20px -2rem;
}
.dash-logo-box {
    background: #1a1a1a; border-radius: 8px;
    padding: 8px 14px; display: flex; flex-direction: column; align-items: center;
}
.dash-logo-main { font-size: 20px; font-weight: 900; color: #FFC000; letter-spacing: 1px; line-height: 1; }
.dash-logo-sub  { font-size: 11px; font-weight: 400; color: #aaa; }
.dash-title     { font-size: 22px; font-weight: 700; color: #1a1a1a; }
.dash-subtitle  { font-size: 13px; color: #888; margin-top: 2px; }

/* ── Section headers ── */
.section-label {
    font-size: 11px; font-weight: 700; color: #888;
    text-transform: uppercase; letter-spacing: 1.5px;
    margin-bottom: 10px; margin-top: 4px;
}

/* ── Upload card ── */
.upload-card {
    background: #fff; border-radius: 12px;
    border: 2px dashed #ddd; padding: 18px 20px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04);
    transition: border-color 0.2s;
}
.upload-card:hover { border-color: #FFC000; }
.upload-card-title { font-size: 13px; font-weight: 600; color: #333; margin-bottom: 4px; }
.upload-card-sub   { font-size: 11px; color: #999; margin-bottom: 10px; }

/* ── KPI cards ── */
.kpi-row { display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 16px; }
.kpi-card {
    background: #fff; border-radius: 12px;
    border-left: 4px solid #FFC000;
    padding: 14px 16px; min-width: 140px; flex: 1;
    box-shadow: 0 2px 8px rgba(0,0,0,0.07);
}
.kpi-label { font-size: 11px; font-weight: 600; color: #888; text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 6px; }
.kpi-value { font-size: 26px; font-weight: 800; color: #111; line-height: 1; }
.kpi-mom   { font-size: 12px; margin-top: 6px; color: #666; }
.kpi-mom-up   { color: #16a34a; font-weight: 700; }
.kpi-mom-down { color: #dc2626; font-weight: 700; }

/* ── Chart containers ── */
.chart-card {
    background: #fff; border-radius: 12px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.07);
    padding: 4px; overflow: hidden;
}

/* ── Generate button ── */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #FFC000, #FFD740) !important;
    color: #000 !important; font-weight: 700 !important;
    border: none !important; border-radius: 8px !important;
    padding: 10px 28px !important; font-size: 14px !important;
    box-shadow: 0 4px 12px rgba(255,192,0,0.4) !important;
    transition: all 0.2s !important;
}
.stButton > button[kind="primary"]:hover {
    box-shadow: 0 6px 16px rgba(255,192,0,0.5) !important;
    transform: translateY(-1px) !important;
}

/* ── Divider ── */
hr { border-color: #e8e8e8 !important; margin: 24px 0 !important; }

/* ── Date input — yellow text ── */
.stDateInput input {
    color: #FFC000 !important;
    background: #1a1a2e !important;
    border: 1px solid #333 !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
}
.stDateInput input:focus { border-color: #FFC000 !important; box-shadow: 0 0 0 2px rgba(255,192,0,0.2) !important; }

/* Calendar popup */
[data-baseweb="calendar"] {
    background: #2a2a2a !important;
    border: 1px solid #3a3a3a !important;
    border-radius: 10px !important;
    box-shadow: 0 8px 24px rgba(0,0,0,0.3) !important;
}
[data-baseweb="calendar"] * { color: #ccc !important; }
[data-baseweb="calendar"] [data-baseweb="select"] span,
[data-baseweb="calendar"] button[aria-label*="month"],
[data-baseweb="calendar"] button[aria-label*="year"] {
    color: #FFC000 !important; font-weight: 700 !important;
}
[data-baseweb="calendar"] [aria-selected="true"] > div {
    background: #FFC000 !important; color: #000 !important; border-radius: 50% !important;
}
[data-baseweb="calendar"] button:hover > div {
    background: rgba(255,255,255,0.08) !important;
    border-radius: 50% !important;
    transition: background 0.15s ease !important;
}
[data-baseweb="calendar"] [data-baseweb="calendar-header"] {
    background: #2a2a2a !important; border-bottom: 1px solid #3a3a3a !important;
}
[data-baseweb="calendar"] abbr { color: #888 !important; font-size: 11px !important; }

/* ── Text area ── */
.stTextArea textarea {
    background: #1a1a2e !important;
    color: #FFC000 !important;
    border: 1px solid #333 !important;
    border-radius: 8px !important;
    font-size: 14px !important;
    caret-color: #FFC000 !important;
}
.stTextArea textarea::placeholder { color: #555 !important; }
.stTextArea textarea:focus { border-color: #FFC000 !important; box-shadow: 0 0 0 2px rgba(255,192,0,0.2) !important; }

/* ── AI section ── */
.ai-panel {
    background: linear-gradient(135deg, #0f0f0f 0%, #1a1a2e 100%);
    border-radius: 16px; padding: 28px 32px;
    border: 1px solid #2a2a2a;
    box-shadow: 0 8px 32px rgba(0,0,0,0.25);
    margin-top: 8px;
}
.ai-header {
    display: flex; align-items: center; gap: 12px; margin-bottom: 6px;
}
.ai-icon { font-size: 28px; }
.ai-title { font-size: 20px; font-weight: 800; color: #FFC000; }
.ai-subtitle { font-size: 13px; color: #888; margin-bottom: 20px; line-height: 1.6; }
.ai-example {
    display: inline-block; background: #1e1e2e; color: #aaa;
    border: 1px solid #333; border-radius: 6px;
    padding: 2px 10px; font-size: 12px; font-style: italic; margin: 2px 4px 2px 0;
}

/* ── Result KPI cards (query results) ── */
.result-kpi {
    background: #1e1e2e; border-radius: 10px;
    border-left: 3px solid #FFC000;
    padding: 12px 14px; text-align: center; flex: 1; min-width: 100px;
}
.result-kpi-label { font-size: 10px; font-weight: 600; color: #888; text-transform: uppercase; letter-spacing: 0.6px; margin-bottom: 4px; }
.result-kpi-value { font-size: 20px; font-weight: 800; color: #FFC000; }

/* ── Ranking cards ── */
.rank-card {
    background: #1a1a2e; border-radius: 10px;
    border: 1px solid #2a2a3e; padding: 16px 18px; height: 100%;
}
.rank-title { font-size: 12px; font-weight: 700; color: #FFC000; margin-bottom: 10px; text-transform: uppercase; letter-spacing: 0.8px; }
.rank-row { display: flex; justify-content: space-between; align-items: center; padding: 5px 0; border-bottom: 1px solid #2a2a3e; }
.rank-row:last-child { border-bottom: none; }
.rank-name  { font-size: 13px; color: #ddd; }
.rank-value { font-size: 13px; font-weight: 700; color: #fff; }

/* ── AI observations card ── */
.obs-card {
    background: #0f1923; border-radius: 12px;
    border-left: 4px solid #FFC000;
    padding: 20px 24px; margin-top: 16px;
}
.obs-title { font-size: 13px; font-weight: 700; color: #FFC000; margin-bottom: 12px; text-transform: uppercase; letter-spacing: 1px; }
.obs-line  { font-size: 14px; color: #d0d0d0; line-height: 1.7; padding: 3px 0; }

/* ── Sidebar filter header ── */
.filter-header {
    background: #FFC000; color: #000 !important; font-weight: 800;
    font-size: 14px; padding: 8px 0; border-radius: 6px;
    text-align: center; letter-spacing: 1px; margin-bottom: 16px;
}
</style>
""", unsafe_allow_html=True)

# ── Header ───────────────────────────────────────────────────────────────────
st.markdown('<div class="top-banner"></div>', unsafe_allow_html=True)
st.markdown("""
<div class="dash-header">
  <div class="dash-logo-box">
    <div class="dash-logo-main">SHRIRAM</div>
    <div class="dash-logo-sub">Finance</div>
  </div>
  <div>
    <div class="dash-title">Regional Collection Dashboard</div>
    <div class="dash-subtitle">Credit &amp; Collection Risk Monitoring</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── File upload ───────────────────────────────────────────────────────────────
import datetime

st.markdown('<div class="section-label">Data Source</div>', unsafe_allow_html=True)
col_up1, col_up2 = st.columns(2)

with col_up1:
    st.markdown('<div class="upload-card"><div class="upload-card-title">📂 Current Month</div><div class="upload-card-sub">Required — main reporting file</div>', unsafe_allow_html=True)
    curr_file = st.file_uploader("Current Month", type=["xlsx", "xls"], key="curr", label_visibility="collapsed")
    curr_month_input = st.date_input(
        "Reporting month (pick any day in that month)",
        value=datetime.date.today().replace(day=1),
        key="curr_month_pick",
        help="Pick any date within the reporting month — only month & year are used",
    )
    st.markdown('</div>', unsafe_allow_html=True)

with col_up2:
    st.markdown('<div class="upload-card"><div class="upload-card-title">📂 Previous Month</div><div class="upload-card-sub">Optional — enables MoM % comparison</div>', unsafe_allow_html=True)
    prev_file = st.file_uploader("Previous Month", type=["xlsx", "xls"], key="prev", label_visibility="collapsed")
    prev_month_input = st.date_input(
        "Reporting month (pick any day in that month)",
        value=(datetime.date.today().replace(day=1) - datetime.timedelta(days=1)).replace(day=1),
        key="prev_month_pick",
        help="Pick any date within the reporting month — only month & year are used",
        disabled=not prev_file,
    )
    st.markdown('</div>', unsafe_allow_html=True)

# Derive period labels from user-selected dates
curr_month = curr_month_input.strftime("%Y-%m")
prev_month = prev_month_input.strftime("%Y-%m") if prev_file else None

if not curr_file:
    st.markdown("""
    <div style="text-align:center;padding:40px 0;color:#999;">
        <div style="font-size:40px;margin-bottom:12px;">📊</div>
        <div style="font-size:16px;font-weight:600;color:#555;">Upload your LCC Excel file to begin</div>
        <div style="font-size:13px;margin-top:6px;">Supports .xlsx and .xls formats</div>
    </div>
    """, unsafe_allow_html=True)
    st.session_state.pop("df_curr_raw", None)
    st.session_state.pop("df_prev_raw", None)
    st.stop()

col_btn, _ = st.columns([1, 3])
with col_btn:
    generate = st.button("⚡  Generate Dashboard", type="primary", use_container_width=True)

# ── Load data — persisted in session_state so reruns (e.g. Run Query) don't reset ──
if generate:
    df_curr_raw, err_curr = load_and_validate(curr_file)
    if err_curr:
        st.error(f"Current month file: {err_curr[0]}")
        st.stop()
    if prev_file:
        df_prev_raw, err_prev = load_and_validate(prev_file)
        if err_prev:
            st.error(f"Previous month file: {err_prev[0]}")
            st.stop()
    else:
        df_prev_raw = df_curr_raw.iloc[0:0].copy()
    st.session_state["df_curr_raw"] = df_curr_raw
    st.session_state["df_prev_raw"] = df_prev_raw

if "df_curr_raw" not in st.session_state:
    st.markdown("""
    <div style="text-align:center;padding:20px 0;color:#aaa;font-size:13px;">
        File ready — click <strong>Generate Dashboard</strong> to build the report.
    </div>
    """, unsafe_allow_html=True)
    st.stop()

df_curr_raw = st.session_state["df_curr_raw"]
df_prev_raw = st.session_state["df_prev_raw"]

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="filter-header">⚙ FILTERS</div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div style="background:#2a2a2a;border-radius:8px;padding:10px 14px;margin-bottom:16px;">
      <div style="font-size:10px;color:#888;font-weight:600;text-transform:uppercase;letter-spacing:1px;">Reporting Month</div>
      <div style="font-size:18px;font-weight:800;color:#FFC000;margin-top:2px;">{curr_month}</div>
    </div>
    """, unsafe_allow_html=True)

    regions = ["All"] + sorted(df_curr_raw["RegionName"].dropna().unique().tolist())
    sel_region = st.selectbox("Region", regions)

    df_for_branch = df_curr_raw if sel_region == "All" else df_curr_raw[df_curr_raw["RegionName"] == sel_region]
    branches = ["All"] + sorted(df_for_branch["Unit"].dropna().unique().tolist())
    sel_branch = st.selectbox("Branch", branches)

    statuses = ["All"] + sorted(df_curr_raw["Loan Status"].dropna().unique().tolist())
    sel_status = st.selectbox("Loan Status", statuses)

    st.markdown("---")
    st.markdown(f"""
    <div style="font-size:11px;color:#555;text-align:center;">
        {len(df_curr_raw):,} total records loaded
    </div>
    """, unsafe_allow_html=True)

# ── Apply filters ──────────────────────────────────────────────────────────────
df_curr = apply_filters(df_curr_raw.copy(), sel_region, sel_branch, sel_status)
df_prev = apply_filters(df_prev_raw.copy(), sel_region, sel_branch, sel_status)

if len(df_curr) == 0:
    st.warning("No data matches the selected filters.")
    st.stop()

# ── Metrics ────────────────────────────────────────────────────────────────────
metrics = compute_metrics(df_curr, df_prev)

KIND = {
    "Month Demand": "money", "Total Collection": "money", "Collection %": "pct",
    "Strike %": "pct", "NPA %": "pct", "Hard Bucket %": "pct",
    "Count": "count", "POS": "money", "LCC%": "pct", "CMD %": "pct",
}

KPI_TOP = ["Month Demand", "Total Collection", "Collection %", "Strike %", "NPA %", "Hard Bucket %"]
KPI_BOT = ["Count", "POS", "CMD %"]


def _kpi_card_styled(label, value, mom):
    arrow = "▲" if mom >= 0 else "▼"
    cls   = "kpi-mom-up" if mom >= 0 else "kpi-mom-down"
    return (
        f'<div class="kpi-card">'
        f'  <div class="kpi-label">{label}</div>'
        f'  <div class="kpi-value">{value}</div>'
        f'  <div class="kpi-mom">MoM <span class="{cls}">{arrow} {abs(mom):.2f}%</span></div>'
        f'</div>'
    )


def _render_kpi_row(keys):
    html = "".join(_kpi_card_styled(k, fmt_value(metrics[k][0], KIND[k]), metrics[k][1]) for k in keys)
    st.markdown(f'<div class="kpi-row">{html}</div>', unsafe_allow_html=True)


# ── KPI top row ────────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">Key Performance Indicators</div>', unsafe_allow_html=True)
_render_kpi_row(KPI_TOP)

# ── Charts ─────────────────────────────────────────────────────────────────────
fig_status  = build_status_bar_chart(df_curr)
fig_branch  = build_branch_bar_chart(df_curr)
fig_closing = build_closing_pc_chart(df_curr)

st.markdown('<div class="section-label">Portfolio Analysis</div>', unsafe_allow_html=True)

col_bar, col_hbar, col_lcc = st.columns([2, 2, 1])
with col_bar:
    with st.container():
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.plotly_chart(fig_status, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
with col_hbar:
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.plotly_chart(fig_branch, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
with col_lcc:
    lcc_val  = fmt_value(metrics["LCC%"][0], "pct")
    lcc_mom  = metrics["LCC%"][1]
    lcc_arrow = "▲" if lcc_mom >= 0 else "▼"
    lcc_cls   = "kpi-mom-up" if lcc_mom >= 0 else "kpi-mom-down"
    st.markdown(f"""
    <div class="kpi-card" style="height:100%;display:flex;flex-direction:column;justify-content:center;align-items:center;text-align:center;margin-top:0;">
      <div class="kpi-label">LCC %</div>
      <div style="font-size:36px;font-weight:800;color:#111;line-height:1.1;">{lcc_val}</div>
      <div class="kpi-mom">MoM <span class="{lcc_cls}">{lcc_arrow} {abs(lcc_mom):.2f}%</span></div>
    </div>
    """, unsafe_allow_html=True)

col_trend, col_bot = st.columns([3, 2])
with col_trend:
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.plotly_chart(fig_closing, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
with col_bot:
    st.markdown("<br>", unsafe_allow_html=True)
    _render_kpi_row(KPI_BOT)

# ── Smart Alerts ──────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="section-label">Smart Alerts</div>', unsafe_allow_html=True)

SEVERITY_STYLE = {
    "critical": ("#dc2626", "#fff5f5"),
    "high":     ("#f97316", "#fff7ed"),
    "medium":   ("#d97706", "#fffbea"),
}
CLEAR_STYLE = ("#16a34a", "#f0fdf4")   # green when count == 0

alerts = run_all_alerts(df_curr_raw)

for i in range(0, len(alerts), 2):
    row_alerts = alerts[i:i+2]
    cols = st.columns(len(row_alerts))
    for col, alert in zip(cols, row_alerts):
        is_clear = alert["count"] == 0
        if is_clear:
            title_color, bg = CLEAR_STYLE
        else:
            title_color, bg = SEVERITY_STYLE.get(alert["severity"], ("#d97706", "#fffbea"))

        border = f"border-left:4px solid {title_color};"
        count_color = "#16a34a" if is_clear else title_color
        pos_fmt = fmt_value(alert["pos"], "money")
        pc_fmt  = fmt_value(alert["closing_pc"], "money")

        with col:
            st.markdown(f"""
            <div style="background:{bg};border-radius:12px;{border}
                        padding:16px 18px;box-shadow:0 2px 8px rgba(0,0,0,0.07);margin-bottom:4px;">
              <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;">
                <span style="font-size:20px;">{"✅" if is_clear else alert["icon"]}</span>
                <span style="font-size:14px;font-weight:700;color:{title_color};">{alert["title"]}</span>
              </div>
              <div style="font-size:12px;color:#666;margin-bottom:10px;">{alert["subtitle"]}</div>
              <div style="display:flex;gap:16px;margin-bottom:10px;">
                <div>
                  <div style="font-size:10px;color:#888;font-weight:600;text-transform:uppercase;">Accounts</div>
                  <div style="font-size:26px;font-weight:800;color:{count_color};">{alert["count"]}</div>
                </div>
                <div>
                  <div style="font-size:10px;color:#888;font-weight:600;text-transform:uppercase;">POS</div>
                  <div style="font-size:22px;font-weight:800;color:#111;">{pos_fmt if not is_clear else "—"}</div>
                </div>
                <div>
                  <div style="font-size:10px;color:#888;font-weight:600;text-transform:uppercase;">ClosingPC</div>
                  <div style="font-size:22px;font-weight:800;color:{title_color};">{pc_fmt if not is_clear else "—"}</div>
                </div>
              </div>
              <div style="font-size:11px;color:#555;font-style:italic;border-top:1px solid rgba(0,0,0,0.08);
                          padding-top:8px;">
                {"✓ All clear — no accounts flagged" if is_clear else f"💬 {alert['action']}"}
              </div>
            </div>
            """, unsafe_allow_html=True)

            if not is_clear:
                with st.expander(f"View {alert['count']} accounts"):
                    display_df = alert["df"]
                    display_df = display_df.loc[:, ~display_df.columns.duplicated()]
                    st.dataframe(
                        display_df.reset_index(drop=True),
                        use_container_width=True,
                        height=min(300, 40 + alert["count"] * 35),
                    )

# ── HTML export ────────────────────────────────────────────────────────────────
st.markdown("---")
col_dl, _ = st.columns([1, 3])
with col_dl:
    filters_applied = {"Region": sel_region, "Branch": sel_branch, "Loan Status": sel_status, "Year Month": str(curr_month)}
    html_content = build_html_export(df_curr, df_prev, metrics, fig_status, fig_branch, fig_closing, filters_applied)
    st.download_button(
        label="⬇  Download as HTML",
        data=html_content.encode("utf-8"),
        file_name=f"shriram_dashboard_{curr_month}.html",
        mime="text/html",
        use_container_width=True,
    )

# ── AI Query Assistant ─────────────────────────────────────────────────────────
st.markdown("""
<div class="ai-panel">
  <div class="ai-header">
    <span class="ai-icon">🤖</span>
    <span class="ai-title">AI Query Assistant</span>
  </div>
  <div class="ai-subtitle">
    Ask any question about your loan portfolio in plain English.<br>
    <span class="ai-example">"Show customers who haven't paid for 3 months"</span>
    <span class="ai-example">"List NPA accounts in MAHAD with POS above 1 lakh"</span>
    <span class="ai-example">"Show SMA-2 customers in PUNE region"</span>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<div style='margin-top:12px;'></div>", unsafe_allow_html=True)

ai_query = st.text_area(
    "Query",
    placeholder="Type your question here...",
    height=90,
    label_visibility="collapsed",
)

col_run, col_hint = st.columns([1, 4])
with col_run:
    run_btn = st.button("🔍  Run Query", type="primary", use_container_width=True)
with col_hint:
    st.markdown("<div style='padding-top:10px;font-size:12px;color:#aaa;'>Powered by Gemini Flash 2 · LangGraph multi-agent pipeline</div>", unsafe_allow_html=True)

if run_btn:
    if not ai_query.strip():
        st.warning("Please enter a question.")
    elif not os.environ.get("GOOGLE_API_KEY"):
        st.error("GOOGLE_API_KEY not found in .env file.")
    else:
        with st.spinner("Running multi-agent pipeline..."):
            st.session_state["ai_result"] = run_query(ai_query.strip(), df_curr_raw)

# ── Render AI result (persists across reruns via session_state) ───────────────
result = st.session_state.get("ai_result")
if result:
    if result.get("error"):
        st.error(f"Query failed: {result['error']}")
    else:
        filtered_df = result["result_df"]
        kpis_q      = result["result_kpis"]
        rankings    = result["result_rankings"]
        insights    = result["insights"]
        plain       = result["parsed_filters"].get("plain_english", "")

        # Domain Expert interpretation card
        is_priority = result.get("priority_mode", False)
        category   = result.get("query_category", "general").replace("_", " ").title()
        query_title = result.get("query_title", "")
        enriched   = result.get("enriched_query", "")
        risk_flag  = result.get("risk_flag", "medium")
        risk_color = {"high": "#dc2626", "medium": "#d97706", "low": "#16a34a"}.get(risk_flag, "#d97706")
        risk_label = {"high": "🔴 High Risk", "medium": "🟡 Medium Risk", "low": "🟢 Low Risk"}.get(risk_flag, "🟡 Medium Risk")

        # Domain Expert card
        st.markdown(f"""
        <div style="background:#111827;border:1px solid #2a2a3e;border-radius:12px;padding:16px 20px;margin:16px 0;">
          <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:10px;">
            <div style="display:flex;align-items:center;gap:10px;">
              <span style="font-size:11px;font-weight:700;background:#1e293b;color:#94a3b8;
                           padding:3px 10px;border-radius:20px;text-transform:uppercase;letter-spacing:1px;">{category}</span>
              <span style="font-size:14px;font-weight:700;color:#f1f5f9;">{query_title}</span>
            </div>
            <span style="font-size:11px;font-weight:700;color:{risk_color};">{risk_label}</span>
          </div>
          <div style="font-size:12px;color:#94a3b8;font-style:italic;line-height:1.6;border-top:1px solid #1e293b;padding-top:10px;">
            <span style="color:#FFC000;font-weight:600;font-style:normal;">🧠 Domain Expert:</span> &nbsp;{enriched}
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Priority mode: account count selector + grouped display
        if is_priority:
            from agents.domain_expert import PRIORITY_RULES
            from agents.data_executor import distribute_priority_accounts

            st.markdown(f"""
            <div style="background:#0f172a;border:1px solid #FFC000;border-radius:12px;
                        padding:16px 20px;margin:0 0 16px 0;">
              <div style="font-size:13px;font-weight:800;color:#FFC000;margin-bottom:8px;letter-spacing:1px;">
                🎯 PRIORITY ACTION MODE
              </div>
              <div style="font-size:11px;color:#94a3b8;">
                Accounts ranked across 7 priority rules. Each account appears only once under its highest priority.
                Total available: <strong style="color:#fff">{kpis_q.get('Count', 0)} accounts</strong>
              </div>
            </div>
            """, unsafe_allow_html=True)

            sel_col, _ = st.columns([1, 3])
            with sel_col:
                n_accounts = st.selectbox(
                    "How many accounts to review?",
                    options=[20, 30, 40, 50],
                    index=1,
                    key="priority_n",
                )

            distributed = distribute_priority_accounts(filtered_df, n_accounts)

            # Show each priority group separately
            for priority_label, grp in distributed.groupby("Priority", sort=False):
                p_num = priority_label.split(":")[0].strip()
                p_name = priority_label.split(":")[-1].strip()
                p_color = "#dc2626" if p_num in ("P1", "P5") else "#f97316" if p_num in ("P2", "P3", "P6", "P7") else "#d97706"

                st.markdown(f"""
                <div style="display:flex;align-items:center;gap:10px;margin:16px 0 6px 0;">
                  <span style="background:{p_color};color:#fff;font-size:11px;font-weight:800;
                               padding:3px 10px;border-radius:12px;">{p_num}</span>
                  <span style="font-size:14px;font-weight:700;color:#1a1a1a;">{p_name}</span>
                  <span style="font-size:12px;color:#888;">{len(grp)} accounts</span>
                </div>
                """, unsafe_allow_html=True)

                display_grp = grp.drop(columns=["Priority", "Why", "_rank"], errors="ignore")
                display_grp = display_grp.loc[:, ~display_grp.columns.duplicated()]
                st.dataframe(display_grp.reset_index(drop=True), use_container_width=True,
                             height=min(280, 45 + len(grp) * 36))

        else:
            # Normal query result
            st.markdown(f"""
            <div style="background:#1a2e1a;border-left:4px solid #16a34a;border-radius:8px;
                        padding:12px 16px;margin:0 0 16px 0;color:#86efac;font-weight:600;font-size:14px;">
                ✓ Found <strong style="color:#fff">{kpis_q.get('Count',0)} accounts</strong>
                &nbsp; {plain}
            </div>
            """, unsafe_allow_html=True)

            # Result KPIs
            QKIND = {"Count":"count","Total POS":"money","Avg Arrears/EMI":"pct",
                     "Total Demand":"money","Total Collection":"money","Collection %":"pct"}
            kpi_html = "".join(
                f'<div class="result-kpi"><div class="result-kpi-label">{k}</div>'
                f'<div class="result-kpi-value">{fmt_value(v, QKIND[k])}</div></div>'
                for k, v in kpis_q.items()
            )
            st.markdown(
                f'<div style="display:flex;gap:10px;flex-wrap:wrap;margin-bottom:20px;">{kpi_html}</div>',
                unsafe_allow_html=True,
            )

            # Rankings
            r1, r2 = st.columns(2)

            def _rank_rows(items, val_fmt="count"):
                rows = ""
                for name, val in list(items)[:5]:
                    display = fmt_value(val, val_fmt) if val_fmt == "money" else f"{int(val)}"
                    rows += f'<div class="rank-row"><span class="rank-name">{name}</span><span class="rank-value">{display}</span></div>'
                return rows

            with r1:
                left_html = ""
                if rankings.get("region_counts"):
                    left_html += f'<div class="rank-card" style="margin-bottom:12px;"><div class="rank-title">🗺 Top Regions by Account Count</div>{_rank_rows(rankings["region_counts"].items())}</div>'
                if rankings.get("bucket_dist"):
                    bucket_rows = "".join(
                        f'<div class="rank-row"><span class="rank-name">{b}</span><span class="rank-value">{p}%</span></div>'
                        for b, p in rankings["bucket_dist"].items()
                    )
                    left_html += f'<div class="rank-card"><div class="rank-title">📊 Bucket Distribution</div>{bucket_rows}</div>'
                st.markdown(left_html, unsafe_allow_html=True)

            with r2:
                right_html = ""
                if rankings.get("branch_counts"):
                    right_html += f'<div class="rank-card" style="margin-bottom:12px;"><div class="rank-title">🏢 Top Branches by Account Count</div>{_rank_rows(rankings["branch_counts"].items())}</div>'
                if rankings.get("branch_pos"):
                    right_html += f'<div class="rank-card"><div class="rank-title">💰 Top Branches by POS</div>{_rank_rows(rankings["branch_pos"].items(), "money")}</div>'
                st.markdown(right_html, unsafe_allow_html=True)

            # Customer table
            st.markdown("<div style='margin-top:20px;font-size:11px;font-weight:700;color:#888;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px;'>Matching Customer Records</div>", unsafe_allow_html=True)
            display_filtered = filtered_df.loc[:, ~filtered_df.columns.duplicated()]
            st.dataframe(display_filtered, use_container_width=True, height=320)

        # AI observations (always shown)
        obs_lines = "".join(
            f'<div class="obs-line">{line}</div>'
            for line in insights.split("\n") if line.strip()
        )
        st.markdown(f"""
        <div class="obs-card">
          <div class="obs-title">💡 AI Observations</div>
          {obs_lines}
        </div>
        """, unsafe_allow_html=True)
