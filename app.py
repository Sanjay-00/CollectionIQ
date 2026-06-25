import datetime
import os

from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import pandas as pd

from utils import apply_filters, compute_metrics
from smart_alerts import run_all_alerts

from ui.styles import inject_styles
from ui.header import render_header
from ui.landing import render_landing
from ui.sidebar import render_sidebar
from ui.components import _load_and_concat
from ui.tabs.dashboard import render_dashboard_tab
from ui.tabs.scorecard import render_scorecard_tab
from ui.tabs.alerts import render_alerts_tab
from ui.tabs.migration import render_migration_tab
from ui.tabs.ai_query import render_ai_query_tab
from ui.tabs.report import render_report_tab

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Shriram Finance Dashboard",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_styles()
render_header()

# ── File upload section ───────────────────────────────────────────────────────
# Pre-populate date pickers for sample data
if st.session_state.pop("_set_sample_dates", False):
    st.session_state["curr_month_pick"] = datetime.date(2026, 3, 1)
    st.session_state["prev_month_pick"] = datetime.date(2026, 2, 1)

st.markdown('<div class="section-label">Data Source</div>', unsafe_allow_html=True)
col_up1, col_up2 = st.columns(2)

with col_up1:
    st.markdown(
        '<div class="upload-card">'
        '<div class="upload-card-title">📂 Current Month</div>'
        '<div class="upload-card-sub">Required - upload one or multiple regional files</div>',
        unsafe_allow_html=True,
    )
    curr_file = st.file_uploader(
        "Current Month", type=["xlsx", "xls", "xlsb"],
        key="curr", label_visibility="collapsed", accept_multiple_files=True,
    )
    curr_month_input = st.date_input(
        "Reporting Month", value=datetime.date.today().replace(day=1),
        key="curr_month_pick",
        help="Select any date in the reporting month - only Month & Year are used",
        format="DD/MM/YYYY",
    )
    st.markdown("</div>", unsafe_allow_html=True)

with col_up2:
    st.markdown(
        '<div class="upload-card">'
        '<div class="upload-card-title">📂 Previous Month</div>'
        '<div class="upload-card-sub">Optional - upload one or multiple regional files</div>',
        unsafe_allow_html=True,
    )
    prev_file = st.file_uploader(
        "Previous Month", type=["xlsx", "xls", "xlsb"],
        key="prev", label_visibility="collapsed", accept_multiple_files=True,
    )
    prev_month_input = st.date_input(
        "Reporting Month",
        value=(datetime.date.today().replace(day=1) - datetime.timedelta(days=1)).replace(day=1),
        key="prev_month_pick",
        help="Select any date in the previous month - only Month & Year are used",
        format="DD/MM/YYYY",
        disabled=(not prev_file and not st.session_state.get("_sample_loaded")),
    )
    st.markdown("</div>", unsafe_allow_html=True)

curr_month = curr_month_input.strftime("%Y-%m")
prev_month = prev_month_input.strftime("%Y-%m") if (prev_file or st.session_state.get("_sample_loaded")) else None

# ── Landing page (no data yet) ────────────────────────────────────────────────
if not curr_file and not st.session_state.get("_sample_loaded"):
    render_landing()
    st.stop()

# ── Generate button ───────────────────────────────────────────────────────────
col_btn, _ = st.columns([1, 3])
with col_btn:
    generate = st.button("⚡  Generate Dashboard", type="primary", width='stretch')

# ── Load & cache data ─────────────────────────────────────────────────────────
if generate and curr_file:
    for _k in ["df_curr_raw", "df_prev_raw", "ai_result", "report_result", "_last_filter_key", "_sample_loaded", "_sel_branch", "_prev_region"]:
        st.session_state.pop(_k, None)

    n_curr = len(curr_file) if isinstance(curr_file, list) else 1
    with st.spinner(f"Loading {n_curr} current month file(s)..."):
        df_curr_raw, err_curr = _load_and_concat(curr_file)
    if df_curr_raw is None:
        st.error(f"Current month: {err_curr[0]}")
        st.stop()
    for _e in err_curr:
        st.warning(f"Skipped: {_e}")

    if prev_file:
        n_prev = len(prev_file) if isinstance(prev_file, list) else 1
        with st.spinner(f"Loading {n_prev} previous month file(s)..."):
            df_prev_raw, err_prev = _load_and_concat(prev_file)
        if df_prev_raw is None:
            st.error(f"Previous month: {err_prev[0]}")
            st.stop()
        for _e in err_prev:
            st.warning(f"Skipped: {_e}")
    else:
        df_prev_raw = pd.DataFrame()

    st.session_state["df_curr_raw"] = df_curr_raw
    st.session_state["df_prev_raw"] = df_prev_raw
    st.rerun()

if "df_curr_raw" not in st.session_state:
    st.markdown(
        '<div style="text-align:center;padding:20px 0;color:#aaa;font-size:13px;">'
        'File ready - click <strong>Generate Dashboard</strong> to build the report.'
        '</div>',
        unsafe_allow_html=True,
    )
    st.stop()

df_curr_raw: pd.DataFrame = st.session_state["df_curr_raw"]
df_prev_raw: pd.DataFrame = st.session_state["df_prev_raw"]

# Auto-load prev if uploaded after initial generate (cache hit — no cost)
if prev_file and len(df_prev_raw) == 0:
    _prev_tmp, _prev_err = _load_and_concat(prev_file)
    if _prev_tmp is not None:
        df_prev_raw = _prev_tmp

# ── Sidebar filters ───────────────────────────────────────────────────────────
sel_region, sel_branch, sel_status = render_sidebar(df_curr_raw, curr_month)

# ── Apply filters ─────────────────────────────────────────────────────────────
df_curr = apply_filters(df_curr_raw.copy(), sel_region, sel_branch, sel_status)
df_prev = apply_filters(df_prev_raw.copy(), sel_region, sel_branch, sel_status)

# Attach prev_bucket for roll-forward/backward agent queries
if len(df_prev_raw) > 0 and "Loan No" in df_curr.columns and "curr_bucket" in df_prev_raw.columns:
    _prev_slim = df_prev_raw[["Loan No", "curr_bucket"]].rename(columns={"curr_bucket": "prev_bucket"})
    df_curr = df_curr.merge(_prev_slim, on="Loan No", how="left")

# Clear AI/report results when filters change
_filter_key = f"{sel_region}|{sel_branch}|{sel_status}"
if st.session_state.get("_last_filter_key") != _filter_key:
    st.session_state.pop("ai_result", None)
    st.session_state.pop("report_result", None)
    st.session_state["_last_filter_key"] = _filter_key

if len(df_curr) == 0:
    st.warning("No data matches the selected filters.")
    st.stop()

# ── Cached computation wrappers ───────────────────────────────────────────────
# Keyed on DataFrame content — cache hit on reruns (e.g. Run Query) with same filters.

@st.cache_data(show_spinner=False)
def _cached_metrics(df_c: pd.DataFrame, df_p: pd.DataFrame):
    return compute_metrics(df_c, df_p)

@st.cache_data(show_spinner=False)
def _cached_alerts(df_c: pd.DataFrame):
    return run_all_alerts(df_c)

@st.cache_data(show_spinner=False)
def _cached_scorecard(df_c: pd.DataFrame):
    from analysis.executive_scorecard import compute_executive_scorecard
    return compute_executive_scorecard(df_c)

@st.cache_data(show_spinner=False)
def _cached_roll_rate(df_c: pd.DataFrame, df_p: pd.DataFrame):
    from analysis.roll_rate import compute_roll_rate_matrix
    return compute_roll_rate_matrix(df_c, df_p)

# ── Pre-compute shared data ───────────────────────────────────────────────────
metrics = _cached_metrics(df_curr, df_prev)
alerts  = _cached_alerts(df_curr)

scorecard_df = None
if "MNT NAME" in df_curr.columns:
    scorecard_df = _cached_scorecard(df_curr)

rr_matrix, rr_meta = None, None
if len(df_prev_raw) > 0:
    rr_matrix, rr_meta = _cached_roll_rate(df_curr, df_prev)

# ── Active filter bar ─────────────────────────────────────────────────────────
active_filters = {k: v for k, v in {"Region": sel_region, "Branch": sel_branch, "Loan Status": sel_status}.items() if v != "All"}
if active_filters:
    chips = " ".join(
        f'<span class="filter-chip">{k}: {v}</span>'
        for k, v in active_filters.items()
    )
    st.markdown(
        f'<div class="filter-bar">🔍 <strong>Active filters:</strong> {chips}'
        f' &nbsp;<span style="color:#92400e;font-size:11px;">{len(df_curr):,} records</span></div>',
        unsafe_allow_html=True,
    )

# ── Tabs ──────────────────────────────────────────────────────────────────────
n_alerts    = sum(1 for a in alerts if a["count"] > 0)
alert_label = f"🚨 Alerts ({n_alerts})" if n_alerts else "✅ Alerts"

tabs = st.tabs(["🗂️ Dashboard", "👤 Scorecard", alert_label, "📈 Migration", "🤖 AI Query", "📋 Report"])


def _tab_error(name: str, exc: Exception) -> None:
    st.error(f"**{name} tab failed to render:** {exc}")
    st.caption("Try clearing the cache from the sidebar, or check your data file.")


with tabs[0]:
    try:
        render_dashboard_tab(
            df_curr, df_prev, metrics, curr_month,
            sel_region, sel_branch, sel_status,
            alerts, scorecard_df, rr_meta,
        )
    except Exception as _e:
        _tab_error("Dashboard", _e)

with tabs[1]:
    try:
        render_scorecard_tab(df_curr, scorecard_df)
    except Exception as _e:
        _tab_error("Scorecard", _e)

with tabs[2]:
    try:
        render_alerts_tab(df_curr, alerts)
    except Exception as _e:
        _tab_error("Alerts", _e)

with tabs[3]:
    try:
        render_migration_tab(df_curr, df_prev_raw, rr_matrix, rr_meta)
    except Exception as _e:
        _tab_error("Migration", _e)

with tabs[4]:
    try:
        render_ai_query_tab(df_curr)
    except Exception as _e:
        _tab_error("AI Query", _e)

with tabs[5]:
    try:
        render_report_tab(
            df_curr, df_prev, curr_month, prev_month,
            sel_region, sel_branch, sel_status,
            scorecard_df, rr_meta,
        )
    except Exception as _e:
        _tab_error("Report", _e)
