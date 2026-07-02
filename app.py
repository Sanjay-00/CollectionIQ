import datetime
import os

from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import pandas as pd

from utils import apply_filters, compute_metrics, PREV_CARRYOVER_COLS
from smart_alerts import run_all_alerts

from ui.styles import inject_styles
from ui.header import render_header
from ui.landing import render_landing
from ui.sidebar import render_sidebar
from ui.components import _load_and_concat
from analysis.executive_scorecard import compute_executive_scorecard
from analysis.roll_rate import compute_roll_rate_matrix
from ui.tabs.dashboard import render_dashboard_tab
from ui.tabs.scorecard import render_scorecard_tab
from ui.tabs.alerts import render_alerts_tab
from ui.tabs.migration import render_migration_tab
from ui.tabs.portfolio_intelligence import render_portfolio_intelligence_tab
from analysis.portfolio_intelligence import (
    compute_pulse_kpis, compute_bucket_waterfall,
    compute_region_scorecard, compute_branch_quadrant,
    compute_executive_recovery, compute_product_analysis,
    compute_risk_indicators, compute_good_bad,
    compute_concentration_treemap, compute_fleet_exposure,
    compute_top_accounts, compute_repossession_list,
    compute_risk_flag_comparison, compute_npa_sma2_comparison,
    compute_good_customers,
)
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

# Auto-load prev if uploaded after initial generate (cache hit  -  no cost)
if prev_file and len(df_prev_raw) == 0:
    _prev_tmp, _prev_err = _load_and_concat(prev_file)
    if _prev_tmp is not None:
        df_prev_raw = _prev_tmp

# ── Sidebar filters ───────────────────────────────────────────────────────────
sel_region, sel_branch, sel_status, sel_segment = render_sidebar(df_curr_raw, curr_month)

# ── Cached computation wrappers (module-level  -  registered once, not per rerun) ──
@st.cache_data(show_spinner=False)
def _cached_filter(df_c: pd.DataFrame, df_p_raw: pd.DataFrame, region: str, branch: str, status: str, segment: tuple = ()):
    df = apply_filters(df_c.copy(), region, branch, status, segment)
    df_p = apply_filters(df_p_raw.copy(), region, branch, status, segment)
    if len(df_p_raw) > 0 and "Loan No" in df.columns and "curr_bucket" in df_p_raw.columns:
        # Carry over the prev-month bucket plus a curated set of numeric columns
        # (renamed prev_*) so the AI can compute month-over-month reductions.
        carry = {"curr_bucket": "prev_bucket"}
        carry.update({src: dst for src, dst in PREV_CARRYOVER_COLS.items() if src in df_p_raw.columns})
        slim = df_p_raw[["Loan No", *carry.keys()]].rename(columns=carry)
        df = df.merge(slim, on="Loan No", how="left")
    return df, df_p

@st.cache_data(show_spinner=False)
def _cached_metrics(df_c: pd.DataFrame, df_p: pd.DataFrame):
    return compute_metrics(df_c, df_p)

@st.cache_data(show_spinner=False)
def _cached_alerts(df_c: pd.DataFrame):
    return run_all_alerts(df_c)

@st.cache_data(show_spinner=False)
def _cached_scorecard(df_c: pd.DataFrame):
    return compute_executive_scorecard(df_c)

@st.cache_data(show_spinner=False)
def _cached_roll_rate(df_c: pd.DataFrame, df_p: pd.DataFrame):
    return compute_roll_rate_matrix(df_c, df_p)

@st.cache_data(show_spinner=False)
def _cached_portfolio_intel(
    df_c: pd.DataFrame, df_p: pd.DataFrame,
    rr_matched: int, rr_fwd: float, rr_bwd: float, rr_formation: float,
    alerts_curr_counts: tuple, alerts_prev_counts: tuple,
):
    rr_meta_local = {
        "matched_count": rr_matched, "roll_forward_rate": rr_fwd,
        "roll_backward_rate": rr_bwd, "npa_formation_rate": rr_formation,
    }
    has_prev = len(df_p) > 0
    pulse_kpis              = compute_pulse_kpis(df_c, df_p)
    fig_waterfall           = compute_bucket_waterfall(df_c, df_p)
    region_df               = compute_region_scorecard(df_c, df_p)
    branch_df, fig_quadrant = compute_branch_quadrant(df_c)
    exec_recovery_df        = compute_executive_recovery(df_c)
    product_data            = compute_product_analysis(df_c)
    risk_indicators         = compute_risk_indicators(df_c, df_p, rr_meta_local if rr_matched > 0 else None)
    exec_df_for_gb          = exec_recovery_df
    good_bad                = compute_good_bad(region_df, branch_df, risk_indicators, exec_df_for_gb, has_prev)
    fig_treemap             = compute_concentration_treemap(df_c)
    fleet                   = compute_fleet_exposure(df_c)
    top_accounts, top_accounts_summary = compute_top_accounts(df_c)
    repo_df                 = compute_repossession_list(df_c)
    npa_sma2_cmp            = compute_npa_sma2_comparison(df_c, df_p)
    good_customers          = compute_good_customers(df_c)
    return (
        pulse_kpis, fig_waterfall,
        region_df, branch_df, fig_quadrant,
        exec_recovery_df, product_data, risk_indicators, good_bad,
        fig_treemap, fleet, top_accounts, top_accounts_summary, repo_df, npa_sma2_cmp, good_customers,
    )

# ── Apply filters (cached  -  no pandas work on same filter rerun) ──────────────
df_curr, df_prev = _cached_filter(df_curr_raw, df_prev_raw, sel_region, sel_branch, sel_status, tuple(sel_segment))

# Clear AI/report results when filters change
_filter_key = f"{sel_region}|{sel_branch}|{sel_status}|{','.join(sorted(sel_segment))}"
if st.session_state.get("_last_filter_key") != _filter_key:
    st.session_state.pop("ai_result", None)
    st.session_state.pop("report_result", None)
    st.session_state["_last_filter_key"] = _filter_key

if len(df_curr) == 0:
    st.warning("No data matches the selected filters.")
    st.stop()

# ── Pre-compute shared data ───────────────────────────────────────────────────
metrics = _cached_metrics(df_curr, df_prev)
alerts  = _cached_alerts(df_curr)

scorecard_df = None
if "MNT NAME" in df_curr.columns:
    scorecard_df = _cached_scorecard(df_curr)

rr_matrix, rr_meta = None, None
if len(df_prev_raw) > 0:
    rr_matrix, rr_meta = _cached_roll_rate(df_curr, df_prev)

alerts_prev = _cached_alerts(df_prev) if len(df_prev) > 0 else []

_rr = rr_meta or {}
(
    pi_pulse_kpis, pi_fig_wf,
    pi_region, pi_branch, pi_fig_quad,
    pi_exec, pi_product, pi_risk, pi_good_bad,
    pi_fig_treemap, pi_fleet, pi_top_accounts, pi_top_accounts_summary, pi_repo_df, pi_npa_sma2_cmp, pi_good_customers,
) = _cached_portfolio_intel(
    df_curr, df_prev,
    int(_rr.get("matched_count", 0)),
    float(_rr.get("roll_forward_rate", 0.0)),
    float(_rr.get("roll_backward_rate", 0.0)),
    float(_rr.get("npa_formation_rate", 0.0)),
    tuple((a["count"], a["title"]) for a in alerts),
    tuple((a["count"], a["title"]) for a in alerts_prev),
)

# ── Active filter bar ─────────────────────────────────────────────────────────
active_filters = {k: v for k, v in {
    "Region": sel_region, "Branch": sel_branch, "Loan Status": sel_status,
    "Segment": ", ".join(sel_segment) if sel_segment else "All",
}.items() if v != "All"}
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

tabs = st.tabs(["🗂️ Dashboard", "👤 Scorecard", alert_label, "📈 Migration", "📊 Portfolio Intelligence", "🤖 AI Query", "📋 Report"])


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
        _pi_flag_df = compute_risk_flag_comparison(alerts, alerts_prev)
        render_portfolio_intelligence_tab(
            pulse_kpis=pi_pulse_kpis,
            fig_waterfall=pi_fig_wf,
            region_df=pi_region,
            branch_df=pi_branch,
            fig_quadrant=pi_fig_quad,
            exec_recovery_df=pi_exec,
            product_data=pi_product,
            risk_indicators=pi_risk,
            good_bad=pi_good_bad,
            flag_df=_pi_flag_df,
            fig_treemap=pi_fig_treemap,
            fleet=pi_fleet,
            top_accounts=pi_top_accounts,
            top_accounts_summary=pi_top_accounts_summary,
            has_prev=len(df_prev) > 0,
            rr_meta=rr_meta,
            repo_df=pi_repo_df,
            npa_sma2_cmp=pi_npa_sma2_cmp,
            good_customers=pi_good_customers,
        )
    except Exception as _e:
        _tab_error("Portfolio Intelligence", _e)

with tabs[5]:
    try:
        # Map uploaded files to their dated bucket columns so the AI can resolve
        # date references ("on 20th June") to curr_bucket / prev_bucket.
        _snapshot_dates = {"curr": curr_month_input.strftime("%Y-%m-%d")}
        if prev_month and len(df_prev_raw) > 0:
            _snapshot_dates["prev"] = prev_month_input.strftime("%Y-%m-%d")
        render_ai_query_tab(df_curr, _snapshot_dates)
    except Exception as _e:
        _tab_error("AI Query", _e)

with tabs[6]:
    try:
        render_report_tab(
            df_curr, df_prev, curr_month, prev_month,
            sel_region, sel_branch, sel_status,
            scorecard_df, rr_meta,
        )
    except Exception as _e:
        _tab_error("Report", _e)
