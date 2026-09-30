import datetime
import os

from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import pandas as pd

from utils import (
    apply_filters, compute_metrics, PREV_CARRYOVER_COLS, count_agreed_after_month,
    build_status_bar_chart, build_branch_bar_chart, build_closing_pc_chart,
)
from smart_alerts import run_all_alerts

from ui.styles import inject_styles
from ui.header import render_header
from ui.landing import render_landing
from ui.sidebar import render_sidebar
from ui.components import _load_and_concat, _bump_data_version, _file_fingerprint
from analysis.executive_scorecard import compute_executive_scorecard
from analysis.roll_rate import compute_roll_rate_matrix
from ui.tabs.dashboard import render_dashboard_tab
from ui.tabs.scorecard import render_scorecard_tab
from ui.tabs.alerts import render_alerts_tab
from ui.tabs.migration import render_migration_tab
from ui.tabs.portfolio_intelligence import render_portfolio_intelligence_tab
from ui.tabs.business import render_business_tab
from analysis.portfolio_intelligence import (
    compute_pulse_kpis, compute_bucket_waterfall,
    compute_region_scorecard, compute_branch_quadrant,
    compute_executive_recovery, compute_product_analysis,
    compute_risk_indicators, compute_good_bad,
    compute_concentration_treemap, compute_fleet_exposure,
    compute_top_accounts, compute_repossession_list,
    compute_risk_flag_comparison, compute_npa_sma2_comparison,
    compute_good_customers, compute_overdue_demand_scorecard,
    compute_new_advances, compute_new_advances_by_dimension,
)
from ui.tabs.ai_query import render_ai_query_tab
from ui.tabs.report import render_report_tab
from ui.tabs.investigator import render_investigator_tab
from ui.tabs.root_cause import render_root_cause_tab
from analysis.root_cause import (
    clean_contaminated_flags, compute_insurance_split,
    compute_chronic_shock_split, compute_region_why_table,
    compute_recent_advances_summary, compute_recent_advances_bucket_summary,
    compute_recent_advances_bucket_by_group,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CollectionIQ",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_styles()
render_header()

# ── File upload section ───────────────────────────────────────────────────────
# Pre-populate date pickers for sample data
if st.session_state.pop("_set_sample_dates", False):
    st.session_state["curr_month_pick"] = datetime.date(2026, 8, 1)   # generate_demo_data.py's months
    st.session_state["prev_month_pick"] = datetime.date(2026, 7, 1)

st.markdown('<div class="section-label">Data Source</div>', unsafe_allow_html=True)
col_up1, col_up2 = st.columns(2)

with col_up1:
    st.markdown(
        '<div class="upload-card">'
        '<div class="upload-card-title">📂 Current Month</div>'
        '<div class="upload-card-sub">Required: upload one or multiple regional files</div>',
        unsafe_allow_html=True,
    )
    curr_file = st.file_uploader(
        "Current Month", type=["xlsx", "xls", "xlsb"],
        key="curr", label_visibility="collapsed", accept_multiple_files=True,
    )
    curr_month_input = st.date_input(
        "Reporting Month", value=datetime.date.today().replace(day=1),
        key="curr_month_pick",
        help="Select any date in the reporting month: only Month & Year are used",
        format="DD/MM/YYYY",
    )
    st.markdown("</div>", unsafe_allow_html=True)

with col_up2:
    st.markdown(
        '<div class="upload-card">'
        '<div class="upload-card-title">📂 Previous Month</div>'
        '<div class="upload-card-sub">Optional: upload one or multiple regional files</div>',
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
        help="Select any date in the previous month: only Month & Year are used",
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
    # Region/Branch/Status selectboxes in ui/sidebar.py persist their selected
    # VALUE across reruns by widget key -- a value that happens to also exist
    # in the newly-uploaded file's own Region/Branch/Status list (common
    # across LCC extracts from the same NBFC) silently carries over and can
    # filter the new data down to an unintended/near-empty slice, tripping
    # "No data matches the selected filters" (looks like an error) instead of
    # defaulting to "All". Popping the widget keys here forces every filter
    # back to "All" on every fresh upload, not just when the old value
    # happens to be absent from the new file.
    for _k in ["df_curr_raw", "df_prev_raw", "ai_result", "report_result", "_last_filter_key",
               "_sample_loaded", "_sel_branch", "_prev_region", "sel_region_key", "sel_status_key",
               "sel_date_from_key"]:
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
    _bump_data_version(f"{_file_fingerprint(curr_file)}-{_file_fingerprint(prev_file)}")
    st.rerun()

if "df_curr_raw" not in st.session_state:
    st.markdown(
        '<div style="text-align:center;padding:20px 0;color:#aaa;font-size:13px;">'
        'File ready: click <strong>Generate Dashboard</strong> to build the report.'
        '</div>',
        unsafe_allow_html=True,
    )
    st.stop()

df_curr_raw: pd.DataFrame = st.session_state["df_curr_raw"]
df_prev_raw: pd.DataFrame = st.session_state["df_prev_raw"]

# Data-quality checks on the upload, gathered into ONE collapsed panel instead
# of a stack of banners above the tabs. Warnings (things that can distort
# numbers) are listed before notes (informational), and the panel's own label
# carries the counts, so nothing is hidden -- just not in the way.
_dq_warnings: list[str] = []
_dq_notes: list[str] = []
for _label, _df, _month in (("Current month", df_curr_raw, curr_month), ("Previous month", df_prev_raw, prev_month)):
    # A date column where many values failed to parse can distort vintage
    # cohorts, repossession windows and "paid this month" filters -- see
    # utils.py::_parse_date_column.
    for _w in _df.attrs.get("date_parse_warnings", []):
        _dq_warnings.append(f"**{_label}:** {_w}")
    _n_future = count_agreed_after_month(_df, _month)
    if _n_future:
        _dq_warnings.append(
            f"**{_label}:** {_n_future:,} loan(s) are dated after the reporting month "
            f"({pd.Timestamp(_month):%b %Y}), usually a day/month swap in the source file "
            "(6 Dec entered as 12 Jun). Kept as-is, but they may land in the wrong new-advances/vintage month."
        )
    # A real extract shouldn't have duplicate Loan Nos; they're dropped, so say how many.
    if _df.attrs.get("dropped_duplicate_loans", 0):
        _dq_notes.append(f"**{_label}:** removed {_df.attrs['dropped_duplicate_loans']:,} duplicate Loan No row(s).")
    # Source-file faults repaired at load (utils.repair_upload): say what changed.
    _dq_notes += [f"**{_label}:** {n}" for n in _df.attrs.get("data_fixes", [])]
# Missing optional columns (e.g. CoLending_Loans) make "no co-lending accounts"
# indistinguishable from "that column isn't in this file" -- so name them.
_missing_cols = set(df_curr_raw.attrs.get("missing_optional_cols", [])) | set(df_prev_raw.attrs.get("missing_optional_cols", []))
if _missing_cols:
    _dq_notes.append(
        f"Column(s) not found in this upload: {', '.join(sorted(_missing_cols))}. "
        "Anything that uses them may show no results, which is not an error."
    )
if _dq_warnings or _dq_notes:
    _parts = [f"{len(_dq_warnings)} warning{'s' * (len(_dq_warnings) != 1)}"] if _dq_warnings else []
    _parts += [f"{len(_dq_notes)} note{'s' * (len(_dq_notes) != 1)}"] if _dq_notes else []
    with st.expander(f"{'⚠️' if _dq_warnings else 'ℹ️'} Data checks: {', '.join(_parts)}", expanded=False):
        st.markdown("\n".join([f"- ⚠️ {w}" for w in _dq_warnings] + [f"- ℹ️ {n}" for n in _dq_notes]))

# Auto-load prev if uploaded after initial generate. _load_and_concat is NOT
# cached, so a file that fails to parse must not be re-parsed on every rerun:
# remember the failure per file fingerprint and just re-show its error.
_prev_autoload_fail = st.session_state.get("_prev_autoload_failure")  # (fingerprint, message) or None
if prev_file and len(df_prev_raw) == 0:
    _prev_fp = _file_fingerprint(prev_file)
    if _prev_autoload_fail and _prev_autoload_fail[0] == _prev_fp:
        _prev_tmp = None
        st.warning(f"Previous month file could not be loaded: {_prev_autoload_fail[1]}")
    else:
        _prev_tmp, _prev_err = _load_and_concat(prev_file)
        if _prev_tmp is None:
            _msg = _prev_err[0] if _prev_err else "unknown error"
            st.session_state["_prev_autoload_failure"] = (_prev_fp, _msg)
            st.warning(f"Previous month file could not be loaded: {_msg}")
    if _prev_tmp is not None:
        df_prev_raw = _prev_tmp
        # Previously only rebound the local variable -- session_state stayed
        # stale, and (before this optimization pass) that "worked" only
        # because every cache below re-hashed the full DataFrame content on
        # every rerun. Once those caches key on _data_version instead, this
        # path MUST also persist the fresh frame and bump the version, or
        # every downstream cached function would silently keep serving the
        # "no previous file" result forever after this point.
        st.session_state["df_prev_raw"] = df_prev_raw
        _bump_data_version(f"{_file_fingerprint(curr_file)}-{_file_fingerprint(prev_file)}")

# ── Guard against rendering a stale dashboard after a file swap ─────────────
# Streamlit reruns the WHOLE script on every interaction, including simply
# picking a different file in the uploader widgets -- with no Generate click
# involved. Session state (df_curr_raw/df_prev_raw) only changes on an actual
# Generate click (or the deliberate prev-auto-load exception just above), so
# unguarded, swapping in a brand-new file (or removing one) rendered the FULL
# dashboard straight from the OLD session data with zero indication anything
# was stale. Confirmed: generate on Region 1 curr+prev, then swap in a
# completely different Zone 1 curr-only file without clicking Generate again
# -- the Region 1 dashboard kept rendering as if nothing had changed. NOT a
# cross-user issue (st.session_state is isolated per browser session; the
# shared st.cache_data layer is keyed on file-content SHA-256 via
# _file_fingerprint, so two users' different files can never collide) -- but
# within one session it's a real risk of acting on stale numbers believing
# they're the just-uploaded file. Skipped for the sample-data path, which
# never touches curr_file/prev_file at all (its own _bump_data_version() call
# uses the per-session-counter fallback, not a file fingerprint).
if not st.session_state.get("_sample_loaded"):
    _live_fingerprint = f"{_file_fingerprint(curr_file)}-{_file_fingerprint(prev_file)}"
    if _live_fingerprint != st.session_state.get("_data_version"):
        st.markdown(
            '<div style="text-align:center;padding:20px 0;color:#aaa;font-size:13px;">'
            'New file(s) selected: click <strong>Generate Dashboard</strong> to load them.'
            '</div>',
            unsafe_allow_html=True,
        )
        st.stop()

# ── Sidebar filters ───────────────────────────────────────────────────────────
sel_region, sel_branch, sel_status, sel_segment, sel_date_from = render_sidebar(df_curr_raw, curr_month)

# data_version is the cheap proxy for "has the underlying raw data changed" --
# read once here, threaded explicitly into every cache-wrapped call below so
# each one can accept its DataFrame args as underscore-prefixed (never hashed
# by Streamlit) instead of paying to hash the full extract on every rerun.
data_version = st.session_state.get("_data_version", 0)

# ── Cached computation wrappers (module-level  -  registered once, not per rerun) ──
# Every wrapper sets max_entries: st.cache_data's store is global to the server
# process and unbounded by default, so without a limit each (data_version x
# filter-combination) key accumulates its full result forever -- _cached_filter
# alone holds two full DataFrame copies per entry, which on a shared deployment
# grows until the process OOMs and restarts for every user. Limits are sized to
# comfortably cover one session's realistic filter-browsing (eviction is LRU,
# so an evicted combo just recomputes -- correctness is never affected).
@st.cache_data(show_spinner=False, max_entries=16)
def _cached_filter(_df_c: pd.DataFrame, _df_p_raw: pd.DataFrame, data_version: int, region: str, branch: str, status: str, segment: tuple = (), date_from=None):
    df = apply_filters(_df_c.copy(), region, branch, status, segment, date_from)
    df_p = apply_filters(_df_p_raw.copy(), region, branch, status, segment, date_from)
    if len(_df_p_raw) > 0 and "Loan No" in df.columns and "curr_bucket" in _df_p_raw.columns:
        # Carry over the prev-month bucket plus a curated set of numeric columns
        # (renamed prev_*) so the AI can compute month-over-month reductions.
        carry = {"curr_bucket": "prev_bucket"}
        carry.update({src: dst for src, dst in PREV_CARRYOVER_COLS.items() if src in _df_p_raw.columns})
        slim = _df_p_raw[["Loan No", *carry.keys()]].rename(columns=carry)
        df = df.merge(slim, on="Loan No", how="left")
    return df, df_p

# The 5 functions below all receive the already-FILTERED df_curr/df_prev (from
# _cached_filter's output above), which vary per filter combination even when
# data_version (the raw-data proxy) is unchanged. Since the DataFrame args are
# underscore-prefixed (never hashed), the filter tuple -- already in scope at
# each call site -- is the only remaining signal that distinguishes e.g.
# "Region=Pune" from "Region=Mumbai" in the cache key.
@st.cache_data(show_spinner=False, max_entries=32)
def _cached_metrics(_df_c: pd.DataFrame, _df_p: pd.DataFrame, data_version: int, region: str, branch: str, status: str, segment: tuple, date_from=None):
    return compute_metrics(_df_c, _df_p)

@st.cache_data(show_spinner=False, max_entries=32)
def _cached_dashboard_charts(_df_c: pd.DataFrame, data_version: int, region: str, branch: str, status: str, segment: tuple, date_from=None):
    # These 3 chart builders used to run uncached directly inside
    # ui/tabs/dashboard.py's render function -- since Dashboard is tabs[0]
    # and Streamlit executes every tab's code on every rerun regardless of
    # which tab is visible, that meant rebuilding all 3 Plotly figures on
    # every single interaction anywhere in the app, not just when df_curr
    # actually changed. Same cached-recompute pattern as every other
    # _cached_* wrapper here.
    return (
        build_status_bar_chart(_df_c),
        build_branch_bar_chart(_df_c),
        build_closing_pc_chart(_df_c),
    )

@st.cache_data(show_spinner=False, max_entries=32)
def _cached_alerts(_df_c: pd.DataFrame, data_version: int, region: str, branch: str, status: str, segment: tuple, which: str, as_of: str = None, date_from=None):
    # as_of anchors alert_recent_advances_at_risk's "last N months" window to
    # the FILE's own reporting month, not wall-clock today -- see that
    # function's own docstring. "which" ("curr"/"prev") already distinguishes
    # this cache entry from its counterpart; as_of naturally participates in
    # the cache key too (a different reporting month is a genuine cache miss).
    return run_all_alerts(_df_c, as_of=as_of)

@st.cache_data(show_spinner=False, max_entries=32)
def _cached_scorecard(_df_c: pd.DataFrame, data_version: int, region: str, branch: str, status: str, segment: tuple, date_from=None):
    return compute_executive_scorecard(_df_c)

@st.cache_data(show_spinner=False, max_entries=32)
def _cached_roll_rate(_df_c: pd.DataFrame, _df_p: pd.DataFrame, data_version: int, region: str, branch: str, status: str, segment: tuple, date_from=None):
    return compute_roll_rate_matrix(_df_c, _df_p)

@st.cache_data(show_spinner=False, max_entries=16)
def _cached_portfolio_intel(
    _df_c: pd.DataFrame, _df_p: pd.DataFrame,
    data_version: int, region: str, branch: str, status: str, segment: tuple, date_from,
    rr_matched: int, rr_fwd: float, rr_bwd: float, rr_formation: float,
    alerts_curr_counts: tuple, alerts_prev_counts: tuple,
    curr_month: str,
):
    df_c, df_p = _df_c, _df_p
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
    product_data            = compute_product_analysis(df_c, as_of=curr_month)
    risk_indicators         = compute_risk_indicators(df_c, df_p, rr_meta_local if rr_matched > 0 else None)
    exec_df_for_gb          = exec_recovery_df
    good_bad                = compute_good_bad(region_df, branch_df, risk_indicators, exec_df_for_gb, has_prev)
    fig_treemap             = compute_concentration_treemap(df_c)
    fleet                   = compute_fleet_exposure(df_c)
    top_accounts, top_accounts_summary = compute_top_accounts(df_c)
    repo_df                 = compute_repossession_list(df_c, as_of=curr_month)
    npa_sma2_cmp            = compute_npa_sma2_comparison(df_c, df_p)
    good_customers          = compute_good_customers(df_c)
    overdue_demand_scorecard = compute_overdue_demand_scorecard(df_c)
    new_advances            = compute_new_advances(df_c, as_of=curr_month)
    new_advances_by_dim     = compute_new_advances_by_dimension(df_c, as_of=curr_month)
    return (
        pulse_kpis, fig_waterfall,
        region_df, branch_df, fig_quadrant,
        exec_recovery_df, product_data, risk_indicators, good_bad,
        fig_treemap, fleet, top_accounts, top_accounts_summary, repo_df, npa_sma2_cmp, good_customers,
        overdue_demand_scorecard, new_advances, new_advances_by_dim,
    )

@st.cache_data(show_spinner=False, max_entries=16)
def _cached_root_cause(
    _df_c: pd.DataFrame, _region_df: pd.DataFrame,
    data_version: int, region: str, branch: str, status: str, segment: tuple, date_from,
    curr_month: str,
):
    # region_df comes from _cached_portfolio_intel (compute_region_scorecard) --
    # reused, not recomputed, so Root Cause's NPA%/Δ NPA%/Status/Collection%
    # figures are numerically identical to what Portfolio Intelligence already
    # shows for the same region, never a second independent calculation.
    df_clean, contamination = clean_contaminated_flags(_df_c)
    insurance_split = compute_insurance_split(df_clean, group_col="Unit")
    chronic_shock = compute_chronic_shock_split(df_clean, group_col="Unit")
    why_df = compute_region_why_table(df_clean, _region_df, as_of=curr_month)
    recent_summary = compute_recent_advances_summary(df_clean)
    recent_bucket = compute_recent_advances_bucket_summary(df_clean)
    recent_by_region_count, recent_by_region_soh = compute_recent_advances_bucket_by_group(df_clean, group_col="RegionName")
    recent_by_branch_count, recent_by_branch_soh = compute_recent_advances_bucket_by_group(df_clean, group_col="Unit")
    return (
        why_df, insurance_split, chronic_shock, contamination,
        recent_summary, recent_bucket,
        recent_by_region_count, recent_by_region_soh,
        recent_by_branch_count, recent_by_branch_soh,
    )

# ── Apply filters (cached  -  no pandas work on same filter rerun) ──────────────
_seg_t = tuple(sel_segment)
df_curr, df_prev = _cached_filter(df_curr_raw, df_prev_raw, data_version, sel_region, sel_branch, sel_status, _seg_t, sel_date_from)

# Clear AI/report results when filters change
_filter_key = f"{sel_region}|{sel_branch}|{sel_status}|{','.join(sorted(sel_segment))}|{sel_date_from}"
if st.session_state.get("_last_filter_key") != _filter_key:
    st.session_state.pop("ai_result", None)
    st.session_state.pop("report_result", None)
    st.session_state.pop("investigator_threads", None)
    st.session_state.pop("investigator_active_thread", None)
    st.session_state["_last_filter_key"] = _filter_key

if len(df_curr) == 0:
    st.warning("No data matches the selected filters.")
    st.stop()

# ── Pre-compute shared data ───────────────────────────────────────────────────
metrics = _cached_metrics(df_curr, df_prev, data_version, sel_region, sel_branch, sel_status, _seg_t, sel_date_from)
alerts  = _cached_alerts(df_curr, data_version, sel_region, sel_branch, sel_status, _seg_t, "curr", curr_month, sel_date_from)
fig_status, fig_branch, fig_closing = _cached_dashboard_charts(
    df_curr, data_version, sel_region, sel_branch, sel_status, _seg_t, sel_date_from,
)

scorecard_df = None
if "MNT NAME" in df_curr.columns:
    scorecard_df = _cached_scorecard(df_curr, data_version, sel_region, sel_branch, sel_status, _seg_t, sel_date_from)

rr_matrix, rr_meta = None, None
if len(df_prev_raw) > 0:
    rr_matrix, rr_meta = _cached_roll_rate(df_curr, df_prev, data_version, sel_region, sel_branch, sel_status, _seg_t, sel_date_from)

# ── Active filter bar ─────────────────────────────────────────────────────────
active_filters = {k: v for k, v in {
    "Region": sel_region, "Branch": sel_branch, "Loan Status": sel_status,
    "Segment": ", ".join(sel_segment) if sel_segment else "All",
    "Loan Date": f"On/after {sel_date_from}" if sel_date_from else "All",
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
# A manual segmented-control switcher, NOT st.tabs(). st.tabs() renders every
# tab's UI on every rerun (only CSS-hides inactive ones), which was both a
# real latency cost (rendering 8 tabs' worth of widgets/charts for the 1 the
# user can see) and the root cause of a real, well-documented Streamlit
# framework bug where the JS that hides inactive tab panels desyncs -- often
# triggered by spinners/st.status() inside tabs (this app's AI Query and
# Report tabs both use them) -- and every tab's content renders stacked as
# one long scrollable page instead of switching. Gating each render_*_tab()
# call behind the active selection (instead of `with tabs[i]:`) fixes both:
# only the active section's UI ever gets built, and there's no st.tabs()
# panel-hiding mechanism left to desync. The analysis/ computation above this
# block (metrics, alerts, dashboard charts, scorecard, roll-rate) stays
# unconditional/cache-driven, since Dashboard (the default landing tab)
# needs all of it -- but the heavier Portfolio Intelligence block below
# (~15 analysis/ functions incl. 3 Plotly chart builders) IS gated by tab
# visibility now; see its own comment further down for why.
n_alerts = sum(1 for a in alerts if a["count"] > 0)
# NOTE: unlike the old st.tabs() label, this stays a fixed string ("🚨
# Alerts") rather than embedding the live count -- st.segmented_control's
# selection is stored in st.session_state by option VALUE, so a label that
# changes text across reruns (count going from e.g. 3 to 5) would no longer
# match the previously-selected option string and silently deselect it.
# The count is shown as a caption next to the selector instead.

_TAB_LABELS = ["🗂️ Dashboard", "👤 Scorecard", "🚨 Alerts", "📈 Migration", "📊 Portfolio Intelligence", "🔎 Root Cause", "💼 Business", "🤖 AI Query", "🕵️ Investigator", "📋 Report"]

active = st.segmented_control(
    "Section", options=_TAB_LABELS, default=_TAB_LABELS[0], key="_active_section", label_visibility="collapsed",
)
if active is None:
    # Single-select segmented_control returns None if the active pill is
    # clicked again (deselect) -- fall back to whichever tab was active
    # last, not all the way back to Dashboard.
    active = st.session_state.get("_last_active_section", _TAB_LABELS[0])
st.session_state["_last_active_section"] = active
if active == "🚨 Alerts" and n_alerts:
    st.caption(f"🚨 {n_alerts} alert{'s' if n_alerts > 1 else ''} active")

# ── Portfolio Intelligence pre-compute (lazy, tab-gated) ────────────────────
# _cached_portfolio_intel runs ~15 analysis/ functions incl. 3 Plotly chart
# builders (bucket waterfall, branch quadrant, concentration treemap) over the
# WHOLE current-filter DataFrame -- the single most expensive block in this
# app. It used to run unconditionally, above the tab switch, on every single
# rerun regardless of which tab was visible -- so Dashboard (the default
# landing tab, and every user's first paint after upload) paid this cost even
# though it reads none of the output. st.cache_data still caches the result
# under the same key as before, so switching INTO one of the 3 tabs that
# actually need it pays the cost once, exactly as before -- it just no longer
# blocks the other 5 tabs (Dashboard, Scorecard, Alerts, Migration, Report).
# Safe to skip precomputed_views entirely on those other tabs: the AI Query
# fast-path view layer (registry/views.py / graph.py::view_node) already
# falls back to calling the analysis/ function fresh whenever a view's
# cache_key isn't present in precomputed_views (e.g. a filtered/non-default
# query), so an empty dict here is a correctness no-op, not a missing case.
_needs_pi = active in ("📊 Portfolio Intelligence", "🔎 Root Cause", "💼 Business", "🤖 AI Query")

alerts_prev = []
precomputed_views = {}
if _needs_pi:
    alerts_prev = _cached_alerts(df_prev, data_version, sel_region, sel_branch, sel_status, _seg_t, "prev", prev_month, sel_date_from) if len(df_prev) > 0 else []

    _rr = rr_meta or {}
    (
        pi_pulse_kpis, pi_fig_wf,
        pi_region, pi_branch, pi_fig_quad,
        pi_exec, pi_product, pi_risk, pi_good_bad,
        pi_fig_treemap, pi_fleet, pi_top_accounts, pi_top_accounts_summary, pi_repo_df, pi_npa_sma2_cmp, pi_good_customers,
        pi_overdue_demand, pi_new_advances, pi_new_advances_by_dim,
    ) = _cached_portfolio_intel(
        df_curr, df_prev,
        data_version, sel_region, sel_branch, sel_status, _seg_t, sel_date_from,
        int(_rr.get("matched_count", 0)),
        float(_rr.get("roll_forward_rate", 0.0)),
        float(_rr.get("roll_backward_rate", 0.0)),
        float(_rr.get("npa_formation_rate", 0.0)),
        tuple((a["count"], a["title"]) for a in alerts),
        tuple((a["count"], a["title"]) for a in alerts_prev),
        curr_month,
    )

    # Precomputed analysis/ results, keyed for the AI Query tab's fast-path view
    # layer (registry/views.py) to reuse directly -- guarantees AI Query answers are
    # numerically identical to what these same tabs already show, no recomputation.
    precomputed_views = {
        "pi_top_accounts":    (pi_top_accounts, pi_top_accounts_summary),
        "pi_fleet":           pi_fleet,
        "scorecard_df":       scorecard_df,
        # Wrapped as tuples to match what a FRESH call to compute_roll_rate_matrix /
        # compute_branch_quadrant returns -- registry/views.py normalizers handle
        # the cached and freshly-computed cases identically this way.
        "rr_matrix":          (rr_matrix, rr_meta),
        "pi_region":          pi_region,
        "pi_branch":          (pi_branch, pi_fig_quad),
        "pi_exec":            pi_exec,
        "pi_risk":            pi_risk,
        "pi_repo_df":         pi_repo_df,
        "pi_good_customers":  pi_good_customers,
        "pi_npa_sma2_cmp":    pi_npa_sma2_cmp,
        "pi_product":         pi_product,
        "pi_pulse_kpis":      pi_pulse_kpis,
        "pi_good_bad":        pi_good_bad,
        "pi_overdue_demand":  pi_overdue_demand,
        "pi_new_advances":    pi_new_advances,
        "pi_new_advances_by_dim": pi_new_advances_by_dim,
    }

    # Root Cause reuses pi_region (compute_region_scorecard) rather than a
    # second independent call -- only actually computed when that tab is the
    # one active, not on every Business/AI Query rerun that also needs pi_region.
    if active == "🔎 Root Cause":
        (
            rc_why_df, rc_insurance_split, rc_chronic_shock, rc_contamination,
            rc_recent_summary, rc_recent_bucket,
            rc_recent_by_region_count, rc_recent_by_region_soh,
            rc_recent_by_branch_count, rc_recent_by_branch_soh,
        ) = _cached_root_cause(
            df_curr, pi_region, data_version, sel_region, sel_branch, sel_status, _seg_t, sel_date_from,
            curr_month,
        )


def _tab_error(name: str, exc: Exception) -> None:
    st.error(f"**{name} tab failed to render:** {exc}")
    st.caption("Try clearing the cache from the sidebar, or check your data file.")


if active == "🗂️ Dashboard":
    try:
        render_dashboard_tab(
            df_curr, df_prev, metrics, curr_month,
            sel_region, sel_branch, sel_status,
            alerts, scorecard_df, rr_meta,
            fig_status, fig_branch, fig_closing,
            data_version=data_version, segment=_seg_t,
        )
    except Exception as _e:
        _tab_error("Dashboard", _e)

elif active == "👤 Scorecard":
    try:
        render_scorecard_tab(df_curr, scorecard_df)
    except Exception as _e:
        _tab_error("Scorecard", _e)

elif active == "🚨 Alerts":
    try:
        render_alerts_tab(df_curr, alerts)
    except Exception as _e:
        _tab_error("Alerts", _e)

elif active == "📈 Migration":
    try:
        render_migration_tab(df_curr, df_prev, rr_matrix, rr_meta, data_version, _filter_key)
    except Exception as _e:
        _tab_error("Migration", _e)

elif active == "📊 Portfolio Intelligence":
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
            overdue_demand_scorecard=pi_overdue_demand,
            df_curr=df_curr,
        )
    except Exception as _e:
        _tab_error("Portfolio Intelligence", _e)

elif active == "🔎 Root Cause":
    try:
        render_root_cause_tab(
            why_df=rc_why_df,
            insurance_split_df=rc_insurance_split,
            chronic_shock_df=rc_chronic_shock,
            contamination=rc_contamination,
            curr_month=curr_month,
            df_curr=df_curr,
            df_all=df_curr_raw,
            recent_summary=rc_recent_summary,
            recent_bucket_df=rc_recent_bucket,
            recent_by_region_count=rc_recent_by_region_count,
            recent_by_region_soh=rc_recent_by_region_soh,
            recent_by_branch_count=rc_recent_by_branch_count,
            recent_by_branch_soh=rc_recent_by_branch_soh,
        )
    except Exception as _e:
        _tab_error("Root Cause", _e)

elif active == "💼 Business":
    try:
        render_business_tab(
            new_advances=pi_new_advances,
            df_curr=df_curr,
            curr_month=curr_month,
            dimension_data=pi_new_advances_by_dim,
            vintage_df=pi_product.get("vintage", pd.DataFrame()),
            data_version=data_version, region=sel_region, branch=sel_branch, status=sel_status, segment=_seg_t,
            date_from=sel_date_from,
        )
    except Exception as _e:
        _tab_error("Business", _e)

elif active == "🤖 AI Query":
    try:
        # Map uploaded files to their dated bucket columns so the AI can resolve
        # date references ("on 20th June") to curr_bucket / prev_bucket.
        _snapshot_dates = {"curr": curr_month_input.strftime("%Y-%m-%d")}
        if prev_month and len(df_prev_raw) > 0:
            _snapshot_dates["prev"] = prev_month_input.strftime("%Y-%m-%d")
        render_ai_query_tab(
            df_curr, _snapshot_dates, df_prev=df_prev, precomputed_views=precomputed_views,
            alerts_curr=alerts, alerts_prev=alerts_prev, rr_meta=rr_meta,
            data_version=data_version, filter_key=_filter_key,
        )
    except Exception as _e:
        _tab_error("AI Query", _e)

elif active == "🕵️ Investigator":
    try:
        render_investigator_tab(
            df_curr, df_prev,
            data_version=data_version, filter_key=_filter_key,
            alerts_curr=alerts, curr_month=curr_month,
        )
    except Exception as _e:
        _tab_error("Investigator", _e)

elif active == "📋 Report":
    try:
        render_report_tab(
            df_curr, df_prev, curr_month, prev_month,
            sel_region, sel_branch, sel_status,
            scorecard_df, rr_meta,
        )
    except Exception as _e:
        _tab_error("Report", _e)
