import datetime

from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import pandas as pd

from utils import apply_filters, compute_metrics, PREV_CARRYOVER_COLS, count_agreed_after_month
from smart_alerts import run_all_alerts

from ui.styles import inject_styles
from ui.header import render_header
from ui.landing import render_landing
from ui.sidebar import render_sidebar
from ui.components import _load_and_concat, _bump_data_version, _file_fingerprint
from analysis.executive_scorecard import compute_executive_scorecard
from analysis.roll_rate import compute_roll_rate_matrix
from analysis.portfolio_intelligence import compute_region_scorecard, compute_product_analysis
from analysis.new_business import compute_new_advances, compute_new_advances_by_dimension
from analysis.root_cause import (
    clean_contaminated_flags, compute_insurance_split,
    compute_chronic_shock_split, compute_region_why_table, add_unit_delinquency,
    compute_recent_advances_summary, compute_recent_advances_bucket_summary,
    compute_recent_advances_bucket_by_group,
)
from ui.tabs.dashboard import render_dashboard_tab
from ui.tabs.action_lists import render_action_lists_tab
from ui.tabs.migration import render_migration_tab
from ui.tabs.portfolio_intelligence import render_portfolio_intelligence_tab
from ui.tabs.root_cause import render_root_cause_tab
from ui.tabs.business import render_business_tab
from ui.tabs.ai_query import render_ai_query_tab
from ui.tabs.investigator import render_investigator_tab
from ui.tabs.report import render_report_tab

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
    for _k in ["df_curr_raw", "df_prev_raw", "ai_result", "rpt2", "rpt2_packs", "_last_filter_key",
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

# Cache keys: data_version stands for "which upload" and _filter_key for "which
# filters", so the cached functions below take their DataFrames underscore-
# prefixed (never hashed) and stay cheap to look up on every rerun.
data_version = st.session_state.get("_data_version", 0)
_seg_t = tuple(sel_segment)
_filter_key = f"{sel_region}|{sel_branch}|{sel_status}|{','.join(sorted(sel_segment))}|{sel_date_from}"


# ── Cached computations ───────────────────────────────────────────────────────
# Each runs only when a tab that shows it is open, and once per upload x filter
# combination. max_entries bounds the server-wide cache (it's shared by every
# session), so old combinations are evicted instead of piling up in memory.
@st.cache_data(show_spinner=False, max_entries=16)
def _cached_filter(_df_c: pd.DataFrame, _df_p_raw: pd.DataFrame, data_version: int,
                   region: str, branch: str, status: str, segment: tuple, date_from):
    # Filter first, then copy: copying the whole upload before filtering doubled memory for nothing.
    df = apply_filters(_df_c, region, branch, status, segment, date_from).copy()
    df_p = apply_filters(_df_p_raw, region, branch, status, segment, date_from).copy()
    if len(_df_p_raw) > 0 and "Loan No" in df.columns and "curr_bucket" in _df_p_raw.columns:
        # Last month's bucket and a few amounts (renamed prev_*) ride along on
        # each loan, for roll analysis and month-over-month changes.
        carry = {"curr_bucket": "prev_bucket"}
        carry.update({src: dst for src, dst in PREV_CARRYOVER_COLS.items() if src in _df_p_raw.columns})
        slim = _df_p_raw[["Loan No", *carry.keys()]].rename(columns=carry)
        df = df.merge(slim, on="Loan No", how="left")
    return df, df_p


@st.cache_data(show_spinner=False, max_entries=32)
def _cached_metrics(_df_c, _df_p, data_version: int, filter_key: str):
    return compute_metrics(_df_c, _df_p)


@st.cache_data(show_spinner=False, max_entries=32)
def _cached_alerts(_df, data_version: int, filter_key: str, which: str, as_of: str | None):
    # as_of: the file's reporting month, so "recent" windows don't follow today's date.
    return run_all_alerts(_df, as_of=as_of)


@st.cache_data(show_spinner=False, max_entries=32)
def _cached_scorecard(_df_c, _df_p, data_version: int, filter_key: str):
    return compute_executive_scorecard(_df_c, df_prev=_df_p)


@st.cache_data(show_spinner=False, max_entries=32)
def _cached_roll_rate(_df_c, _df_p, data_version: int, filter_key: str):
    return compute_roll_rate_matrix(_df_c, _df_p)


@st.cache_data(show_spinner=False, max_entries=16)
def _cached_action_lists(_df_c, _alerts_c, _alerts_p, data_version: int, filter_key: str, curr_month: str) -> list:
    from analysis.action_lists import build_lists
    return build_lists(_df_c, curr_month, _alerts_c, _alerts_p)


@st.cache_data(show_spinner=False, max_entries=16)
def _cached_business(_df_c, data_version: int, filter_key: str, curr_month: str):
    return (compute_new_advances(_df_c, as_of=curr_month),
            compute_new_advances_by_dimension(_df_c, as_of=curr_month),
            compute_product_analysis(_df_c, as_of=curr_month).get("vintage", pd.DataFrame()))


@st.cache_data(show_spinner=False, max_entries=16)
def _cached_root_cause(_df_c, _df_p, data_version: int, filter_key: str, curr_month: str) -> dict:
    df_clean, contamination = clean_contaminated_flags(_df_c)
    region_df = compute_region_scorecard(_df_c, _df_p)   # the same region figures Portfolio Intelligence shows
    by_region = compute_recent_advances_bucket_by_group(df_clean, group_col="RegionName")
    by_branch = compute_recent_advances_bucket_by_group(df_clean, group_col="Unit")
    return {
        "why_df": compute_region_why_table(df_clean, region_df, as_of=curr_month),
        "insurance_split_df": add_unit_delinquency(compute_insurance_split(df_clean, group_col="Unit"), _df_c, _df_p),
        "chronic_shock_df": add_unit_delinquency(compute_chronic_shock_split(df_clean, group_col="Unit"), _df_c, _df_p),
        "contamination": contamination,
        "recent_summary": compute_recent_advances_summary(df_clean),
        "recent_bucket_df": compute_recent_advances_bucket_summary(df_clean),
        "recent_by_region_count": by_region[0], "recent_by_region_soh": by_region[1],
        "recent_by_branch_count": by_branch[0], "recent_by_branch_soh": by_branch[1],
    }


df_curr, df_prev = _cached_filter(df_curr_raw, df_prev_raw, data_version, sel_region, sel_branch, sel_status,
                                   _seg_t, sel_date_from)

# Clear AI/report results when filters change
if st.session_state.get("_last_filter_key") != _filter_key:
    for _k in ("ai_result", "rpt2", "rpt2_packs", "investigator_threads", "investigator_active_thread"):
        st.session_state.pop(_k, None)
    st.session_state["_last_filter_key"] = _filter_key

if len(df_curr) == 0:
    st.warning("No data matches the selected filters.")
    st.stop()


def _alerts() -> list:
    return _cached_alerts(df_curr, data_version, _filter_key, "curr", curr_month)


def _alerts_prev() -> list:
    return _cached_alerts(df_prev, data_version, _filter_key, "prev", prev_month) if len(df_prev) else []


def _scorecard():
    return _cached_scorecard(df_curr, df_prev, data_version, _filter_key) if "MNT NAME" in df_curr.columns else None


def _roll_rate() -> tuple:
    """(matrix, meta), or (None, None) without last month's file."""
    return _cached_roll_rate(df_curr, df_prev, data_version, _filter_key) if len(df_prev_raw) else (None, None)


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
# A segmented control, not st.tabs(): st.tabs builds every tab on every rerun
# (and its panel hiding can desync, stacking all tabs on one page). Here only
# the open tab is built, and each tab computes only what it shows.
from ui.components import TAB_LABELS as _TAB_LABELS

active = st.segmented_control(
    "Section", options=_TAB_LABELS, default=_TAB_LABELS[0], key="_active_section", label_visibility="collapsed",
)
if active is None:
    # Clicking the open tab again deselects it: stay on the last tab.
    active = st.session_state.get("_last_active_section", _TAB_LABELS[0])
st.session_state["_last_active_section"] = active


def _tab_error(name: str, exc: Exception) -> None:
    st.error(f"**{name} tab failed to render:** {exc}")
    st.caption("Try clearing the cache from the sidebar, or check your data file.")


try:
    if active == "🗂️ Dashboard":
        render_dashboard_tab(
            df_curr, df_prev, _cached_metrics(df_curr, df_prev, data_version, _filter_key), curr_month,
            sel_region, sel_branch, sel_status, _alerts(),
            data_version=data_version, segment=_seg_t, date_from=sel_date_from, alerts_prev=_alerts_prev(),
        )

    elif active == "🎯 Action Lists":
        render_action_lists_tab(
            _cached_action_lists(df_curr, _alerts(), _alerts_prev(), data_version, _filter_key, curr_month), df_curr)

    elif active == "📈 Migration":
        render_migration_tab(df_curr, df_prev, *_roll_rate(), data_version, _filter_key)

    elif active == "📊 Portfolio Intelligence":
        render_portfolio_intelligence_tab(df_curr, df_prev, curr_month, data_version, _filter_key)

    elif active == "🔎 Root Cause":
        render_root_cause_tab(
            **_cached_root_cause(df_curr, df_prev, data_version, _filter_key, curr_month),
            curr_month=curr_month, df_curr=df_curr, df_all=df_curr_raw,
        )

    elif active == "💼 Business":
        new_advances, new_advances_by_dim, vintage_df = _cached_business(df_curr, data_version, _filter_key, curr_month)
        render_business_tab(
            new_advances=new_advances, df_curr=df_curr, curr_month=curr_month,
            dimension_data=new_advances_by_dim, vintage_df=vintage_df,
            data_version=data_version, filter_key=_filter_key,
        )

    elif active == "🤖 AI Query":
        # The reporting dates of each file, so the AI can resolve "on 20th June"
        # to this month's or last month's bucket.
        _snapshot_dates = {"curr": curr_month_input.strftime("%Y-%m-%d")}
        if prev_month and len(df_prev_raw) > 0:
            _snapshot_dates["prev"] = prev_month_input.strftime("%Y-%m-%d")
        rr_matrix, rr_meta = _roll_rate()
        # Views not listed here are computed on demand by the AI's view layer
        # (registry/views.py) with the same function, so the numbers match.
        precomputed_views = {"scorecard_df": _scorecard(), "rr_matrix": (rr_matrix, rr_meta)}
        render_ai_query_tab(
            df_curr, _snapshot_dates, df_prev=df_prev, precomputed_views=precomputed_views,
            alerts_curr=_alerts(), alerts_prev=_alerts_prev(), rr_meta=rr_meta,
            data_version=data_version, filter_key=_filter_key,
        )

    elif active == "🕵️ Investigator":
        render_investigator_tab(
            df_curr, df_prev, data_version=data_version, filter_key=_filter_key,
            alerts_curr=_alerts(), curr_month=curr_month,
        )

    elif active == "📋 Report":
        render_report_tab(df_curr, df_prev, curr_month, prev_month, sel_region, sel_branch, sel_status)

except Exception as _e:
    _tab_error(active.split(" ", 1)[-1], _e)
