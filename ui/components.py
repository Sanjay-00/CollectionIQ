"""Shared UI helpers used across multiple tab modules."""

import hashlib
import html
import re
from io import BytesIO

import pandas as pd
import streamlit as st

from utils import load_and_validate, REQUIRED_COLS, CRITICAL_COLS


def _esc(val):
    """html.escape for DATA-ORIGINATED values interpolated into
    unsafe_allow_html f-string HTML across the ui/ layer.

    LCC fields like Unit/RegionName/MNT NAME/SegmentName are manually typed
    at origination and can legitimately contain &, <, > (e.g. "R&B MOTORS",
    "AUTO <PUNE>") -- unescaped, those break the surrounding table markup,
    and since uploads are user-supplied they're also a stored-XSS vector on
    a shared deployment. Gemini/planner output (query titles, insight
    bullets, plan descriptions) is the same risk class and must go through
    here too. report_agent/nodes/report_builder.py has its own _esc for the
    same reason; this is the dashboard-side twin.

    Strings are escaped (quote=True so it's also safe inside attribute
    values like title="..."); everything else (numbers headed into a format
    spec, None) passes through unchanged. Code-defined literals (static
    labels, bucket names) don't need this -- apply it at the point where a
    value from the uploaded frame or an LLM reaches an f-string.
    """
    return html.escape(val, quote=True) if isinstance(val, str) else val


def query_confidence_tier(result: dict) -> tuple[str, str, str]:
    """Classify how much validation an AI Query answer received, purely from
    signals already present in the final QueryState (graph.py) -- no new
    computation, just naming a distinction the pipeline already makes.

    Returns (label, color, explanation), in descending confidence order:
    - Verified: served by a fast-path view (registry/views.py) -- the exact
      same analysis/ function the dashboard tabs call, so this number IS the
      dashboard number, not a re-derivation of it.
    - Priority Rules: served by the fixed seven-tier business-priority
      framework (agents/data_executor.py), a deterministic rule set, not an
      LLM-authored query plan.
    - Validated: the Logical Planner's IR-1 compiled and passed validation
      against the file's actual columns on the first attempt.
    - Self-corrected: the first attempt failed compile/validation and needed
      one repair pass (graph.py's one-shot repair loop) before it validated.
    """
    # Verified: the query matched a pre-built view (registry/views.py) -- the
    # answer was computed by the EXACT SAME function a dashboard tab calls,
    # not re-derived by the LLM's own query plan. Checked first: a view match
    # wins even if repair_attempts is nonzero below, since the view/priority
    # paths never go through the compiler (repair_attempts is meaningless there).
    if result.get("view_render"):
        return (
            "Verified", "#16a34a",
            "Served by the same pre-built analysis used on the dashboard tabs -- this number matches what you'd see there.",
        )
    # Priority Rules: routed to the fixed seven-tier business-priority
    # framework (agents/data_executor.py::execute_priority_mode) instead of
    # the general LLM-authored query path -- a deterministic rule set written
    # once, not the LLM improvising a plan for this specific query.
    if result.get("priority_mode"):
        return (
            "Priority Rules", "#2563eb",
            "Served by the fixed seven-tier business-priority framework, not an LLM-authored query plan.",
        )
    # Self-corrected: the LLM's first IR-1 plan failed compiler validation
    # (e.g. referenced a column/filter that didn't check out against the
    # file), and graph.py's one-shot repair loop fed the error back and got a
    # plan that passed on the retry. Still a validated answer, just weaker
    # evidence than passing validation on the first attempt.
    if result.get("repair_attempts", 0) > 0:
        return (
            "Self-corrected", "#d97706",
            "The query plan failed validation on the first attempt and needed one automatic correction before it ran.",
        )
    # Validated: the LLM's first IR-1 plan compiled and passed validation
    # against the file's actual columns on the very first attempt -- no
    # repair needed. The default/best outcome on the general compiler path.
    return (
        "Validated", "#0891b2",
        "The query plan was checked against your data's actual columns and validated on the first attempt.",
    )


def _confidence_badge_html(result: dict) -> str:
    label, color, tooltip = query_confidence_tier(result)
    return (
        f'<span title="{_esc(tooltip)}" style="font-size:11px;font-weight:700;color:{color};'
        f'border:1px solid {color};padding:2px 9px;border-radius:20px;white-space:nowrap;">{label}</span>'
    )


# Column-name tokens that mark a numeric column as non-summable: either an
# identifier (Loan No, Cust Mob No, MNT CODE, Veh ID, DPD, Tenure, Rank) or
# already an average/rate over its group (Avg Ticket (L), Avg Loan (L)) --
# summing either across rows produces a meaningless number (a summed phone
# number, or an "average of averages"), so append_total_row() leaves them
# blank instead of guessing. Matched as whole words against the column name
# so "NPA Count"/"Accounts" (genuinely summable) don't get caught by a loose
# substring match.
_NON_SUMMABLE_TOKENS = {"no", "number", "code", "id", "mob", "dpd", "tenure", "rank", "avg", "average", "mean"}


def _is_id_like_column(col: str) -> bool:
    tokens = re.findall(r"[a-z]+", col.lower())
    return any(t in _NON_SUMMABLE_TOKENS for t in tokens)


def append_total_row(df: pd.DataFrame, ratio_cols: dict | None = None) -> pd.DataFrame:
    """Append a synthetic 'Total' row to `df` for display.

    - First column: literal "Total".
    - Other text/object columns: repeat the column's own header (so a scan
      down the Total row reads as labels, not blanks).
    - Plain numeric columns: summed -- except id-like columns (see
      _is_id_like_column), which are left blank rather than summed.
    - Columns named in `ratio_cols` (e.g. {"Collection %": ("Collected",
      "Month Demand")}) are recomputed as sum(numerator)/sum(denominator)*scale
      rather than averaged, since averaging per-row percentages/averages is not
      the same number as the portfolio-wide ratio. `scale` defaults to 100 (for
      "%" columns) -- pass a 3-tuple (num_col, den_col, scale) for a non-percent
      derived average like "Avg Ticket (L)" = Funded (Cr) / Accounts * 100 (the
      Cr-to-L unit conversion). Both the ratio column and its numerator/
      denominator must all be present in `df` (they don't need to be displayed
      columns themselves, just present in the frame passed in here).

    No-op (returns `df` unchanged) if `df` is empty -- there's nothing to
    total, and an all-NaN/empty-string Total row on an empty table reads as a
    display bug.
    """
    if df.empty:
        return df
    first_col = df.columns[0]
    ratio_cols = ratio_cols or {}
    total = {}
    for col in df.columns:
        if col == first_col:
            total[col] = "Total"
        elif col in ratio_cols:
            spec = ratio_cols[col]
            num_col, den_col = spec[0], spec[1]
            scale = spec[2] if len(spec) > 2 else 100
            num = pd.to_numeric(df[num_col], errors="coerce").sum() if num_col in df.columns else None
            den = pd.to_numeric(df[den_col], errors="coerce").sum() if den_col in df.columns else None
            total[col] = round(num / den * scale, 2) if den else 0.0
        elif "%" in col:
            # A %-named column with no explicit ratio_cols entry: summing (or
            # averaging) per-row percentages doesn't produce a valid portfolio
            # ratio, so leave it blank rather than show a misleading number.
            total[col] = ""
        elif pd.api.types.is_numeric_dtype(df[col]) and not _is_id_like_column(col):
            s = pd.to_numeric(df[col], errors="coerce").sum()
            total[col] = round(float(s), 2) if pd.api.types.is_float_dtype(df[col]) else int(s)
        else:
            total[col] = col
    return pd.concat([df, pd.DataFrame([total])], ignore_index=True)


def _file_fingerprint(files) -> str:
    """SHA-256 fingerprint (first 16 hex chars) of one or more uploaded files'
    raw bytes, used to build a globally-unique `_data_version` (see
    _bump_data_version below). `.getvalue()` reads the buffer without
    consuming it, so this is safe to call before the same file object is
    later read by load_and_validate()."""
    if not files:
        return "none"
    if not isinstance(files, list):
        files = [files]
    h = hashlib.sha256()
    for f in files:
        h.update(f.getvalue())
    return h.hexdigest()[:16]


def _bump_data_version(fingerprint: str | None = None) -> None:
    """Call exactly once at every point df_curr_raw/df_prev_raw are freshly
    assigned in session_state (a new "Generate Dashboard" click, the sample-data
    button, or the deferred prev-file auto-load). Every @st.cache_data-wrapped
    function downstream keys on this value instead of hashing the full
    DataFrame content -- Streamlit's default hasher walking a 50k-row x
    85-column frame on every rerun (even on a guaranteed cache hit) is the
    single biggest cost paid on every interaction with this app, not just
    actual filter changes. Missing a call site here means every downstream
    cache silently serves stale data with no error -- never skip this.

    `fingerprint` should be `_file_fingerprint(curr_file, prev_file)` (or
    similar) for any real user upload. st.cache_data's cache is shared across
    ALL sessions on the server process, not private per browser tab -- a bare
    per-session incrementing counter lets two independent users' sessions
    each start at the same count (e.g. both at 1 on their first upload) and
    collide on an identical cache key despite completely different
    underlying data, silently serving one user's results to another. Hashing
    the actual uploaded bytes makes the key deterministic on content, so this
    collision becomes structurally impossible instead of merely unlikely.
    Falls back to a per-session counter only for the sample-data path, where
    every session loads the exact same public GitHub file anyway, so sharing
    that cache entry across sessions is correct behavior, not a privacy risk."""
    if fingerprint is not None:
        st.session_state["_data_version"] = fingerprint
    else:
        st.session_state["_data_version"] = st.session_state.get("_data_version", 0) + 1


def _style_main_content_selectbox(color: str = "#fff") -> None:
    """Fix near-invisible dark-on-dark text on a main-content (non-sidebar)
    st.selectbox -- ui/styles.py's white-text rule only targets
    `[data-testid="stSidebar"] .stSelectbox`, so any selectbox rendered
    outside the sidebar keeps the default dark text on the dark selectbox
    background.

    This targets EVERY `stSelectbox` on the page (Streamlit doesn't scope
    injected `st.markdown` styles to one call site), so every caller should
    use the SAME color for visual consistency. Only the active tab's render
    code runs per rerun (app.py's segmented-control switcher), so a mismatched
    color between callers can no longer leak into a different, inactive tab --
    it would just look inconsistent if this tab's own color ever differed from
    the others'. Callers: ui/tabs/migration.py, ui/tabs/ai_query.py,
    ui/tabs/business.py.
    """
    # data-baseweb="select" is gone as of Streamlit 1.59.0 (the Selectbox
    # widget dropped BaseWeb entirely) -- role="combobox" is the new stable
    # target. Both kept side by side, same cross-version-safety reasoning as
    # ui/styles.py's own tab-navigation/sidebar-selectbox fixes.
    st.markdown(f"""
<style>
div[data-testid="stSelectbox"] [data-baseweb="select"] *,
div[data-testid="stSelectbox"] [data-baseweb="select"] div,
div[data-testid="stSelectbox"] [data-baseweb="select"] span,
div[data-testid="stSelectbox"] [role="combobox"],
div[data-testid="stSelectbox"] [role="combobox"] * {{ color: {color} !important; }}
</style>""", unsafe_allow_html=True)


def _dateonly(df: pd.DataFrame) -> pd.DataFrame:
    """Drop the always-00:00:00 time component from datetime64 columns.

    LCC date columns (Ag_Date, Last Receipt Date, etc.) carry no real time-of-day
    signal -- they're loan-level dates, not timestamps -- but pandas' datetime64
    dtype always renders the full "2022-11-01 00:00:00" in both st.dataframe and
    an exported .xlsx cell. .dt.date yields plain python date objects, which
    display and export as a bare date with no formatting/dtype trickery needed."""
    df = df.copy()
    for col in df.select_dtypes(include="datetime64[ns]").columns:
        df[col] = df[col].dt.date
    return df


def _safe_df(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce mixed-type object columns to string for safe st.dataframe display."""
    df = _dateonly(df)
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).replace({"nan": "", "None": ""})
    return df


def _chart_card(fig) -> None:
    """Render a Plotly figure inside the standard bordered .chart-card wrapper."""
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.plotly_chart(fig, width='stretch')
    st.markdown('</div>', unsafe_allow_html=True)


def _divider(margin: str = "24px 0") -> None:
    """Standard horizontal section divider."""
    st.markdown(f'<div style="border-top:1px solid #e5e7eb;margin:{margin};"></div>', unsafe_allow_html=True)


def _empty_state(icon: str, title: str, sub: str) -> None:
    """Standard 'nothing to show yet' placeholder block."""
    st.markdown(
        f'<div class="empty-state">'
        f'<div class="empty-state-icon">{icon}</div>'
        f'<div class="empty-state-title">{title}</div>'
        f'<div class="empty-state-sub">{sub}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


# max_entries=64: one entry per distinct table rendered behind a download
# button (~20+ per full tab walk, x filter combinations). Excel bytes are
# smaller than the DataFrames upstream, so the limit is looser -- but still
# bounded, since by default st.cache_data keeps every entry forever.
@st.cache_data(show_spinner=False, max_entries=64)
def _excel_bytes(df: pd.DataFrame) -> bytes:
    buf = BytesIO()
    _dateonly(df).to_excel(buf, index=False, engine="openpyxl")
    return buf.getvalue()


def _dl_btn(df: pd.DataFrame, filename: str, key: str) -> None:
    """Right-aligned compact Excel download button.

    Streamlit reruns the ENTIRE script (all 7 tabs, not just the active one --
    tabs are only CSS-hidden when inactive, their code still executes) on
    every single interaction anywhere in the app, e.g. clicking "Run Query" in
    the AI Query tab. Without caching, that meant every one of this app's ~20
    _dl_btn call sites re-ran an uncached openpyxl df.to_excel() -- cell-by-cell,
    not vectorized -- on every rerun, regardless of whether the underlying
    table had changed.

    _excel_bytes is @st.cache_data itself rather than hand-rolling a `df is`
    identity cache here (the previous approach): st.cache_data hashes the
    upstream df's own upstream st.cache_data call also returns a fresh COPY on
    every cache hit, not the original object -- proven directly: two calls to
    a trivial @st.cache_data function with identical args gave `is` False both
    times. So a `df is` check keyed on "the same object survives a cache hit"
    was actually a permanent cache miss on every single rerun, for every
    _dl_btn call site in the app (confirmed via profiling: render_alerts_tab's
    31 _dl_btn calls alone cost ~53s of a ~62s warm rerun on real 60k-row
    data). Content hashing fixes this correctly regardless of object
    identity -- verified: a fresh, freshly-copied 2,000-row DataFrame with
    identical content hits cache in ~0.025s vs a ~3.3s cold write."""
    data = _excel_bytes(df)

    _, col = st.columns([5, 1])
    with col:
        st.download_button(
            "⬇ Excel", data=data, file_name=filename,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key=key, width='stretch',
        )


def _cached_html_export(cache_key: tuple, build_fn, *args, **kwargs) -> str:
    """ui/tabs/dashboard.py's build_html_export call ran unconditionally on
    every render (not gated behind the download button click -- Streamlit's
    download_button needs its `data=` bytes ready upfront, so there's no
    native "compute on click" for this), and it's expensive: pio.to_html() on
    3 full Plotly figures plus assembling a large formatted HTML string, every
    single Streamlit rerun of the ENTIRE app, since Dashboard is tabs[0] and
    always executes regardless of which tab is actually visible.

    `cache_key` must be the same (data_version, region, branch, status,
    segment) tuple app.py's own _cached_* functions already use as their
    cache key -- NOT `df_curr is` the object from last time. That was the
    original design here (mirroring _dl_btn's old approach) but it never
    actually worked: df_curr itself comes from a @st.cache_data call
    (_cached_filter), and st.cache_data returns a fresh COPY on every cache
    hit, not the original object -- proven directly by calling a trivial
    @st.cache_data function twice with identical args and getting `is` False
    both times. So this was a guaranteed cache miss on every single rerun.
    build_fn/its figure args aren't reliably auto-hashable by st.cache_data
    (Plotly Figure objects, arbitrary callables), so this uses the same
    explicit-tuple-key idiom the rest of app.py's caching already relies on,
    rather than trying to make Streamlit hash them."""
    cache = st.session_state.setdefault("_html_export_cache", {})
    cached = cache.get("dashboard")
    if cached is not None and cached[0] == cache_key:
        return cached[1]
    html_content = build_fn(*args, **kwargs)
    cache["dashboard"] = (cache_key, html_content)
    return html_content


def _kpi_card_html(
    label: str, value: str, delta, *, unit: str = "%", inverse: bool = False,
    count_delta: int | None = None, zero_delta_bad: bool = False,
    good_override: bool | None = None,
) -> str:
    """Shared KPI card markup: label + big value + MoM delta arrow.

    `inverse=True` means a falling value is the good direction (e.g. NPA %)
    so the arrow color logic flips. `delta=None` renders "no prev data"
    instead of an arrow (used when no previous-month file was uploaded).
    `delta` must always be the RAW (current - previous) movement -- arrow
    direction is derived directly from its sign and is never flipped by
    `inverse`, which only controls color. (Regression this fixes: a caller
    used to pre-flip the delta's sign for inverse metrics before passing it
    in here, which combined with this function's own inverse-aware color
    logic into a double sign-flip -- both the arrow direction AND the color
    ended up backwards for every inverse metric it fed. Fixed at the source;
    this contract is why.)

    `good_override` lets a caller supply the good/bad color directly instead
    of deriving it from delta's sign + `inverse`, for metrics where "which
    direction is good" isn't a fixed property of the metric but depends on
    what's driving the change this period (e.g. SOH rising is good if POS
    growth is driving it, bad if Closing Arrears growth is driving it).
    Arrow direction is unaffected -- it still always reflects delta's sign.

    When the displayed delta rounds to 0.00%, two independent tabs in this
    app have always disagreed on the color (Dashboard: red: Portfolio
    Intelligence: green) - `zero_delta_bad` lets each caller keep its own
    pre-existing convention instead of silently picking one for both.

    If `count_delta` (the underlying raw account-count movement) is also
    passed and the displayed delta rounds to 0.00%, it takes priority over
    `zero_delta_bad` and breaks the tie using the real count instead: a
    +/-1 account move that got rounded away by the percentage still shows
    as worsened/improved, and a true zero-count move shows neutral/grey.
    """
    if delta is None:
        mom_html = '<div class="kpi-mom" style="color:#9ca3af;">no prev data</div>'
    else:
        is_tied = round(delta, 2) == 0.0
        if is_tied and count_delta:
            arrow = "▲" if count_delta > 0 else "▼"
            good  = (count_delta < 0) if inverse else (count_delta > 0)
            cls   = "kpi-mom-up" if good else "kpi-mom-down"
        elif is_tied and count_delta == 0:
            arrow, cls = "–", "kpi-mom-neutral"
        elif is_tied and zero_delta_bad:
            arrow = "▲" if delta >= 0 else "▼"
            cls   = "kpi-mom-down" if inverse else "kpi-mom-up"
        else:
            arrow = "▲" if delta >= 0 else "▼"
            good  = ((delta <= 0) if inverse else (delta >= 0)) if good_override is None else good_override
            cls   = "kpi-mom-up" if good else "kpi-mom-down"
        mom_html = f'<div class="kpi-mom">MoM <span class="{cls}">{arrow} {abs(delta):.2f}{unit}</span></div>'
    return (
        f'<div class="kpi-card">'
        f'<div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div>'
        f'{mom_html}</div>'
    )


def _static_kpi_card_html(
    label: str, value, caption: str = "", *,
    color: str = "", value_style: str = "", card_style: str = "",
) -> str:
    """A plain (no MoM-delta) KPI card: colored top border, label, big value,
    optional caption line. Used for one-off summary numbers (e.g. "Fleet
    Operators: 12") where there's no month-over-month comparison to show."""
    border    = f"border-top-color:{color};" if color else ""
    card_attr = f' style="{border}{card_style}"' if (border or card_style) else ""
    val_attr  = f' style="color:{color};{value_style}"' if (color or value_style) else ""
    caption_html = f'<div class="kpi-mom">{caption}</div>' if caption else ""
    return (
        f'<div class="kpi-card"{card_attr}>'
        f'<div class="kpi-label">{label}</div>'
        f'<div class="kpi-value"{val_attr}>{value}</div>'
        f'{caption_html}</div>'
    )


def _npa_pct_color(val: float, hi: float = 10.0, mid: float = 5.0) -> str:
    """Red/orange/green threshold color for an NPA%-style risk percentage."""
    v = val or 0
    return "#dc2626" if v > hi else ("#d97706" if v > mid else "#16a34a")


def _sma2_pct_color(val: float, hi: float = 10.0, mid: float = 5.0) -> str:
    """Red/orange/grey threshold color for a SMA-2%-style risk percentage."""
    v = val or 0
    return "#ef4444" if v > hi else ("#d97706" if v > mid else "#374151")


def _send_feedback(run_id: str, score: float) -> None:
    """Post thumbs-up (1.0) or thumbs-down (0.0) feedback to LangSmith."""
    try:
        from langsmith import Client as _LSClient
        _LSClient().create_feedback(run_id=run_id, key="result_quality", score=score)
    except Exception:
        pass


def _send_report_email(html_report: str, email_to: str, curr_month: str) -> tuple[bool, str]:
    """Send the HTML report via SMTP. Returns (success, error_msg).

    Thin wrapper around report_agent's send_report_email - the same function
    the auto-send-on-generate LangGraph node uses, so this tab's standalone
    "resend" button can't silently drift out of sync with it.
    """
    from report_agent.nodes.email_dispatcher import send_report_email
    return send_report_email(html_report, email_to, curr_month)


def _load_and_concat(files) -> tuple[pd.DataFrame | None, list[str]]:
    """Load one or more Excel files and concatenate into a single DataFrame."""
    if not files:
        return None, ["No file uploaded"]
    if not isinstance(files, list):
        files = [files]

    dfs, errors = [], []
    dropped_dupes = 0  # per-file dupes (from load_and_validate) + any cross-file dupes below
    for f in files:
        df, errs = load_and_validate(f)
        if errs:
            errors.append(f"{getattr(f, 'name', 'file')}: {errs[0]}")
        else:
            dropped_dupes += df.attrs.get("dropped_duplicate_loans", 0)
            dfs.append(df)

    if not dfs:
        return None, errors

    if len(dfs) == 1:
        return dfs[0], errors

    def _normalize_dt(df):
        for col in df.select_dtypes(include="datetime64").columns:
            try:
                df[col] = df[col].astype("datetime64[ms]")
            except Exception:
                df[col] = pd.to_datetime(df[col], errors="coerce")
        return df

    dfs = [_normalize_dt(d) for d in dfs]
    combined = pd.concat(dfs, ignore_index=True)
    if "Loan No" in combined.columns:
        before = len(combined)
        combined = combined.drop_duplicates(subset=["Loan No"], keep="first")
        dropped_dupes += before - len(combined)
    # pd.concat doesn't propagate .attrs from its inputs, so set it explicitly
    # on the combined frame -- this is the only place callers need to check.
    combined.attrs["dropped_duplicate_loans"] = dropped_dupes
    # Recomputed fresh on the FINAL combined frame, not unioned from individual
    # files' own attrs -- pd.concat already merges each file's columns (a column
    # present in only one regional file still ends up in `combined`, NaN-filled
    # for the others), so "missing" only means anything once evaluated here.
    combined.attrs["missing_optional_cols"] = [
        c for c in REQUIRED_COLS if c not in CRITICAL_COLS and c not in combined.columns
    ]
    return combined, errors
