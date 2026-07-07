"""Shared UI helpers used across multiple tab modules."""

from io import BytesIO

import pandas as pd
import streamlit as st

from utils import load_and_validate, REQUIRED_COLS, CRITICAL_COLS


def _bump_data_version() -> None:
    """Call exactly once at every point df_curr_raw/df_prev_raw are freshly
    assigned in session_state (a new "Generate Dashboard" click, the sample-data
    button, or the deferred prev-file auto-load). Every @st.cache_data-wrapped
    function downstream keys on this counter instead of hashing the full
    DataFrame content -- Streamlit's default hasher walking a 50k-row x
    85-column frame on every rerun (even on a guaranteed cache hit) is the
    single biggest cost paid on every interaction with this app, not just
    actual filter changes. Missing a call site here means every downstream
    cache silently serves stale data with no error -- never skip this."""
    st.session_state["_data_version"] = st.session_state.get("_data_version", 0) + 1


def _style_main_content_selectbox(color: str = "#fff") -> None:
    """Fix near-invisible dark-on-dark text on a main-content (non-sidebar)
    st.selectbox -- ui/styles.py's white-text rule only targets
    `[data-testid="stSidebar"] .stSelectbox`, so any selectbox rendered
    outside the sidebar keeps the default dark text on the dark selectbox
    background.

    This targets EVERY `stSelectbox` on the page (Streamlit doesn't scope
    injected `st.markdown` styles to one tab -- all tabs render into the DOM
    simultaneously, just CSS-hidden when inactive), so every caller must use
    the SAME color, or whichever caller renders last in a given rerun wins
    for all of them. Callers: ui/tabs/migration.py, ui/tabs/ai_query.py.
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


def _safe_df(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce mixed-type object columns to string for safe st.dataframe display."""
    df = df.copy()
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


def _excel_bytes(df: pd.DataFrame) -> bytes:
    buf = BytesIO()
    df.to_excel(buf, index=False, engine="openpyxl")
    return buf.getvalue()


def _dl_btn(df: pd.DataFrame, filename: str, key: str) -> None:
    """Right-aligned compact Excel download button.

    Streamlit reruns the ENTIRE script (all 7 tabs, not just the active one --
    tabs are only CSS-hidden when inactive, their code still executes) on
    every single interaction anywhere in the app, e.g. clicking "Run Query" in
    the AI Query tab. Without caching, that meant every one of this app's ~20
    _dl_btn call sites re-ran an uncached openpyxl df.to_excel() -- cell-by-cell,
    not vectorized -- on every rerun, regardless of whether the underlying
    table had changed. Cache one entry per button key, keyed on `df is` the
    exact object we cached bytes for last time (the df objects passed in are
    themselves already st.cache_data-cached upstream, so the SAME object
    survives across reruns whenever filters/data are unchanged). This holds a
    strong reference to that df in the cache entry itself, which is what makes
    the identity check safe: as long as the entry exists, Python can't garbage
    -collect that df and hand its id to an unrelated object -- the exact
    ABA-style collision an `id(df)`-keyed cache would otherwise be exposed to."""
    cache = st.session_state.setdefault("_excel_bytes_cache", {})
    cached = cache.get(key)
    if cached is not None and cached[0] is df:
        data = cached[1]
    else:
        data = _excel_bytes(df)
        cache[key] = (df, data)

    _, col = st.columns([5, 1])
    with col:
        st.download_button(
            "⬇ Excel", data=data, file_name=filename,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key=key, width='stretch',
        )


def _cached_html_export(df_curr: pd.DataFrame, build_fn, *args, **kwargs) -> str:
    """Same identity-cache idiom _dl_btn already uses above, for a different
    culprit: ui/tabs/dashboard.py's build_html_export call ran unconditionally
    on every render (not gated behind the download button click -- Streamlit's
    download_button needs its `data=` bytes ready upfront, so there's no native
    "compute on click" for this), and it's expensive: pio.to_html() on 3 full
    Plotly figures plus assembling a large formatted HTML string, every single
    Streamlit rerun of the ENTIRE app, since Dashboard is tabs[0] and always
    executes regardless of which tab is actually visible.

    Keyed on `df_curr is` the exact object from last time -- df_curr is itself
    already st.cache_data-cached upstream (via app.py's _cached_filter), so the
    SAME object survives across reruns whenever filters/data are unchanged, and
    every other build_html_export argument (metrics, figures, alerts,
    scorecard_df) is derived from that same df_curr+filter combination, so an
    unchanged df_curr identity guarantees they're unchanged too -- same
    reasoning _dl_btn's own docstring spells out for the Excel case."""
    cache = st.session_state.setdefault("_html_export_cache", {})
    cached = cache.get("dashboard")
    if cached is not None and cached[0] is df_curr:
        return cached[1]
    html_content = build_fn(*args, **kwargs)
    cache["dashboard"] = (df_curr, html_content)
    return html_content


def _kpi_card_html(
    label: str, value: str, delta, *, unit: str = "%", inverse: bool = False,
    count_delta: int | None = None, zero_delta_bad: bool = False,
) -> str:
    """Shared KPI card markup: label + big value + MoM delta arrow.

    `inverse=True` means a falling value is the good direction (e.g. NPA %)
    so the arrow color logic flips. `delta=None` renders "no prev data"
    instead of an arrow (used when no previous-month file was uploaded).

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
            good  = (delta <= 0) if inverse else (delta >= 0)
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
