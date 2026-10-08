"""Shared UI helpers used across multiple tab modules."""

import datetime
import hashlib
import html
import re
from io import BytesIO

import pandas as pd
import streamlit as st
from openpyxl.formatting.rule import ColorScaleRule
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

from utils import load_and_validate, expand_to_all_columns, add_customer_loan_count, REQUIRED_COLS, CRITICAL_COLS


def _esc(val):
    """html.escape for DATA-ORIGINATED values interpolated into
    unsafe_allow_html f-string HTML across the ui/ layer.

    LCC fields like Unit/RegionName/MNT NAME/SegmentName are manually typed
    at origination and can legitimately contain &, <, > (e.g. "R&B MOTORS",
    "AUTO <PUNE>") -- unescaped, those break the surrounding table markup,
    and since uploads are user-supplied they're also a stored-XSS vector on
    a shared deployment. Gemini/planner output (query titles, insight
    bullets, plan descriptions) is the same risk class and must go through
    here too. report_agent/render.py escapes its own HTML for the
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
            "Served by the same pre-built analysis used on the dashboard tabs: this number matches what you'd see there.",
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

    # app.py calls this on EVERY rerun (the stale-file guard), and hashing a
    # 200MB upload each click is real latency. Streamlit gives each upload a
    # unique file_id, so remember the digest per (file_id, size) and only
    # hash when a new file actually appears. Objects without a file_id
    # (tests, non-Streamlit callers) are always hashed.
    ident = tuple((getattr(f, "file_id", None), getattr(f, "size", None)) for f in files)
    memo = None
    if all(fid is not None for fid, _ in ident):
        memo = st.session_state.setdefault("_file_fingerprint_memo", {})
        if ident in memo:
            return memo[ident]

    h = hashlib.sha256()
    for f in files:
        h.update(f.getvalue())
    digest = h.hexdigest()[:16]
    if memo is not None:
        if len(memo) >= 8:  # only the current curr/prev selections matter
            memo.clear()
        memo[ident] = digest
    return digest


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


TOP_CHOICES = [10, 20, 50, 100, "All"]


WHOLE_LIST = "Whole list"


def list_controls(key: str, df: pd.DataFrame, branch_col: str = "Branch", noun: str = "rows",
                  region_col: str = "Region", source: pd.DataFrame | None = None) -> pd.DataFrame:
    """Top X with Zone, Region and Branch filters (each narrowing the next)
    and "Top X in each" zone / region / branch, for any list already sorted
    with the most important rows first. A level shows only when the list has
    its column with 2+ values. source: the loans the list came from; when the
    list has Loan No but no Zone, each loan's zone is looked up from it."""
    if "Zone" not in df.columns and source is not None and "Loan No" in df.columns \
            and {"Loan No", "Zone"} <= set(source.columns):
        zone_of = source.drop_duplicates("Loan No").set_index("Loan No")["Zone"]
        df = df.assign(Zone=df["Loan No"].map(zone_of))
    levels = [(label, col) for label, col in (("Zone", "Zone"), ("Region", region_col), ("Branch", branch_col))
              if col in df.columns and df[col].nunique(dropna=True) > 1]
    boxes = st.columns([1] + [1.1] * len(levels) + [1.3])
    n = boxes[0].selectbox("Show top", TOP_CHOICES, index=0, key=f"{key}_n")
    scope = df
    for (label, col), box in zip(levels, boxes[1:]):
        options = ["All"] + sorted(scope[col].dropna().astype(str).unique())
        wkey = f"{key}_{label[0].lower()}"            # _z, _r, _b
        if st.session_state.get(wkey) not in options:  # e.g. a branch outside the region just picked
            st.session_state[wkey] = "All"
        pick = box.selectbox(label, options, key=wkey)
        if pick != "All":
            scope = scope[scope[col].astype(str) == pick]
    # "Top X in each": only the levels still with 2+ values after the filters.
    group_by = {label: col for label, col in levels if scope[col].nunique(dropna=True) > 1}
    each_options = [WHOLE_LIST, *group_by]
    if st.session_state.get(f"{key}_each") not in each_options:
        st.session_state[f"{key}_each"] = WHOLE_LIST
    each = boxes[-1].selectbox(f"Top {n} in each" if n != "All" else "Group by", each_options, key=f"{key}_each")
    d = scope
    if each != WHOLE_LIST:
        col = group_by[each]
        d = d.groupby(col, sort=True, group_keys=False).head(n if n != "All" else len(d))
        d = d.sort_values(col, kind="stable")
    elif n != "All":
        d = d.head(n)
    st.caption(f"Showing {len(d):,} of {len(df):,} {noun}"
               + (f": the top {n} in each {each.lower()}." if each != WHOLE_LIST and n != "All" else "."))
    return d


# The place hierarchy every screen filters and breaks down by: (label, column).
PLACE_LEVELS = (("Zone", "Zone"), ("Region", "RegionName"), ("Branch", "Unit"))


def place_filters(df: pd.DataFrame, key: str, container=None) -> tuple[pd.DataFrame, dict]:
    """Zone, Region and Branch dropdowns over loan rows, each narrowing the
    next (a level shows only when its column has 2+ values in view). A pick
    that no longer fits (a branch outside the region just picked) goes back to
    All. Returns (the loans in the chosen place, {label: pick} for each level
    not on All)."""
    levels = [(label, col) for label, col in PLACE_LEVELS if col in df.columns and df[col].nunique(dropna=True) > 1]
    if not levels:
        return df, {}
    boxes = (container or st).columns(len(levels))
    scope, picks = df, {}
    for (label, col), box in zip(levels, boxes):
        options = ["All"] + sorted(scope[col].dropna().astype(str).unique())
        wkey = f"{key}_{label.lower()}"
        if st.session_state.get(wkey) not in options:
            st.session_state[wkey] = "All"
        pick = box.selectbox(label, options, key=wkey)
        if pick != "All":
            scope, picks[label] = scope[scope[col].astype(str) == pick], pick
    return scope, picks


def apply_place(df: pd.DataFrame, picks: dict, loose: bool = False) -> pd.DataFrame:
    """Rows in the place `picks` names ({label: value}, as place_filters
    returns). loose: match ignoring case and spaces (last month's file, whose
    names may be typed differently)."""
    from utils import _unit_key
    cols = dict(PLACE_LEVELS)
    for label, value in picks.items():
        col = cols[label]
        if col not in df.columns:
            continue
        df = df[df[col].map(_unit_key) == _unit_key(value)] if loose else df[df[col].astype(str) == str(value)]
    return df


def takeaway(text: str) -> None:
    """The one-line yellow summary at the top of a view (text may hold <b>)."""
    st.markdown(f'<div style="border-left:4px solid #FFC000;background:#fffbea;padding:8px 12px;'
                f'border-radius:6px;font-size:13.5px;color:#111827;margin:4px 0 10px 0;">{text}</div>',
                unsafe_allow_html=True)


def section_label(title: str, margin_top: str = "0px") -> None:
    """The yellow section heading used across the tabs."""
    st.markdown(f'<div class="section-label" style="margin-top:{margin_top};">{title}</div>',
                unsafe_allow_html=True)


def _chart_card(fig) -> None:
    """Render a Plotly figure inside the standard bordered .chart-card wrapper."""
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.plotly_chart(fig, width='stretch')
    st.markdown('</div>', unsafe_allow_html=True)


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


# max_entries=24: one entry per download actually clicked (downloads are
# built on click). A full-column loan list can be tens of MB, and the cache is
# shared by every session, so it stays bounded.
@st.cache_data(show_spinner=False, max_entries=24)
def _excel_bytes(df: pd.DataFrame) -> bytes:
    buf = BytesIO()
    out = _dateonly(df)
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        out.to_excel(writer, index=False, sheet_name="Sheet1")
        style_excel_sheet(writer.sheets["Sheet1"], out)
    return buf.getvalue()


# ── Excel download styling ────────────────────────────────────────────────────
# Same look as the app's own tables: black header with amber text, bold Total
# row with an amber rule, and red/green colour scales on metric columns.
# One header style for every Excel download: blue with white text.
XL_HEADER_BLUE, XL_HEADER_TEXT, XL_TOTAL_TINT = "1F4E78", "FFFFFF", "DDEBF7"
_XL_HEADER_FILL = PatternFill("solid", fgColor=XL_HEADER_BLUE)
_XL_HEADER_FONT = Font(bold=True, color=XL_HEADER_TEXT)
_XL_TOTAL_FILL = PatternFill("solid", fgColor=XL_TOTAL_TINT)
_XL_TOTAL_BORDER = Border(top=Side(style="medium", color=XL_HEADER_BLUE))
_XL_GREEN, _XL_YELLOW, _XL_RED = "63BE7B", "FFEB84", "F8696B"  # Excel's own default scale
_XL_WHITE = "FFFFFF"

# Colour direction by column name. Only "%" columns (and Concern Score) get a
# scale -- plain counts and amounts aren't "good" or "bad" on their own.
_XL_HIGHER_IS_WORSE = (
    "NPA", "SMA", "DELINQUEN", "HARD BUCKET", "ROLL FWD", "ROLL FORWARD", "1-30 DPD",
    "NOT PAYING", "HARD +", "INSURANCE-ONLY", "INSTALLMENT-ONLY", "CONCERN SCORE",
)
_XL_HIGHER_IS_BETTER = ("COLLECTION", "STRIKE", "LCC", "ROLL BWD", "ROLL BACK")

# Per-cell number formats are the one slow part (openpyxl styles cell by cell);
# past this size (e.g. a 60k-row x 90-col loan dump) they're skipped. The
# header, filters, widths and colour scales are cheap at any size.
_XL_FORMAT_MAX_CELLS = 200_000


def _xl_metric_direction(col) -> str | None:
    name = str(col).upper()
    if "%" not in name and "CONCERN SCORE" not in name:
        return None
    if any(k in name for k in _XL_HIGHER_IS_WORSE):
        return "worse"
    if any(k in name for k in _XL_HIGHER_IS_BETTER):
        return "better"
    return None


def xl_color_scale(direction: str) -> ColorScaleRule:
    """Risk ("worse") columns: white -> red, never green. Good ("better")
    columns keep Excel's red -> yellow -> green."""
    if direction == "worse":
        return ColorScaleRule(start_type="min", start_color=_XL_WHITE, end_type="max", end_color=_XL_RED)
    return ColorScaleRule(
        start_type="min", start_color=_XL_RED,
        mid_type="percentile", mid_value=50, mid_color=_XL_YELLOW,
        end_type="max", end_color=_XL_GREEN,
    )


def style_excel_sheet(ws, df: pd.DataFrame) -> None:
    """Format a sheet that `df` was just written to (header in row 1, no index)."""
    n_rows, n_cols = len(df), len(df.columns)
    if n_cols == 0:
        return

    for cell in ws[1]:
        cell.fill, cell.font = _XL_HEADER_FILL, _XL_HEADER_FONT
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    ws.freeze_panes = "A2"
    if n_rows:
        ws.auto_filter.ref = f"A1:{get_column_letter(n_cols)}{n_rows + 1}"

    sample = df.head(200)
    for j, col in enumerate(df.columns, start=1):
        lengths = [len(str(v)) for v in sample[col].tolist() if not _is_blank(v)]
        ws.column_dimensions[get_column_letter(j)].width = min(max([len(str(col))] + lengths) + 3, 40)

    first = str(df.iloc[-1, 0]).strip() if n_rows else ""
    has_total = first == "Grand Total" or first == "Total" or first.startswith("Total (")
    n_data = n_rows - 1 if has_total else n_rows
    if has_total:
        for cell in ws[n_rows + 1]:
            cell.font, cell.fill, cell.border = Font(bold=True), _XL_TOTAL_FILL, _XL_TOTAL_BORDER

    small = n_rows * n_cols <= _XL_FORMAT_MAX_CELLS
    for j, col in enumerate(df.columns, start=1):
        values = pd.to_numeric(df[col].iloc[:n_data], errors="coerce")
        non_blank = int((~df[col].iloc[:n_data].map(_is_blank)).sum())
        if pd.api.types.is_bool_dtype(df[col]) or non_blank == 0 or values.notna().sum() != non_blank:
            continue  # not a numeric column
        letter = get_column_letter(j)

        direction = _xl_metric_direction(col)
        if direction and n_data >= 2:
            ws.conditional_formatting.add(f"{letter}2:{letter}{n_data + 1}", xl_color_scale(direction))

        if not small:
            continue
        if "%" in str(col):
            fmt = '0.0"%"'  # values are already on a 0-100 scale
        elif _is_id_like_column(str(col)):
            continue
        elif (values.dropna() % 1 == 0).all():
            if values.abs().max() < 10_000:
                continue  # small whole numbers (counts, years, tenure) read fine as-is
            fmt = "#,##0"
        else:
            fmt = "#,##0.00"
        for (cell,) in ws.iter_rows(min_row=2, max_row=n_rows + 1, min_col=j, max_col=j):
            cell.number_format = fmt


def _is_blank(v) -> bool:
    return v is None or v == "" or (not isinstance(v, (list, tuple, dict)) and pd.isna(v))


# The app's tabs, in order (app.py's tab bar). Shared so a "See details"
# button can switch to a tab by its exact label.
TAB_LABELS = ["🗂️ Dashboard", "🎯 Action Lists", "📈 Migration", "📊 Portfolio Intelligence",
              "🔎 Root Cause", "💼 Business", "🤖 AI Query", "🕵️ Investigator", "📋 Report"]
TAB_KEY = "_active_section"


def goto_tab_button(tab: str, key: str, label: str | None = None, state: dict | None = None) -> None:
    """A small button that opens another tab (sets the tab bar's value), and
    optionally a view inside it (`state`: session keys to set)."""
    def _go() -> None:
        st.session_state[TAB_KEY] = tab
        for k, v in (state or {}).items():
            st.session_state[k] = v
    st.button(label or f"See details in {tab.split(' ', 1)[1]} →", key=key, on_click=_go)


def _download_frame(df: pd.DataFrame, full_source: pd.DataFrame | None) -> pd.DataFrame:
    """What a table's download contains: the table itself, or (with
    full_source) every raw upload column for the same loans."""
    return expand_to_all_columns(df, full_source) if full_source is not None else df


def _dl_btn(df: pd.DataFrame, filename: str, key: str, full_source: pd.DataFrame | None = None) -> None:
    """Right-aligned compact Excel download button.

    full_source: pass the loan-level frame the table was derived from (e.g.
    df_curr) to make the DOWNLOAD carry every raw upload column plus the
    table's own computed columns, while the on-screen table stays curated.
    A no-op for tables without a Loan No column (aggregates, scorecards) --
    see utils.expand_to_all_columns.

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
    _, col = st.columns([5, 1])
    with col:
        st.download_button(
            # A function, not bytes: Streamlit builds the file only when the
            # button is clicked. Building every table's styled workbook up
            # front cost seconds per page load (~10s for an 8,000-loan table)
            # for files nobody downloaded.
            "⬇ Excel", data=lambda: _excel_bytes(_download_frame(df, full_source)), file_name=filename,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key=key, width='stretch',
        )


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
            arrow, cls = "●", "kpi-mom-neutral"
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


# ── One table builder for every on-screen table ─────────────────────────────
# A % with a count behind it is ONE cell: "11.9% (37)". The count tells the
# reader how big the base is without a second column; downloads keep both as
# separate numeric columns so they can still be sorted and summed in Excel.

def pct_count(pct, count, decimals: int = 1) -> str:
    """'11.9% (37)' as HTML (the count in grey). '-' when the % is missing."""
    if pct is None or pct == "" or (isinstance(pct, float) and pd.isna(pct)):
        return '<span style="color:#9ca3af;">-</span>'
    tail = "" if count is None or count == "" or (isinstance(count, float) and pd.isna(count)) else \
        f' <span style="color:#6b7280;font-weight:400;">({int(count):,})</span>'
    return f"{float(pct):.{decimals}f}%{tail}"


def pct_amount(pct, cr, decimals: int = 1) -> str:
    """'34.6% (₹121.40 Cr)' as HTML (the amount in grey): a % by SOH with the
    money behind it, as pct_count does for a % by count."""
    if pct is None or pct == "" or (isinstance(pct, float) and pd.isna(pct)):
        return '<span style="color:#9ca3af;">-</span>'
    tail = "" if cr is None or cr == "" or (isinstance(cr, float) and pd.isna(cr)) else \
        f' <span style="color:#6b7280;font-weight:400;">(₹{float(cr):,.2f} Cr)</span>'
    return f"{float(pct):.{decimals}f}%{tail}"


def is_blank(v) -> bool:
    """None / NaN / NA (an empty string is a value, not a blank)."""
    return v is None or (not isinstance(v, str) and pd.isna(v))


_NUMBER_KINDS = ("int", "pct", "num", "cr", "inr", "rs_cr", "pp", "count_change", "pct_change")
_CHANGE_KINDS = ("pp", "count_change", "pct_change")     # coloured: red when worse, green when better


def format_value(v, kind: str) -> str:
    """One value as plain text, by kind: int "1,234" | pct "12.3%" | num "1.23"
    | cr "₹1.23 Cr" | inr "₹1,234" | rs_cr (rupees shown in crore) | pp
    "▲ 0.25 pts" | count_change "▲ 12" | pct_change "▲ 9.5%" (a relative change); "-" when blank. The one formatter
    behind the on-screen tables and the report (HTML, PDF)."""
    if is_blank(v):
        return "-"
    if kind == "int":
        return f"{int(v):,}"
    if kind == "pct":
        return f"{float(v):.1f}%"
    if kind == "num":
        return f"{float(v):.2f}"
    if kind == "cr":
        return f"₹{float(v):,.2f} Cr"
    if kind == "inr":
        return f"₹{round(float(v)):,}"
    if kind == "rs_cr":
        return f"₹{float(v) / 1e7:,.2f} Cr"
    if kind == "pp":
        v = float(v)
        return "no change" if v == 0 else f"{'▲' if v > 0 else '▼'} {abs(v):.2f} pts"
    if kind == "count_change":
        v = int(v)
        return "no change" if v == 0 else f"{'▲' if v > 0 else '▼'} {abs(v):,}"
    if kind == "pct_change":
        v = float(v)
        return "no change" if v == 0 else f"{'▲' if v > 0 else '▼'} {abs(v):.1f}%"
    if isinstance(v, (pd.Timestamp, datetime.date)):
        return pd.Timestamp(v).strftime("%Y-%m-%d")
    return str(v)


def _fmt_cell(v, kind: str, row=None, count_key: str | None = None, amount_key: str | None = None) -> str:
    if callable(kind):              # trusted HTML from code (a badge, an icon)
        return kind(v, row)
    if kind == "pct_count":
        return pct_count(v, row.get(count_key) if row is not None and count_key else None)
    if kind == "pct_cr":
        return pct_amount(v, row.get(amount_key) if row is not None and amount_key else None)
    if is_blank(v) or (kind in _NUMBER_KINDS and isinstance(v, str) and not v.strip()):
        return '<span style="color:#9ca3af;">-</span>'   # "" = a Total-row cell that can't be summed
    if kind in _NUMBER_KINDS:
        return format_value(v, kind)
    return _esc(v)


def _align(c: dict) -> str:
    return c.get("align") or ("left" if c.get("fmt", "text") == "text" else "right")


def html_table(df: pd.DataFrame, cols: list[dict], total: dict | None = None, max_height: int | None = None,
               highlight=None) -> str:
    """cols: {"key", "label"?, "fmt": text|int|pct|pct_count|pct_cr|cr|inr|num|pp|count_change|pct_change, or a
    function (value, row) -> trusted HTML, "align"?: "left"/"right" (default: text left, numbers right),
    "count"?: count column for pct_count, "amount"?: ₹ Cr column for pct_cr, "heat"?: True (red shading by rank,
    lowest = palest), "good_if_up"?: True, or row -> bool for a per-row choice (a rise is good: green arrows),
    "help"?: hover text, "bold"?: True}. total: an optional final row (dict).
    max_height: scroll inside the table with the header kept in view.
    highlight: optional row -> bool; True rows get a red edge and a pale red
    tint (e.g. a customer with a loan behind on payment)."""
    heat = {c["key"]: heat_range(df[c["key"]].tolist()) for c in cols if c.get("heat") and c["key"] in df.columns}
    head = "".join(
        f'<th style="text-align:{_align(c)};">{_esc(c.get("label", c["key"]))}'
        f'{_info(c.get("help"))}</th>' for c in cols)
    rows = []
    frames = [(r, False) for _, r in df.iterrows()] + ([(pd.Series(total), True)] if total else [])
    for r, is_total in frames:
        tds = ""
        flagged = not is_total and highlight is not None and bool(highlight(r))
        for i, c in enumerate(cols):
            kind, v = c.get("fmt", "text"), r.get(c["key"])
            style = f'text-align:{_align(c)};'
            if flagged:
                style += "background:#fff1f2;" + ("box-shadow:inset 4px 0 0 #dc2626;" if i == 0 else "")
            if c.get("bold"):
                style += "font-weight:700;"
            if not is_total and c["key"] in heat:
                style += heat_style(v, heat[c["key"]])
            if kind in _CHANGE_KINDS and not is_blank(v) and v != "" and float(v) != 0:
                good_up = c.get("good_if_up")
                worse = (float(v) > 0) != bool(good_up(r) if callable(good_up) else good_up)
                style += f'color:{"#dc2626" if worse else "#16a34a"};font-weight:700;'
            tds += f'<td style="{style}">{_fmt_cell(v, kind, r, c.get("count"), c.get("amount"))}</td>'
        rows.append(f'<tr class="{"ht-total" if is_total else ""}">{tds}</tr>')
    scroll = f"max-height:{max_height}px;overflow-y:auto;" if max_height else ""
    css = ("<style>.ht th{background:#111;color:#FFC000;padding:7px 10px;font-size:11px;white-space:nowrap;"
           "position:sticky;top:0;z-index:1;}.ht td{padding:6px 10px;font-size:12.5px;border-bottom:1px solid #f0f0f0;"
           "background:#fff;white-space:nowrap;}.ht tr.ht-total td{font-weight:800;border-top:2px solid #FFC000;"
           "background:#fffbea;}</style>")
    return (f'{css}<div style="overflow-x:auto;{scroll}border-radius:10px;border:1px solid #e5e7eb;">'
            f'<table class="ht" style="width:100%;border-collapse:collapse;font-family:Inter,sans-serif;">'
            f'<thead><tr>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>')


def _info(text: str | None) -> str:
    if not text:
        return ""
    return (f'<span title="{_esc(text)}" style="cursor:help;color:#fde68a;font-weight:400;'
            f'font-size:0.9em;margin-left:3px;">&#9432;</span>')


# ── Risk heat-map shading ────────────────────────────────────────────────────
# Risk measures (NPA, SMA, Delinquent, Hard Bucket, Roll Fwd...) have no "safe"
# level, so they're never green: each cell is shaded from pale pink (lowest in
# its column) to red (highest). Zero or missing gets no shade. Text stays dark.
_HEAT_LOW, _HEAT_HIGH = (254, 242, 242), (248, 113, 113)  # #fef2f2 -> #f87171
HEAT_TEXT = "#7f1d1d"


def _is_num(v) -> bool:
    try:
        return v is not None and not pd.isna(v) and float(v) == float(v)
    except (TypeError, ValueError):
        return False


def heat_range(values) -> tuple[float, float] | None:
    """(lowest, highest) of the positive values in a column, or None if none."""
    pos = [float(v) for v in values if _is_num(v) and float(v) > 0]
    return (min(pos), max(pos)) if pos else None


def heat_bg(val, rng: tuple[float, float] | None) -> str | None:
    """Background hex for `val` within its column's `rng`, or None (no shade)."""
    if rng is None or not _is_num(val) or float(val) <= 0:
        return None
    lo, hi = rng
    t = 0.5 if hi <= lo else min(max((float(val) - lo) / (hi - lo), 0.0), 1.0)
    return "#" + "".join(f"{round(a + (b - a) * t):02x}" for a, b in zip(_HEAT_LOW, _HEAT_HIGH))


def heat_style(val, rng: tuple[float, float] | None) -> str:
    """Inline CSS for a shaded cell ("" when unshaded)."""
    bg = heat_bg(val, rng)
    return f"background:{bg};color:{HEAT_TEXT};font-weight:600;" if bg else ""


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
    data_fixes: list[str] = []
    for f in files:
        df, errs = load_and_validate(f)
        if errs:
            errors.append(f"{getattr(f, 'name', 'file')}: {errs[0]}")
        else:
            dropped_dupes += df.attrs.get("dropped_duplicate_loans", 0)
            prefix = f"{getattr(f, 'name', 'file')}: " if len(files) > 1 else ""
            data_fixes += [prefix + n for n in df.attrs.get("data_fixes", [])]
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
    # Recount across ALL files: a customer's loans can sit in different
    # regional files, and each file only counted its own.
    combined = add_customer_loan_count(combined)
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
    combined.attrs["data_fixes"] = data_fixes
    return combined, errors
