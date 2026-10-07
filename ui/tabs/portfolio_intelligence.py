"""
Portfolio Intelligence tab: one question at a time. A row of views at the
top (Regions & Branches, Executives, Segments, Exposure, Action Lists); each
shows a one-line takeaway, ONE table in the shared format (a % and its count
in one cell, red shading on risk), optional detail collapsed underneath, and
a download. Overview, alerts and early warning live on the Dashboard, Alerts
and Migration tabs. Zero AI calls.
"""
import pandas as pd
import streamlit as st

from ui.components import (
    _dl_btn, _safe_df, _chart_card, append_total_row, _esc, heat_range, heat_style, html_table, pct_count,
    section_label,
)
from config import (
    FLEET_MIN_LOANS, REPOSSESSION_WINDOW_MONTHS, GOOD_CUSTOMER_MIN_TENURE_PCT, GOOD_CUSTOMER_MIN_LCC_PCT,
    LARGE_CUSTOMER_MIN_SOH_CR,
)

VIEWS = ["Regions & Branches", "Executives", "Segments", "Exposure", "Action Lists"]
VIEW_KEY = "pi_view"
_TOP_CHOICES = [10, 20, 50, 100, "All"]


# ── Kept helpers (also used by the Business tab) ─────────────────────────────

def _render_overdue_demand(scorecard_data: dict) -> None:
    from analysis.portfolio_intelligence import compute_overdue_demand_chart, OVERDUE_DEMAND_IDENTITY_COLS

    section_label("Overdue vs Month Demand Collection", margin_top="6px")
    st.caption(
        "A payment clears last month's carried-over overdue FIRST; only what's left over "
        "counts against this month's own EMI demand. 100% means nothing was outstanding on "
        "that side to begin with, not that nothing was collected."
    )

    if not scorecard_data:
        st.info("No data available.")
        return

    dim_tabs_avail = []
    if not scorecard_data.get("region", pd.DataFrame()).empty:    dim_tabs_avail.append(("Region",    "Region",    "region"))
    if not scorecard_data.get("branch", pd.DataFrame()).empty:    dim_tabs_avail.append(("Branch",    "Branch",    "branch"))
    if not scorecard_data.get("executive", pd.DataFrame()).empty: dim_tabs_avail.append(("Executive", "Executive", "executive"))

    if not dim_tabs_avail:
        st.info("No dimension data found.")
        return

    _IDENTITY_COLS = OVERDUE_DEMAND_IDENTITY_COLS
    _PCT_COLS = {"Overdue Collection %", "Month Demand Collection %", "Overall Collection %"}
    _CR_COLS = {"Overdue (Cr)", "Overdue Collection (Cr)", "Month Demand (Cr)", "Month Demand Collection (Cr)", "Overall Collection (Cr)"}

    dim_sub = st.tabs([t[0] for t in dim_tabs_avail])
    for tab, (label, col, key) in zip(dim_sub, dim_tabs_avail):
        with tab:
            df = scorecard_data[key]
            _chart_card(compute_overdue_demand_chart(df, col, label))

            identity_cols = _IDENTITY_COLS.get(key, [])
            show_cols = [
                col, *identity_cols, "Accounts",
                "Overdue (Cr)", "Overdue Collection (Cr)", "Overdue Collection %",
                "Month Demand (Cr)", "Month Demand Collection (Cr)", "Month Demand Collection %",
                "Overall Collection (Cr)", "Overall Collection %",
            ]
            show_cols = [c for c in show_cols if c in df.columns]
            th = "".join(
                f'<th style="background:#111;color:#FFC000;padding:6px 10px;font-size:11px;'
                f'text-align:{"left" if c in (col, *identity_cols) else "center"};white-space:nowrap;">{c}</th>'
                for c in show_cols
            )
            rows_html = ""
            _ratio_cols = {
                "Overdue Collection %": ("Overdue Collection (Cr)", "Overdue (Cr)"),
                "Month Demand Collection %": ("Month Demand Collection (Cr)", "Month Demand (Cr)"),
            }
            df_display = append_total_row(df[show_cols], ratio_cols=_ratio_cols)
            n_data_rows = len(df)
            for i, row in df_display.iterrows():
                if i == n_data_rows:
                    cells = "".join(
                        f'<td style="padding:6px 10px;font-size:12px;font-weight:800;border-top:2px solid #FFC000;'
                        f'text-align:{"left" if c in (col, *identity_cols) else "center"};">'
                        f'{f"{row[c]:.2f}%" if c in _PCT_COLS and row[c] != "" else _esc(row[c])}</td>'
                        for c in show_cols
                    )
                    rows_html += f'<tr style="background:#fffbea;">{cells}</tr>'
                    continue
                cells = ""
                for c in show_cols:
                    val = row[c]
                    align = "left" if c in (col, *identity_cols) else "center"
                    style = f"padding:6px 10px;font-size:12px;text-align:{align};"
                    if c == col:
                        cells += f'<td style="{style}font-weight:700;">{_esc(val)}</td>'
                    elif c in identity_cols:
                        cells += f'<td style="{style}">{_esc(val) if val is not None else " - "}</td>'
                    elif c in _PCT_COLS:
                        color = "#16a34a" if val >= 90 else ("#d97706" if val >= 60 else "#dc2626")
                        cells += f'<td style="{style}color:{color};font-weight:700;">{val:.2f}%</td>'
                    elif c in _CR_COLS:
                        cells += f'<td style="{style}">₹{val:.2f}Cr</td>'
                    else:
                        cells += f'<td style="{style}">{int(val):,}</td>'
                rows_html += f'<tr style="border-bottom:1px solid #f0f0f0;">{cells}</tr>'

            st.markdown(
                f'<div style="overflow-x:auto;border-radius:8px;border:1px solid #e5e7eb;">'
                f'<table style="width:100%;border-collapse:collapse;font-family:Inter,sans-serif;">'
                f'<thead><tr>{th}</tr></thead><tbody>{rows_html}</tbody>'
                f'</table></div>',
                unsafe_allow_html=True,
            )
            _dl_btn(df, f"overdue_demand_{key}.xlsx", f"dl_overdue_demand_{key}")


def _render_product_table(df: pd.DataFrame, npa_col: str = "NPA%") -> None:
    if df.empty:
        return
    # A % and its count share one cell ("11.9% (37)"); the count columns
    # aren't shown on their own.
    counts = {p: c for p, c in ((npa_col, "NPA Count"), ("SMA-2%", "SMA-2 Count")) if p in df.columns and c in df.columns}
    headers = [c for c in df.columns if c not in counts.values()]
    # position:sticky keeps the header row visible while scrolling a long
    # table (e.g. Sourcing Channel routinely has 1,000+ rows on a real file)
    # -- needs its own background since sticky content scrolls underneath it.
    th = "".join(
        f'<th style="background:#111;color:#FFC000;padding:6px 10px;font-size:11px;'
        f'text-align:{"left" if i == 0 else "right"};white-space:nowrap;'
        f'position:sticky;top:0;z-index:1;">{h}</th>'
        for i, h in enumerate(headers)
    )
    rows_html = ""
    df_display = append_total_row(df)
    n_data_rows = len(df)
    heat = {c: heat_range(df[c].tolist()) for c in (npa_col, "NPA% (SOH)", "SMA-2%") if c in df.columns}
    for i, row in df_display.iterrows():
        if i == n_data_rows:
            cells = "".join(
                f'<td style="padding:6px 10px;font-size:12px;text-align:{"left" if j == 0 else "right"};'
                f'font-weight:800;border-top:2px solid #FFC000;">'
                f'{f"{row[c]:,}" if isinstance(row[c], (int, float)) and row[c] != "" else _esc(row[c])}</td>'
                for j, c in enumerate(headers)
            )
            rows_html += f'<tr style="background:#fffbea;">{cells}</tr>'
            continue
        cells = ""
        for i2, col in enumerate(headers):
            val = row[col]
            align = "left" if i2 == 0 else "right"
            style = f"padding:6px 10px;font-size:12px;text-align:{align};"
            if col in heat:
                shown = pct_count(val, row.get(counts[col])) if col in counts else f"{val:.1f}%"
                cells += f'<td style="{style}{heat_style(val, heat[col])}">{shown}</td>'
            elif isinstance(val, float) and "%" in col:
                cells += f'<td style="{style}">{val:.1f}%</td>'
            elif isinstance(val, float) and any(k in col for k in ("SOH", "Loan", "Avg")):
                cells += f'<td style="{style}">₹{val:.2f}</td>'
            elif isinstance(val, float):
                cells += f'<td style="{style}">{val:.2f}</td>'
            elif isinstance(val, int):
                cells += f'<td style="{style}">{val:,}</td>'
            else:
                cells += f'<td style="{style}">{_esc(val)}</td>'
        rows_html += f'<tr style="border-bottom:1px solid #f0f0f0;">{cells}</tr>'

    st.markdown(
        f'<div style="overflow-x:auto;overflow-y:auto;max-height:520px;'
        f'border-radius:8px;border:1px solid #e5e7eb;">'
        f'<table style="width:100%;border-collapse:collapse;font-family:Inter,sans-serif;">'
        f'<thead><tr>{th}</tr></thead><tbody>{rows_html}</tbody>'
        f'</table></div>',
        unsafe_allow_html=True,
    )


def _roll_vintage(df: pd.DataFrame, granularity: str) -> pd.DataFrame:
    """Roll monthly cohorts up to Quarter / Half-Year / Year / Financial Year
    and recompute NPA% + SMA-2%. Uses analysis/portfolio_intelligence.py's
    _period_label/_period_sort_key -- the SAME definitions the New Advances
    Trend chart's own rollup (roll_new_advances_trend) uses, so the Business
    tab's two granularity pickers can't silently diverge into two different
    "Quarterly"/"Financial Year" meanings."""
    from analysis.new_business import _period_label, _period_sort_key

    if granularity == "Monthly" or df.empty:
        return df

    df = df.copy()
    df["_period"] = df["Disbursement Month"].apply(lambda m: _period_label(m, granularity))

    agg = (
        df.groupby("_period", sort=False)
        .agg(
            Accounts=("Accounts", "sum"),
            **{"NPA Count":   ("NPA Count",   "sum")},
            **{"SMA-2 Count": ("SMA-2 Count", "sum")},
            **{"SOH (Cr)":    ("SOH (Cr)",    "sum")},
        )
        .reset_index()
        .rename(columns={"_period": "Disbursement Month"})
    )
    agg["NPA%"]   = (agg["NPA Count"]   / agg["Accounts"] * 100).round(2)
    agg["SMA-2%"] = (agg["SMA-2 Count"] / agg["Accounts"] * 100).round(2)

    agg["_sk"] = agg["Disbursement Month"].apply(_period_sort_key)
    return agg.sort_values("_sk").drop(columns=["_sk"]).reset_index(drop=True)


# ── Cached data for the views (one entry per filter selection) ───────────────

@st.cache_data(show_spinner=False, max_entries=24)
def _cached_units(_c, _p, data_version: int, filter_key: str, grain: str) -> pd.DataFrame:
    from analysis.summary import unit_table
    return unit_table(_c, _p, grain)


@st.cache_data(show_spinner=False, max_entries=24)
def _cached_totals(_c, _p, data_version: int, filter_key: str, branch: str = "All") -> dict:
    """The Total row: the whole view (or one branch) as one unit -- sum over
    sum, never an average of rows -- with its own change against last month."""
    from analysis.summary import portfolio_total
    from utils import _unit_key
    if branch != "All":
        _c = _c[_c["Unit"].astype(str) == branch]
        _p = _p[_p["Unit"].map(_unit_key) == _unit_key(branch)] if len(_p) and "Unit" in _p.columns else _p
    return portfolio_total(_c, _p)


@st.cache_data(show_spinner=False, max_entries=32)
def _cached_segments(_c, data_version: int, filter_key: str, dim_col: str, breakdown: str, branch: str) -> pd.DataFrame:
    from utils import unit_metrics
    d = _c if branch == "All" else _c[_c["Unit"].astype(str) == branch]
    by = {"Together": [dim_col], "By branch": [dim_col, "Unit"], "By executive": [dim_col, "MNT NAME", "Unit"]}[breakdown]
    by = [c for c in by if c in d.columns]
    m = unit_metrics(d, by, min_accounts=1 if breakdown == "Together" else 3)
    return m.rename(columns={"Unit": "Branch", "MNT NAME": "Executive", dim_col: "Name"}) if not m.empty else m


@st.cache_data(show_spinner=False, max_entries=16)
def _cached_delinquent(_c, data_version: int, filter_key: str) -> pd.DataFrame:
    from analysis.exposure import compute_top_accounts
    return compute_top_accounts(_c, n=len(_c))[0]


@st.cache_data(show_spinner=False, max_entries=16)
def _cached_large_customers(_c, data_version: int, filter_key: str, min_cr: float) -> dict:
    from analysis.exposure import large_customers
    return large_customers(_c, min_cr)


# Each view computes its own charts and lists, only when it's open.
@st.cache_data(show_spinner=False, max_entries=16)
def _cached_quadrant(_c, _p, data_version: int, filter_key: str):
    from analysis.portfolio_intelligence import compute_branch_quadrant
    return compute_branch_quadrant(_c, _p)[1]


@st.cache_data(show_spinner=False, max_entries=16)
def _cached_overdue(_c, data_version: int, filter_key: str) -> dict:
    from analysis.portfolio_intelligence import compute_overdue_demand_scorecard
    return compute_overdue_demand_scorecard(_c)


@st.cache_data(show_spinner=False, max_entries=16)
def _cached_treemap(_c, data_version: int, filter_key: str):
    from analysis.exposure import compute_concentration_treemap
    return compute_concentration_treemap(_c)


@st.cache_data(show_spinner=False, max_entries=16)
def _cached_repossession(_c, data_version: int, filter_key: str, curr_month: str) -> pd.DataFrame:
    from analysis.exposure import compute_repossession_list
    return compute_repossession_list(_c, as_of=curr_month)


@st.cache_data(show_spinner=False, max_entries=16)
def _cached_good_customers(_c, data_version: int, filter_key: str) -> pd.DataFrame:
    from analysis.exposure import compute_good_customers
    return compute_good_customers(_c)


@st.cache_data(show_spinner=False, max_entries=16)
def _cached_fleet(_c, data_version: int, filter_key: str) -> dict:
    from analysis.exposure import compute_fleet_exposure
    return compute_fleet_exposure(_c, top_n=None)


# ── Shared pieces ────────────────────────────────────────────────────────────

def _with_total_row(df: pd.DataFrame, total: dict | None) -> pd.DataFrame:
    """The table as downloaded: its rows plus the same Total row as on screen."""
    if not total:
        return df
    return pd.concat([df, pd.DataFrame([{k: v for k, v in total.items() if k in df.columns}])], ignore_index=True)


def _takeaway(text: str) -> None:
    st.markdown(f'<div style="border-left:4px solid #FFC000;background:#fffbea;padding:8px 12px;'
                f'border-radius:6px;font-size:13.5px;color:#111827;margin:4px 0 10px 0;">{text}</div>',
                unsafe_allow_html=True)


def _move(v) -> str:
    if v is None or pd.isna(v):
        return ""
    return "unchanged" if v == 0 else f"{'up' if v > 0 else 'down'} {abs(v):.1f} pts"


def _list_controls(key: str, df: pd.DataFrame, branch_col: str = "Branch", noun: str = "rows") -> pd.DataFrame:
    """Top X, a branch filter and "top X in each branch", for any list
    already sorted with the most important rows first."""
    c1, c2, c3 = st.columns([1, 1.4, 1.6])
    n = c1.selectbox("Show top", _TOP_CHOICES, index=0, key=f"{key}_n")
    branches = ["All"] + (sorted(df[branch_col].dropna().astype(str).unique()) if branch_col in df.columns else [])
    b = c2.selectbox("Branch", branches, key=f"{key}_b")
    c3.markdown('<div style="height:28px;"></div>', unsafe_allow_html=True)
    each = c3.checkbox(f"Top {n} in each branch" if n != "All" else "Group by branch", key=f"{key}_each",
                       disabled=b != "All" or branch_col not in df.columns)
    d = df if b == "All" else df[df[branch_col].astype(str) == b]
    if each and b == "All" and branch_col in d.columns:
        d = d.groupby(branch_col, sort=True, group_keys=False).head(n if n != "All" else len(d))
        d = d.sort_values(branch_col, kind="stable")
    elif n != "All":
        d = d.head(n)
    st.caption(f"Showing {len(d):,} of {len(df):,} {noun}.")
    return d


_RENAME = {"RegionName": "Region", "Unit": "Branch", "MNT NAME": "Executive", "curr_bucket": "Bucket",
           "SegmentName": "Segment", "Cust Name": "Customer", "Cust Mob No": "Mobile"}


# ── View 1: Regions & Branches ───────────────────────────────────────────────

_SORTS = {  # label -> (column, ascending)
    "Delinquency % (worst first)": ("Delinquency%", False),
    "Change in delinquency (biggest rise)": ("Δ Delinquency%", False),
    "NPA % (worst first)": ("NPA%", False),
    "NPA % by SOH (worst first)": ("NPA% (SOH)", False),
    "SMA-2 % (worst first)": ("SMA-2%", False),
    "Collection % (lowest first)": ("Collection%", True),
    "Accounts (largest first)": ("Accounts", False),
}


def unit_columns(name: str, extra: list[str] = ()) -> list[dict]:
    """The shared column set for a region / branch / executive table."""
    return [
        {"key": name, "bold": True},
        *[{"key": e} for e in extra],
        {"key": "Accounts", "fmt": "int"},
        {"key": "Delinquency%", "label": "Delinquent", "fmt": "pct_count", "count": "Delinquent", "heat": True,
         "help": "Loans with any EMI or charge overdue: % of accounts (number of loans)."},
        {"key": "Δ Delinquency%", "label": "Change vs last month", "fmt": "pp"},
        {"key": "SMA-2%", "label": "SMA-2", "fmt": "pct_count", "count": "SMA-2", "heat": True},
        {"key": "NPA%", "label": "NPA", "fmt": "pct_count", "count": "NPA", "heat": True},
        {"key": "NPA% (SOH)", "label": "NPA by SOH", "fmt": "pct_cr", "amount": "NPA SOH (Cr)", "heat": True,
         "help": "SOH of NPA loans as % of total SOH (₹ Cr of NPA SOH)."},
        {"key": "Hard Bucket%", "label": "Hard Bucket", "fmt": "pct_count", "count": "Hard Bucket", "heat": True},
        {"key": "Collection%", "label": "Collection", "fmt": "pct"},
        {"key": "Strike%", "label": "Strike", "fmt": "pct"},
        {"key": "Roll Fwd%", "label": "Rolled forward", "fmt": "pct_count", "count": "Slipped", "heat": True,
         "help": "Loans in a worse bucket than last month: % of loans with both months' bucket (number of loans)."},
    ]


def _total_row(name: str, t: dict, label: str = "Total") -> dict:
    if not t:
        return {}
    return {name: label, "Accounts": t["Accounts"], "Delinquency%": t["Delinquency%"], "Delinquent": t["Delinquent"],
            "Δ Delinquency%": t.get("Δ Delinquency%"), "SOH (Cr)": t.get("SOH (Cr)"),
            "SMA-2%": t["SMA-2%"], "SMA-2": t["SMA-2"], "NPA%": t["NPA%"], "NPA": t["NPA"],
            "NPA% (SOH)": t["NPA% (SOH)"], "NPA SOH (Cr)": t.get("NPA SOH (Cr)"), "Hard Bucket%": t["Hard Bucket%"], "Hard Bucket": t["Hard Bucket"],
            "Collection%": t["Collection%"], "Strike%": t["Strike%"],
            "Roll Fwd%": t.get("Roll Fwd%"), "Slipped": t.get("Slipped")}


def units_table_html(df: pd.DataFrame, name: str, total: dict | None = None, extra=()) -> str:
    cols = [c for c in unit_columns(name, list(extra)) if c["key"] in df.columns]
    return html_table(df, cols, total=total, max_height=560 if len(df) > 15 else None)


def _units_takeaway(df: pd.DataFrame, name: str, sort_col: str) -> str:
    top = df.iloc[0]
    noun = name.lower() + ("es" if name == "Branch" else "s")
    text = f"<b>{_esc(top[name])}</b> is first on this list: delinquency {top['Delinquency%']:.1f}% " \
           f"({int(top['Delinquent']):,} loans)"
    if pd.notna(top.get("Δ Delinquency%")):
        text += f", {_move(top['Δ Delinquency%'])} on last month"
    text += f"; NPA {top['NPA%']:.1f}% ({int(top['NPA']):,})."
    if "Δ Delinquency%" in df.columns and df["Δ Delinquency%"].notna().any():
        worse = int((df["Δ Delinquency%"] > 0).sum())
        text += f" {worse} of {len(df)} {noun} got worse since last month."
    return text


def _view_units(c, p, data_version, filter_key) -> None:
    a, b = st.columns([1, 2])
    level = a.radio("Show", ["Region", "Branch"], horizontal=True, key="pi_level")
    sort = b.selectbox("Sort by", list(_SORTS), key="pi_units_sort")
    df = _cached_units(c, p, data_version, filter_key, level)
    if df.empty:
        st.info("No region/branch columns in this file.")
        return
    col, asc = _SORTS[sort]
    df = df.sort_values(col, ascending=asc, na_position="last", kind="stable").reset_index(drop=True)
    _takeaway(_units_takeaway(df, level, col))
    total = _total_row(level, _cached_totals(c, p, data_version, filter_key))
    st.markdown(units_table_html(df, level, total, extra=["Region"] if level == "Branch" else []), unsafe_allow_html=True)
    _dl_btn(_with_total_row(df, total), f"{level.lower()}_table.xlsx", f"dl_pi_units_{level}")
    if level == "Branch":
        with st.expander("Chart: Collection % vs NPA % by branch (bubble = SOH)"):
            _chart_card(_cached_quadrant(c, p, data_version, filter_key))
            st.caption("Top-left (high NPA, low collection) needs action first. Dashed lines = portfolio middle.")
    overdue = _cached_overdue(c, data_version, filter_key)
    if overdue:
        with st.expander("Collections detail: overdue vs this month's demand"):
            _render_overdue_demand(overdue)


# ── View 2: Executives ───────────────────────────────────────────────────────

def _view_executives(c, p, data_version, filter_key) -> None:
    df = _cached_units(c, p, data_version, filter_key, "Executive")
    if df.empty:
        st.info("No executive column (MNT NAME) in this file.")
        return
    a, b = st.columns([1, 2])
    branch = a.selectbox("Branch", ["All"] + sorted(df["Branch"].astype(str).unique()), key="pi_exec_branch")
    sorts = {**_SORTS, "Most rolled forward": ("Slipped", False), "Most rescued": ("Rescued", False)}
    sort = b.selectbox("Sort by", [k for k in sorts if sorts[k][0] in df.columns], key="pi_exec_sort")
    if branch != "All":
        df = df[df["Branch"].astype(str) == branch]
    col, asc = sorts[sort]
    df = df.sort_values(col, ascending=asc, na_position="last", kind="stable").reset_index(drop=True)
    _takeaway(_units_takeaway(df, "Executive", col))
    cols = unit_columns("Executive", ["Branch", "Region"])
    if "Rescued" in df.columns:
        cols.append({"key": "Rescued", "label": "Rescued", "fmt": "int",
                     "help": "Loans moved from SMA-1/SMA-2/NPA to a better bucket since last month."})
    t = _cached_totals(c, p, data_version, filter_key, branch)       # every loan in view, not just listed rows
    total = _total_row("Executive", t, "Total" if branch == "All" else f"Total ({branch})")
    if total and "Rescued" in df.columns:
        total["Rescued"] = t.get("Rescued")
    st.markdown(html_table(df, [x for x in cols if x["key"] in df.columns], total=total,
                           max_height=560 if len(df) > 15 else None), unsafe_allow_html=True)
    st.caption("Executives with fewer than the minimum number of loans are left out (config.MIN_ACCOUNTS_EXECUTIVE).")
    _dl_btn(_with_total_row(df, total), "executives.xlsx", "dl_pi_exec")


# ── View 3: Segments ─────────────────────────────────────────────────────────

def _view_segments(c, data_version, filter_key) -> None:
    from utils import segment_column
    dims = {}
    seg = segment_column(c)
    if seg:
        dims["Segment"] = seg
    if "FUEL_TYPE" in c.columns:
        dims["Fuel type"] = "FUEL_TYPE"
    if not dims:
        st.info("No segment or fuel type column in this file.")
        return
    a, b, d = st.columns([1, 1.6, 1.2])
    dim = a.radio("By", list(dims), horizontal=True, key="pi_seg_dim")
    breakdown = b.radio("Break down", ["Together", "By branch", "By executive"], horizontal=True, key="pi_seg_breakdown")
    branch = d.selectbox("Branch", ["All"] + sorted(c["Unit"].dropna().astype(str).unique()), key="pi_seg_branch")
    df = _cached_segments(c, data_version, filter_key, dims[dim], breakdown, branch)
    if df.empty:
        st.info("No loans for this selection.")
        return
    sorts = {"NPA % (worst first)": ("NPA%", False), "Delinquency % (worst first)": ("Delinquency%", False),
             "Accounts (largest first)": ("Accounts", False), "SOH (largest first)": ("SOH (Cr)", False)}
    sort = st.selectbox("Sort by", list(sorts), key="pi_seg_sort")
    col, asc = sorts[sort]
    df = df.sort_values(col, ascending=asc, kind="stable").reset_index(drop=True)
    top = df.iloc[0]
    where = "".join(f" in {_esc(top[k])}" for k in ("Branch",) if k in df.columns) + \
        (f" ({_esc(top['Executive'])})" if "Executive" in df.columns else "")
    _takeaway(f"<b>{_esc(top['Name'])}</b>{where} is first on this list: NPA {top['NPA%']:.1f}% "
              f"({int(top['NPA']):,} of {int(top['Accounts']):,} loans), delinquency {top['Delinquency%']:.1f}%.")
    cols = [{"key": "Name", "label": dim, "bold": True},
            *([{"key": "Branch"}] if "Branch" in df.columns else []),
            *([{"key": "Executive"}] if "Executive" in df.columns else []),
            {"key": "Accounts", "fmt": "int"},
            {"key": "Delinquency%", "label": "Delinquent", "fmt": "pct_count", "count": "Delinquent", "heat": True},
            {"key": "SMA-2%", "label": "SMA-2", "fmt": "pct_count", "count": "SMA-2", "heat": True},
            {"key": "NPA%", "label": "NPA", "fmt": "pct_count", "count": "NPA", "heat": True},
            {"key": "NPA% (SOH)", "label": "NPA by SOH", "fmt": "pct_cr", "amount": "NPA SOH (Cr)", "heat": True},
            {"key": "Collection%", "label": "Collection", "fmt": "pct"},
            {"key": "SOH (Cr)", "label": "SOH", "fmt": "cr"}]
    t = _cached_totals(c, pd.DataFrame(), data_version, filter_key, branch)
    total = {"Name": "Total" if branch == "All" else f"Total ({branch})", **{k: t.get(k) for k in (
        "Accounts", "Delinquency%", "Delinquent", "SMA-2%", "SMA-2", "NPA%", "NPA", "NPA% (SOH)", "NPA SOH (Cr)", "Collection%", "SOH (Cr)")}} if t else None
    st.markdown(html_table(df, cols, total=total, max_height=560 if len(df) > 15 else None), unsafe_allow_html=True)
    if breakdown != "Together":
        st.caption("Groups with fewer than 3 loans are left out.")
    keep = [k for k in ("Name", "Branch", "Executive", "Accounts", "Delinquent", "Delinquency%", "SMA-2", "SMA-2%",
                        "NPA", "NPA%", "NPA% (SOH)", "NPA SOH (Cr)", "Collection%", "SOH (Cr)") if k in df.columns]
    _dl_btn(_with_total_row(df[keep], total).rename(columns={"Name": dim}),
            f"{dim.lower().replace(' ', '_')}_table.xlsx", "dl_pi_segments")


# ── View 4: Exposure ─────────────────────────────────────────────────────────

_BUCKET_FILTERS = {"All delinquent": None, "SMA-1 and worse": ["SMA-1", "SMA-2", "NPA"],
                   "SMA-2 and worse": ["SMA-2", "NPA"], "NPA only": ["NPA"]}


def _view_exposure(c, data_version, filter_key) -> None:
    what = st.radio("Show", ["Large customers", "Largest delinquent loans", "Fleet operators", "Concentration map"],
                    horizontal=True, key="pi_exp_what")
    if what == "Large customers":
        _view_large_customers(c, data_version, filter_key)
    elif what == "Largest delinquent loans":
        df = _cached_delinquent(c, data_version, filter_key)
        if df.empty:
            st.info("No delinquent loans.")
            return
        flt = st.selectbox("Buckets", list(_BUCKET_FILTERS), key="pi_exp_bucket")
        if _BUCKET_FILTERS[flt]:
            df = df[df["curr_bucket"].isin(_BUCKET_FILTERS[flt])]
        df = df.rename(columns=_RENAME)
        view = _list_controls("pi_top_loans", df, "Branch", "delinquent loans")
        book = float(pd.to_numeric(c.get("SOH"), errors="coerce").sum()) if "SOH" in c.columns else 0.0
        soh = float(view["SOH"].sum())
        _takeaway(f"These {len(view):,} loans hold <b>₹{soh / 1e7:,.2f} Cr</b> SOH "
                  f"({soh / book * 100 if book else 0:.1f}% of the whole book); "
                  f"{int((view['Bucket'] == 'NPA').sum()):,} of them are already NPA.")
        st.dataframe(_safe_df(view), use_container_width=True, hide_index=True)
        _dl_btn(view.rename(columns={v: k for k, v in _RENAME.items() if v in view.columns and k in c.columns}),
                "largest_delinquent_loans.xlsx", "dl_pi_top_loans", full_source=c)
    elif what == "Fleet operators":
        fl = _cached_fleet(c, data_version, filter_key)
        if not fl.get("count"):
            st.info(f"No fleet operators (customers with {FLEET_MIN_LOANS}+ loans, matched by mobile number).")
            return
        _takeaway(f"<b>{fl['count']:,}</b> fleet operators (customers with {FLEET_MIN_LOANS}+ loans) hold "
                  f"<b>₹{fl['total_soh_cr']:,.2f} Cr</b> SOH; {fl['npa_operators']:,} of them have at least one NPA loan.")
        df = fl["top_df"].rename(columns={"Unit": "Branch"})
        view = _list_controls("pi_fleet", df, "Branch", "fleet operators")
        st.dataframe(_safe_df(view), use_container_width=True, hide_index=True)
        if fl.get("excluded_blank_mobile_loans"):
            st.caption(f"{fl['excluded_blank_mobile_loans']:,} loans with no mobile number can't be grouped by customer.")
        _dl_btn(view, "fleet_operators.xlsx", "dl_pi_fleet")
    else:
        _chart_card(_cached_treemap(c, data_version, filter_key))
        st.caption("Box size = SOH; colour = NPA % (pale = lower, dark red = higher). Click a region to see its branches.")


_CUSTOMER_COLS = [
    {"key": "Customer", "bold": True}, {"key": "Mobile"}, {"key": "Branches"},
    {"key": "Exposure (Cr)", "label": "Exposure", "fmt": "cr",
     "help": "The customer's total SOH across all their loans in the upload."},
    {"key": "Loans", "fmt": "int", "help": "Loans in view (the download also has Loans in Book: all of them)."},
    {"key": "Delinquent %", "label": "Delinquent loans", "fmt": "pct_count", "count": "Delinquent", "heat": True,
     "help": "Loans with any EMI or charge overdue: % of the customer's loans (number of loans)."},
    {"key": "Delinquent SOH %", "label": "Delinquent SOH", "fmt": "pct_cr", "amount": "Delinquent SOH (Cr)",
     "heat": True, "help": "SOH of the delinquent loans as % of the customer's SOH (₹ Cr)."},
    {"key": "NPA", "label": "NPA loans", "fmt": "int"},
    {"key": "Worst Bucket"},
    {"key": "Overdue (Closing Arrears)", "label": "Overdue", "fmt": "inr"},
]
_LOAN_MONEY = ("SOH", "POS", "Closing Arrears", "ARREARS AGAINST INST", "ARREARS AGAINST EXP", "Month Due-Inst",
               "Month Collection (Excluding Reserve Collection)", "Last Receipt Amount", "Loan Amount")


def _customer_total(cust: pd.DataFrame) -> dict:
    loans, delq = int(cust["Loans"].sum()), int(cust["Delinquent"].sum())
    soh, bad_soh = float(cust["SOH (Cr)"].sum()), float(cust["Delinquent SOH (Cr)"].sum())
    return {"Customer": "Total", "Exposure (Cr)": round(float(cust["Exposure (Cr)"].sum()), 2), "Loans": loans,
            "Delinquent": delq, "Delinquent %": round(delq / loans * 100, 2) if loans else 0.0,
            "Delinquent SOH (Cr)": round(bad_soh, 2), "Delinquent SOH %": round(bad_soh / soh * 100, 2) if soh else 0.0,
            "NPA": int(cust["NPA"].sum()), "Overdue (Closing Arrears)": float(cust["Overdue (Closing Arrears)"].sum())}


def _view_large_customers(c, data_version, filter_key) -> None:
    a, b, d = st.columns([1, 1.4, 1.6])
    min_cr = a.number_input("Exposure at least (₹ Cr)", min_value=0.1, value=float(LARGE_CUSTOMER_MIN_SOH_CR),
                            step=0.25, format="%.2f", key="pi_cust_min")
    r = _cached_large_customers(c, data_version, filter_key, float(min_cr))
    cust, loans = r["customers"], r["loans"]
    if cust.empty:
        st.info(f"No customer has ₹{min_cr:,.2f} Cr or more SOH. The largest customer exposure in this file is "
                f"₹{r['largest_cr']:,.2f} Cr; lower the amount to see the biggest customers.")
        return
    branches = sorted({x.strip() for v in cust["Branches"] for x in str(v).split(",") if x.strip()})
    branch = b.selectbox("Branch", ["All"] + branches, key="pi_cust_branch")
    d.markdown('<div style="height:28px;"></div>', unsafe_allow_html=True)
    only = d.checkbox("Only customers with a delinquent loan", key="pi_cust_only")
    if branch != "All":
        cust = cust[cust["Branches"].astype(str).str.split(", ").apply(lambda xs: branch in xs)]
    if only:
        cust = cust[cust["Has Delinquent Loan"]]
    if cust.empty:
        st.info("No customer matches these choices.")
        return

    total = _customer_total(cust)
    _takeaway(f"<b>{len(cust):,}</b> customers with ₹{min_cr:,.2f} Cr or more exposure hold "
              f"<b>₹{total['Exposure (Cr)']:,.2f} Cr</b>. <b>{int(cust['Has Delinquent Loan'].sum()):,}</b> of them "
              f"have a loan behind on payment, with <b>₹{total['Delinquent SOH (Cr)']:,.2f} Cr</b> SOH on those "
              f"loans; they are marked in red.")
    st.markdown(html_table(cust, _CUSTOMER_COLS, total=total, max_height=480 if len(cust) > 12 else None,
                           highlight=lambda row: bool(row.get("Has Delinquent Loan"))), unsafe_allow_html=True)
    st.caption("A customer is matched by mobile number, so two people sharing a number count as one. "
               "Exposure counts all of the customer's loans in the upload, even outside the sidebar filter.")
    keep = ["Customer", "Mobile", "Exposure (Cr)", "Loans in Book", "Loans", "Delinquent", "Delinquent %",
            "Delinquent SOH (Cr)", "Delinquent SOH %", "NPA", "SOH (Cr)", "Overdue (Closing Arrears)",
            "Worst Bucket", "Branches", "Regions", "Executives"]
    _dl_btn(_with_total_row(cust[keep], total), "large_customers.xlsx", "dl_pi_cust")

    # Loan level: every loan of the customers above, delinquent ones shaded.
    st.markdown('<div class="section-label" style="margin-top:18px;">Loans of these customers</div>',
                unsafe_allow_html=True)
    labels = {f"{row['Customer']} ({row['Mobile']}), ₹{row['Exposure (Cr)']:,.2f} Cr": str(row["Mobile"]).strip()
              for _, row in cust.iterrows()}
    x, y = st.columns([2.4, 1.6])
    pick = x.selectbox("Customer", ["All customers above", *labels], key="pi_cust_pick")
    y.markdown('<div style="height:28px;"></div>', unsafe_allow_html=True)
    only_bad = y.checkbox("Only delinquent loans", key="pi_cust_only_loans")
    mobs = set(labels.values()) if pick not in labels else {labels[pick]}
    lv = loans[loans["Cust Mob No"].astype(str).str.strip().isin(mobs)]
    if only_bad:
        lv = lv[lv["Delinquent"] == "Yes"]
    if lv.empty:
        st.info("No loans to show for this choice.")
        return
    view = lv.rename(columns=_RENAME)
    money = [k for k in _LOAN_MONEY if k in view.columns]
    red = "background-color:#fff1f2;color:#7f1d1d"
    st.dataframe(_safe_df(view).style
                 .apply(lambda row: [red if row.get("Delinquent") == "Yes" else ""] * len(row), axis=1)
                 .format({**{k: "{:,.0f}" for k in money},
                          **({"Arrears / EMI": "{:.2f}"} if "Arrears / EMI" in view.columns else {})}, na_rep=""),
                 use_container_width=True, hide_index=True, height=min(38 + 35 * len(view), 520))
    st.caption(f"{len(view):,} loans. Delinquent loans (any EMI or charge overdue) are shaded red and come first "
               "for each customer.")
    _dl_btn(lv, "large_customer_loans.xlsx", "dl_pi_cust_loans")


# ── View 5: Action lists ─────────────────────────────────────────────────────

def _view_actions(df_curr, curr_month, data_version, filter_key) -> None:
    what = st.radio("List", ["Repossession candidates", "Good customers"], horizontal=True, key="pi_act_what")
    if what == "Repossession candidates":
        repo_df = _cached_repossession(df_curr, data_version, filter_key, curr_month)
        if repo_df is None or repo_df.empty:
            st.info("No repossession candidates.")
            return
        sorts = {"Largest SOH": ("SOH", False), "Worst payers (lowest LCC%)": ("LCC%", True),
                 "Most recent loans": ("Ag_Date", False)}
        sort = st.selectbox("Sort by", [k for k, v in sorts.items() if v[0] in repo_df.columns], key="pi_repo_sort")
        col, asc = sorts[sort]
        df = repo_df.sort_values(col, ascending=asc, kind="stable").rename(columns=_RENAME)
        _takeaway(f"<b>{len(df):,}</b> SMA-2/NPA loans agreed in the last {REPOSSESSION_WINDOW_MONTHS} months "
                  f"(seized-and-sold excluded), <b>₹{df['SOH'].sum() / 1e7:,.2f} Cr</b> SOH: "
                  f"the vehicle still has value.")
        view = _list_controls("pi_repo", df, "Branch", "loans")
        st.dataframe(_safe_df(view), use_container_width=True, hide_index=True)
        _dl_btn(view.rename(columns={"Branch": "Unit", "Region": "RegionName", "Executive": "MNT NAME",
                                     "Bucket": "curr_bucket", "Customer": "Cust Name", "Mobile": "Cust Mob No"}),
                "repossession_candidates.xlsx", "dl_pi_repo", full_source=df_curr)
    else:
        good_df = _cached_good_customers(df_curr, data_version, filter_key)
        if good_df is None or good_df.empty:
            st.info("No good customers by these rules.")
            return
        sorts = {"Lowest SOH (easiest to refinance)": ("SOH", True),
                 "Most of the tenure done": ("Tenure Completed %", False)}
        sort = st.selectbox("Sort by", [k for k, v in sorts.items() if v[0] in good_df.columns], key="pi_good_sort")
        col, asc = sorts[sort]
        df = good_df.sort_values(col, ascending=asc, kind="stable").rename(columns=_RENAME)
        _takeaway(f"<b>{len(df):,}</b> customers have finished {GOOD_CUSTOMER_MIN_TENURE_PCT}%+ of their tenure "
                  f"and paid everything due (LCC {GOOD_CUSTOMER_MIN_LCC_PCT}%+): candidates for a top-up or a new loan.")
        view = _list_controls("pi_good", df, "Branch", "customers")
        st.dataframe(_safe_df(view), use_container_width=True, hide_index=True)
        _dl_btn(view.rename(columns={"Branch": "Unit", "Region": "RegionName", "Customer": "Cust Name"}),
                "good_customers.xlsx", "dl_pi_good", full_source=df_curr)


# ── The tab ──────────────────────────────────────────────────────────────────

def render_portfolio_intelligence_tab(df_curr: pd.DataFrame, df_prev: pd.DataFrame, curr_month: str,
                                      data_version: int = 0, filter_key: str = "") -> None:
    """One view at a time (VIEWS); each view computes only what it shows."""
    df_prev = df_prev if df_prev is not None else pd.DataFrame()
    if df_prev.empty:
        st.caption("Last month's file isn't loaded, so changes and roll rates are blank.")
    view = st.segmented_control("View", VIEWS, default=VIEWS[0], key=VIEW_KEY, label_visibility="collapsed")
    view = view or VIEWS[0]
    if view == "Regions & Branches":
        _view_units(df_curr, df_prev, data_version, filter_key)
    elif view == "Executives":
        _view_executives(df_curr, df_prev, data_version, filter_key)
    elif view == "Segments":
        _view_segments(df_curr, data_version, filter_key)
    elif view == "Exposure":
        _view_exposure(df_curr, data_version, filter_key)
    else:
        _view_actions(df_curr, curr_month, data_version, filter_key)
