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
    _dl_btn, _safe_df, _chart_card, _esc, html_table, section_label, takeaway, list_controls,
)
from config import FLEET_MIN_LOANS, LARGE_CUSTOMER_MIN_SOH_CR

VIEWS = ["Regions & Branches", "Executives", "Segments", "Exposure"]
VIEW_KEY = "pi_view"


# ── Kept helpers (also used by the Business tab) ─────────────────────────────

def _collection_cell(v, _row=None) -> str:
    """A collection % in green (90%+), amber (60%+) or red."""
    if v is None or v == "" or pd.isna(v):
        return '<span style="color:#9ca3af;">-</span>'
    color = "#16a34a" if v >= 90 else ("#d97706" if v >= 60 else "#dc2626")
    return f'<span style="color:{color};font-weight:700;">{float(v):.1f}%</span>'


def _render_overdue_demand(scorecard_data: dict) -> None:
    from analysis.portfolio_intelligence import compute_overdue_demand_chart, OVERDUE_DEMAND_IDENTITY_COLS

    section_label("Overdue vs Month Demand Collection", margin_top="6px")
    st.caption(
        "A payment clears last month's carried-over overdue FIRST; only what's left over "
        "counts against this month's own EMI demand. 100% means nothing was outstanding on "
        "that side to begin with, not that nothing was collected."
    )
    levels = [(lb, k) for lb, k in (("Zone", "zone"), ("Region", "region"), ("Branch", "branch"),
                                    ("Executive", "executive"))
              if not (scorecard_data or {}).get(k, pd.DataFrame()).empty]
    if not levels:
        st.info("No data available.")
        return
    for tab, (label, key) in zip(st.tabs([lb for lb, _ in levels]), levels):
        with tab:
            df = scorecard_data[key]
            _chart_card(compute_overdue_demand_chart(df, label, label))
            identity = OVERDUE_DEMAND_IDENTITY_COLS.get(key, [])
            sums = {c: round(float(df[c].sum()), 2) for c in (
                "Overdue (Cr)", "Overdue Collection (Cr)", "Month Demand (Cr)", "Month Demand Collection (Cr)",
                "Overall Collection (Cr)") if c in df.columns}
            ratio = lambda num, den: round(sums[num] / sums[den] * 100, 2) if sums.get(den) else None  # noqa: E731
            total = {label: "Total", "Accounts": int(df["Accounts"].sum()), **sums,
                     "Overdue Collection %": ratio("Overdue Collection (Cr)", "Overdue (Cr)"),
                     "Month Demand Collection %": ratio("Month Demand Collection (Cr)", "Month Demand (Cr)")}
            cols = [{"key": label, "bold": True}, *[{"key": c} for c in identity],
                    {"key": "Accounts", "fmt": "int"},
                    {"key": "Overdue (Cr)", "label": "Overdue", "fmt": "cr"},
                    {"key": "Overdue Collection (Cr)", "label": "Overdue collected", "fmt": "cr"},
                    {"key": "Overdue Collection %", "label": "Overdue %", "fmt": _collection_cell},
                    {"key": "Month Demand (Cr)", "label": "Month demand", "fmt": "cr"},
                    {"key": "Month Demand Collection (Cr)", "label": "Demand collected", "fmt": "cr"},
                    {"key": "Month Demand Collection %", "label": "Demand %", "fmt": _collection_cell},
                    {"key": "Overall Collection (Cr)", "label": "Collected in all", "fmt": "cr"},
                    {"key": "Overall Collection %", "label": "Overall %", "fmt": _collection_cell,
                     "help": "The unit's Collection %: collected / this month's net demand (blank in the Total row)."}]
            st.markdown(html_table(df, [c for c in cols if c["key"] in df.columns], total=total,
                                   max_height=480 if len(df) > 15 else None), unsafe_allow_html=True)
            _dl_btn(df, f"overdue_demand_{key}.xlsx", f"dl_overdue_demand_{key}")


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
    if dim_col in _BLANK_AS:      # a blank legal stage means the loan isn't in legal, not "unknown"
        d = d.assign(**{dim_col: d[dim_col].where(d[dim_col].astype(str).str.strip().ne("") & d[dim_col].notna(),
                                                  _BLANK_AS[dim_col])})
    if dim_col in _VALUE_LABELS:     # "Y"/"N" read as words
        labels = _VALUE_LABELS[dim_col]
        d = d.assign(**{dim_col: d[dim_col].map(lambda v: labels.get(str(v).strip().upper(), v))})
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
def _cached_fleet(_c, data_version: int, filter_key: str) -> dict:
    from analysis.exposure import compute_fleet_exposure
    return compute_fleet_exposure(_c, top_n=None)


# ── Shared pieces ────────────────────────────────────────────────────────────

def _with_total_row(df: pd.DataFrame, total: dict | None) -> pd.DataFrame:
    """The table as downloaded: its rows plus the same Total row as on screen."""
    if not total:
        return df
    return pd.concat([df, pd.DataFrame([{k: v for k, v in total.items() if k in df.columns}])], ignore_index=True)


def _move(v) -> str:
    if v is None or pd.isna(v):
        return ""
    return "unchanged" if v == 0 else f"{'up' if v > 0 else 'down'} {abs(v):.1f} pts"


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
    levels = (["Zone"] if "Zone" in c.columns else []) + ["Region", "Branch"]
    level = a.radio("Show", levels, horizontal=True, key="pi_level")
    sort = b.selectbox("Sort by", list(_SORTS), key="pi_units_sort")
    df = _cached_units(c, p, data_version, filter_key, level)
    if df.empty:
        st.info(f"No {level.lower()} column in this file.")
        return
    col, asc = _SORTS[sort]
    df = df.sort_values(col, ascending=asc, na_position="last", kind="stable").reset_index(drop=True)
    takeaway(_units_takeaway(df, level, col))
    total = _total_row(level, _cached_totals(c, p, data_version, filter_key))
    above = {"Region": ["Zone"], "Branch": ["Region"]}.get(level, [])      # the level above, when shown
    st.markdown(units_table_html(df, level, total, extra=[x for x in above if x in df.columns]), unsafe_allow_html=True)
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

_EXEC_SORTS = {
    "Collection % (best first)": ("Collection%", False),
    "Strike % (best first)": ("Strike%", False),
    "Strike % (lowest first)": ("Strike%", True),
    "Most rolled forward": ("Slipped", False),
    "Most rescued": ("Rescued", False),
}
_TIER_LABEL = {"top": "Top 25%", "mid": "Middle", "bottom": "Bottom 25%"}
_TIER_BY = {"Collection%": "collection %", "Strike%": "strike rate"}


def _tier_note(df: pd.DataFrame, tier_by: str) -> str:
    if "Tier" not in df.columns or df.empty:
        return ""
    n = df["Tier"].value_counts()
    return (f" By {_TIER_BY[tier_by]}: <b>{int(n.get(_TIER_LABEL['top'], 0)):,}</b> in the top quarter, "
            f"<b>{int(n.get(_TIER_LABEL['bottom'], 0)):,}</b> in the bottom quarter (marked red).")


def _view_executives(c, p, data_version, filter_key) -> None:
    df = _cached_units(c, p, data_version, filter_key, "Executive")
    if df.empty:
        st.info("No executive column (MNT NAME) in this file.")
        return
    a, b = st.columns([1, 2])
    branch = a.selectbox("Branch", ["All"] + sorted(df["Branch"].astype(str).unique()), key="pi_exec_branch")
    sorts = {**_SORTS, **_EXEC_SORTS}
    sort = b.selectbox("Sort by", [k for k in sorts if sorts[k][0] in df.columns], key="pi_exec_sort")
    if branch != "All":
        df = df[df["Branch"].astype(str) == branch]
    col, asc = sorts[sort]
    df = df.sort_values(col, ascending=asc, na_position="last", kind="stable").reset_index(drop=True)
    # Tier: quarter of the executives listed, by strike rate when sorted by it, else by collection.
    tier_by = "Strike%" if col == "Strike%" else "Collection%"
    if tier_by in df.columns and len(df):
        from analysis.executive_scorecard import quartile_tier
        df["Tier"] = quartile_tier(df[tier_by]).map(_TIER_LABEL)
    takeaway(_units_takeaway(df, "Executive", col) + _tier_note(df, tier_by))
    cols = [*unit_columns("Executive", ["Branch", "Region"])[:3],
            {"key": "Tier", "help": f"Quarter of the executives listed, by {_TIER_BY[tier_by]}: "
                                    "top 25%, middle, bottom 25% (marked red)."},
            *unit_columns("Executive", ["Branch", "Region"])[3:]]
    if "Rescued" in df.columns:
        cols.append({"key": "Rescued", "label": "Rescued", "fmt": "int",
                     "help": "Loans moved from SMA-1/SMA-2/NPA to a better bucket since last month."})
    t = _cached_totals(c, p, data_version, filter_key, branch)       # every loan in view, not just listed rows
    total = _total_row("Executive", t, "Total" if branch == "All" else f"Total ({branch})")
    if total and "Rescued" in df.columns:
        total["Rescued"] = t.get("Rescued")
    st.markdown(html_table(df, [x for x in cols if x["key"] in df.columns], total=total,
                           max_height=560 if len(df) > 15 else None,
                           highlight=lambda row: row.get("Tier") == _TIER_LABEL["bottom"]), unsafe_allow_html=True)
    st.caption("Executives with fewer than the minimum number of loans are left out (config.MIN_ACCOUNTS_EXECUTIVE).")
    _dl_btn(_with_total_row(df, total), "executives.xlsx", "dl_pi_exec")


# ── View 3: Segments ─────────────────────────────────────────────────────────

# Loan attributes the Segments view can group by (label -> column), shown when
# the file has the column with at least two values.
_SEGMENT_DIMS = {"Fuel type": "FUEL_TYPE", "Payment mode": "Paymethod", "NACH registered": "NACHStatus",
                 "Legal stage": "LGL_DESCRIPTION", "Secured / unsecured": "Security_Type"}
_BLANK_AS = {"LGL_DESCRIPTION": "Not in legal"}
_VALUE_LABELS = {"NACHStatus": {"Y": "Registered", "N": "Not registered"}}


def _view_segments(c, data_version, filter_key) -> None:
    from utils import segment_column
    dims = {}
    seg = segment_column(c)
    if seg:
        dims["Segment"] = seg
    for label, col in _SEGMENT_DIMS.items():
        if col in c.columns and (col in _BLANK_AS or c[col].nunique(dropna=True) > 1):
            dims[label] = col
    if not dims:
        st.info("No segment or loan-attribute column in this file.")
        return
    a, b, d = st.columns([1.3, 1.6, 1.2])
    dim = a.selectbox("By", list(dims), key="pi_seg_dim")
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
    takeaway(f"<b>{_esc(top['Name'])}</b>{where} is first on this list: NPA {top['NPA%']:.1f}% "
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
        view = list_controls("pi_top_loans", df, "Branch", "delinquent loans")
        book = float(pd.to_numeric(c.get("SOH"), errors="coerce").sum()) if "SOH" in c.columns else 0.0
        soh = float(view["SOH"].sum())
        takeaway(f"These {len(view):,} loans hold <b>₹{soh / 1e7:,.2f} Cr</b> SOH "
                  f"({soh / book * 100 if book else 0:.1f}% of the whole book); "
                  f"{int((view['Bucket'] == 'NPA').sum()):,} of them are already NPA.")
        st.dataframe(_safe_df(view), width="stretch", hide_index=True)
        _dl_btn(view.rename(columns={v: k for k, v in _RENAME.items() if v in view.columns and k in c.columns}),
                "largest_delinquent_loans.xlsx", "dl_pi_top_loans", full_source=c)
    elif what == "Fleet operators":
        fl = _cached_fleet(c, data_version, filter_key)
        if not fl.get("count"):
            st.info(f"No fleet operators (customers with {FLEET_MIN_LOANS}+ loans, matched by mobile number).")
            return
        takeaway(f"<b>{fl['count']:,}</b> fleet operators (customers with {FLEET_MIN_LOANS}+ loans) hold "
                  f"<b>₹{fl['total_soh_cr']:,.2f} Cr</b> SOH; {fl['npa_operators']:,} of them have at least one NPA loan.")
        df = fl["top_df"].rename(columns={"Unit": "Branch"})
        view = list_controls("pi_fleet", df, "Branch", "fleet operators")
        st.dataframe(_safe_df(view), width="stretch", hide_index=True)
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
    takeaway(f"<b>{len(cust):,}</b> customers with ₹{min_cr:,.2f} Cr or more exposure hold "
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
                 width="stretch", hide_index=True, height=min(38 + 35 * len(view), 520))
    st.caption(f"{len(view):,} loans. Delinquent loans (any EMI or charge overdue) are shaded red and come first "
               "for each customer.")
    _dl_btn(lv, "large_customer_loans.xlsx", "dl_pi_cust_loans")


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
    else:
        _view_exposure(df_curr, data_version, filter_key)
