"""
Business tab: new advances (originations), not collection performance. One
question per view (VIEWS): this month, where it came from, the trend, and how
each disbursement month's loans are doing now (vintage). Each view has a
one-line takeaway and tables in the app's shared format. Zero AI calls.
"""
import pandas as pd
import streamlit as st

from analysis.new_business import (
    build_vintage_chart, compute_new_advances_trend, compute_new_advances_trend_chart, roll_new_advances_trend,
    roll_vintage,
)
from config import NEW_ADVANCES_TREND_DEFAULT_MONTHS, NEW_ADVANCES_TREND_MONTH_OPTIONS, VINTAGE_HIGHLIGHT_MIN_SHARE_PCT
from ui.components import _chart_card, _dl_btn, _static_kpi_card_html, html_table, takeaway

VIEWS = ["This month", "By region, branch, executive", "Trend", "Vintage"]
_GRANULARITY_OPTIONS = ["Monthly", "Quarterly", "Half-Yearly", "Yearly", "Financial Year"]


# Cached per upload, sidebar filters and this tab's own picks (the frames
# themselves are underscore-prefixed, never hashed).
@st.cache_data(show_spinner=False, max_entries=32)
def _cached_trend(_df_c: pd.DataFrame, data_version: int, filter_key: str, curr_month: str, months,
                  granularity: str) -> pd.DataFrame:
    return roll_new_advances_trend(compute_new_advances_trend(_df_c, as_of=curr_month, months=months), granularity)


@st.cache_data(show_spinner=False, max_entries=32)
def _cached_vintage(_vintage_df: pd.DataFrame, data_version: int, filter_key: str, granularity: str) -> pd.DataFrame:
    return roll_vintage(_vintage_df, granularity)


def _ticket_total(df: pd.DataFrame, accounts: str, label_col: str) -> dict:
    """Sums, with the average ticket recomputed from them (never an average of averages)."""
    n, funded = int(df[accounts].sum()), float(df["Funded (Cr)"].sum())
    return {label_col: "Total", accounts: n, "Funded (Cr)": round(funded, 2),
            "Avg Ticket (L)": round(funded * 100 / n, 2) if n else None}


def _month_total(data: dict, accounts: str, label_col: str) -> dict:
    """This month's true totals (from the whole month, not a sum of rounded rows)."""
    return {label_col: "Total", accounts: data["accounts"], "Funded (Cr)": data["funded_cr"],
            "Avg Ticket (L)": data["avg_ticket_l"]}


# ── View 1: this month ───────────────────────────────────────────────────────

def _view_this_month(data: dict) -> None:
    if not data or not data.get("accounts"):
        st.info("No new advances in this reporting month (loans whose agreement date falls in it).")
        return
    line = (f"<b>{data['accounts']:,}</b> new loans this month, <b>₹{data['funded_cr']:,.2f} Cr</b> funded, "
            f"average ticket ₹{data['avg_ticket_l']:,.2f} L.")
    if data.get("has_prev"):
        def move(pct):
            return "no change" if not pct else f"{'up' if pct > 0 else 'down'} {abs(pct):.1f}%"
        line += (f" Last month: {data['prev_accounts']:,} loans, ₹{data['prev_funded_cr']:,.2f} Cr "
                 f"(loans {move(data.get('accounts_mom_pct'))}, money {move(data.get('funded_mom_pct'))}).")
    takeaway(line)
    cards = (_static_kpi_card_html("New Advances", f"{data['accounts']:,}", "Loans agreed this month")
             + _static_kpi_card_html("Funded", f"&#8377;{data['funded_cr']:,.2f} Cr", "Total loan amount")
             + _static_kpi_card_html("Average Ticket", f"&#8377;{data['avg_ticket_l']:,.2f} L", "Loan amount per loan"))
    st.markdown(f'<div class="kpi-row">{cards}</div>', unsafe_allow_html=True)

    seg = data.get("segment", pd.DataFrame())
    if not seg.empty:
        st.markdown('<div style="height:10px;"></div>', unsafe_allow_html=True)
        total = {**_month_total(data, "Accounts", "Segment"), "Share %": 100.0}
        st.markdown(html_table(seg, [
            {"key": "Segment", "bold": True}, {"key": "Accounts", "label": "Loans", "fmt": "int"},
            {"key": "Share %", "label": "Share of loans", "fmt": "pct"},
            {"key": "Funded (Cr)", "label": "Funded", "fmt": "cr"},
            {"key": "Avg Ticket (L)", "label": "Avg ticket (₹ L)", "fmt": "num"},
        ], total=total), unsafe_allow_html=True)
        _dl_btn(seg, "new_advances_segment.xlsx", "dl_new_advances_segment")


# ── View 2: where it came from ───────────────────────────────────────────────

def _view_by_unit(dimension_data: dict, month: dict) -> None:
    levels = [(lb, k) for lb, k in (("Region", "region"), ("Branch", "branch"), ("Executive", "executive"))
              if not dimension_data.get(k, pd.DataFrame()).empty]
    if not levels:
        st.info("No new advances by region, branch or executive in this view.")
        return
    label = st.radio("Show", [lb for lb, _ in levels], horizontal=True, key="biz_level")
    key = dict(levels)[label]
    df = dimension_data[key].sort_values("Accounts This Month", ascending=False, kind="stable").reset_index(drop=True)
    top = df.iloc[0]
    takeaway(f"<b>{len(df):,}</b> {label.lower()}s booked new loans this month; the most: <b>{top[label]}</b> with "
             f"{int(top['Accounts This Month']):,} loans (₹{top['Funded (Cr)']:,.2f} Cr).")
    # The whole month, not a sum of the rows: a unit with new loans last month
    # but none this month has no row, so summing "Last month" would undercount.
    total = None
    if month.get("accounts"):
        total = _month_total(month, "Accounts This Month", label)
        if month.get("has_prev"):
            total.update({"Prev Accounts": month["prev_accounts"], "Accounts MoM %": month.get("accounts_mom_pct"),
                          "Funded MoM %": month.get("funded_mom_pct")})
    up = {"good_if_up": True, "help": "Change against the same unit's own new loans last month."}
    cols = [{"key": label, "bold": True}, *[{"key": c} for c in ("Branch", "Region") if c in df.columns and c != label],
            {"key": "Accounts This Month", "label": "New loans", "fmt": "int"},
            {"key": "Prev Accounts", "label": "Last month", "fmt": "int"},
            {"key": "Accounts MoM %", "label": "Change", "fmt": "pct_change", **up},
            {"key": "Funded (Cr)", "label": "Funded", "fmt": "cr"},
            {"key": "Funded MoM %", "label": "Funded change", "fmt": "pct_change", **up},
            {"key": "Avg Ticket (L)", "label": "Avg ticket (₹ L)", "fmt": "num"},
            {"key": "Total Accounts", "label": "Book (all loans)", "fmt": "int",
             "help": "Every loan this unit holds, not just this month's."}]
    st.markdown(html_table(df, [c for c in cols if c["key"] in df.columns], total=total,
                           max_height=560 if len(df) > 15 else None), unsafe_allow_html=True)
    _dl_btn(df, f"new_advances_{key}.xlsx", f"dl_new_advances_{key}")


# ── View 3: trend ────────────────────────────────────────────────────────────

def _view_trend(df_curr: pd.DataFrame, curr_month: str, data_version: int, filter_key: str) -> None:
    a, b = st.columns([1, 2])
    window = a.selectbox("Window", NEW_ADVANCES_TREND_MONTH_OPTIONS,
                         index=NEW_ADVANCES_TREND_MONTH_OPTIONS.index(NEW_ADVANCES_TREND_DEFAULT_MONTHS),
                         key="new_adv_trend_window",
                         format_func=lambda v: "All months" if v == "All" else f"Last {v} months")
    granularity = b.radio("Group by", _GRANULARITY_OPTIONS, horizontal=True, key="new_adv_trend_gran", index=1)
    months = None if window == "All" else int(window)
    trend = _cached_trend(df_curr, data_version, filter_key, curr_month, months, granularity)
    if trend.empty:
        st.info("No agreement dates to build a trend from.")
        return
    best = trend.loc[trend["Accounts"].idxmax()]
    takeaway(f"Busiest period: <b>{best['Month']}</b> with {int(best['Accounts']):,} new loans "
             f"(₹{best['Funded (Cr)']:,.2f} Cr).")
    _chart_card(compute_new_advances_trend_chart(trend, granularity))
    view = trend.iloc[::-1].reset_index(drop=True)          # most recent first
    st.markdown(html_table(view, [
        {"key": "Month", "label": "Period", "bold": True}, {"key": "Accounts", "label": "New loans", "fmt": "int"},
        {"key": "Funded (Cr)", "label": "Funded", "fmt": "cr"},
        {"key": "Avg Ticket (L)", "label": "Avg ticket (₹ L)", "fmt": "num"},
    ], total=_ticket_total(view, "Accounts", "Month"), max_height=480 if len(view) > 12 else None),
        unsafe_allow_html=True)
    _dl_btn(trend, "new_advances_trend.xlsx", "dl_new_advances_trend")


# ── View 4: vintage ──────────────────────────────────────────────────────────

def _view_vintage(vintage_df: pd.DataFrame, data_version: int, filter_key: str) -> None:
    if vintage_df.empty:
        st.info("No disbursement months in this file.")
        return
    granularity = st.radio("Group by", _GRANULARITY_OPTIONS, horizontal=True, key="vintage_gran", index=1)
    df = _cached_vintage(vintage_df, data_version, filter_key, granularity)
    sizable = df[df["Accounts"] >= df["Accounts"].sum() * VINTAGE_HIGHLIGHT_MIN_SHARE_PCT / 100]
    if not sizable.empty:
        worst = sizable.loc[sizable["NPA%"].idxmax()]
        takeaway(f"Highest NPA among periods with {VINTAGE_HIGHLIGHT_MIN_SHARE_PCT}%+ of the loans: disbursed in "
                 f"<b>{worst['Disbursement Month']}</b>, {worst['NPA%']:.1f}% ({int(worst['NPA Count']):,} of "
                 f"{int(worst['Accounts']):,}). A rise on older periods is normal ageing; a spike on one period "
                 "points to that period's sourcing.")
    fig = build_vintage_chart(df)
    if fig.data:
        _chart_card(fig)
    view = df.iloc[::-1].reset_index(drop=True)              # most recent first
    n, npa, sma2 = int(view["Accounts"].sum()), int(view["NPA Count"].sum()), int(view["SMA-2 Count"].sum())
    total = {"Disbursement Month": "Total", "Accounts": n, "NPA Count": npa, "SMA-2 Count": sma2,
             "NPA%": round(npa / n * 100, 2) if n else None, "SMA-2%": round(sma2 / n * 100, 2) if n else None,
             "SOH (Cr)": round(float(view["SOH (Cr)"].sum()), 2)}
    st.markdown(html_table(view, [
        {"key": "Disbursement Month", "label": "Disbursed", "bold": True},
        {"key": "Accounts", "label": "Loans", "fmt": "int"},
        {"key": "SMA-2%", "label": "SMA-2", "fmt": "pct_count", "count": "SMA-2 Count", "heat": True},
        {"key": "NPA%", "label": "NPA", "fmt": "pct_count", "count": "NPA Count", "heat": True},
        {"key": "SOH (Cr)", "label": "SOH", "fmt": "cr"},
    ], total=total, max_height=480 if len(view) > 12 else None), unsafe_allow_html=True)
    _dl_btn(df, "disbursement_vintage.xlsx", "dl_vintage")


# ── The tab ──────────────────────────────────────────────────────────────────

def render_business_tab(
    new_advances: dict, df_curr: pd.DataFrame, curr_month: str, dimension_data: dict, vintage_df: pd.DataFrame,
    data_version: int = 0, filter_key: str = "",
) -> None:
    view = st.segmented_control("View", VIEWS, default=VIEWS[0], key="biz_view", label_visibility="collapsed") or VIEWS[0]
    if view == VIEWS[0]:
        _view_this_month(new_advances or {})
    elif view == VIEWS[1]:
        _view_by_unit(dimension_data or {}, new_advances or {})
    elif view == VIEWS[2]:
        _view_trend(df_curr, curr_month, data_version, filter_key)
    else:
        _view_vintage(vintage_df if vintage_df is not None else pd.DataFrame(), data_version, filter_key)
