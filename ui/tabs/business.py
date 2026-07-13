"""
Business tab  -  new advances (originations), pre-computed, zero AI calls.
A different analytical axis from the rest of the app: origination/business
volume, not collection performance -- same reasoning that gave Migration its
own tab instead of living inside Portfolio Intelligence.
"""
import pandas as pd
import streamlit as st

from ui.components import _dl_btn, _safe_df, _static_kpi_card_html, _chart_card, _divider, _style_main_content_selectbox, append_total_row
from ui.tabs.portfolio_intelligence import _section, _render_product_table, _roll_vintage
from analysis.portfolio_intelligence import (
    compute_new_advances_trend, roll_new_advances_trend, compute_new_advances_trend_chart,
    build_vintage_chart,
)
from config import NEW_ADVANCES_TREND_DEFAULT_MONTHS, NEW_ADVANCES_TREND_MONTH_OPTIONS

_GRANULARITY_OPTIONS = ["Monthly", "Quarterly", "Half-Yearly", "Yearly", "Financial Year"]


# ── Cached recompute wrappers ─────────────────────────────────────────────────
# Streamlit reruns EVERY tab's code on EVERY interaction anywhere on the page,
# not just the currently-visible tab -- so a widget change on Dashboard still
# re-executes this whole module's render functions. Without these caches, the
# Ag_Date groupby + period rollup below (window/granularity dropdowns, both
# tab-local widgets the sidebar-level app.py caches don't know about) reran
# from scratch on every single click anywhere in the app, not just when the
# Business tab's own selections actually changed -- same pattern
# ui/tabs/migration.py's _cached_filtered_roll_rate already uses for its own
# drill-down filter. _df_c/_vintage_df are underscore-prefixed so Streamlit
# never hashes the frame itself; data_version + the sidebar filter tuple +
# curr_month + this tab's own widget values are the cheap, explicit,
# accuracy-preserving cache key -- any one of them changing is a cache miss,
# so a genuinely different result is never served stale.
@st.cache_data(show_spinner=False, max_entries=32)
def _cached_new_advances_trend(
    _df_c: pd.DataFrame, data_version: int, region: str, branch: str, status: str, segment: tuple,
    curr_month: str, months, granularity: str,
) -> pd.DataFrame:
    trend_df = compute_new_advances_trend(_df_c, as_of=curr_month, months=months)
    return roll_new_advances_trend(trend_df, granularity)


@st.cache_data(show_spinner=False, max_entries=32)
def _cached_vintage_rollup(
    _vintage_df: pd.DataFrame, data_version: int, region: str, branch: str, status: str, segment: tuple,
    granularity: str,
) -> pd.DataFrame:
    return _roll_vintage(_vintage_df, granularity)


# ── Section 1: New Advances This Month ────────────────────────────────────────

def _render_new_advances(data: dict) -> None:
    _section("Section 1  -  New Advances This Month")
    st.caption(
        "Loans whose Agreement Date falls in this reporting month -- new business funded, "
        "regardless of that loan's current collection status."
    )

    if not data or not data.get("accounts"):
        st.info("No new advances found for this reporting month (check that Ag_Date is populated).")
        return

    has_prev = data.get("has_prev", False)

    cards = (
        _static_kpi_card_html("New Advances", f"{data['accounts']:,}", "Accounts funded this month")
        + _static_kpi_card_html("Funded Amount", f"&#8377;{data['funded_cr']:,.2f} Cr", "Total loan amount disbursed")
        + _static_kpi_card_html("Avg Ticket Size", f"&#8377;{data['avg_ticket_l']:,.2f} L", "Average loan amount per account")
    )
    st.markdown(f'<div class="kpi-row">{cards}</div>', unsafe_allow_html=True)

    if has_prev:
        st.markdown("<br>", unsafe_allow_html=True)
        mom_cards = (
            _static_kpi_card_html(
                "Accounts vs Last Month", f"{data['prev_accounts']:,} → {data['accounts']:,}",
                f"{data['accounts_mom_pct']:+.2f}% MoM" if data.get("accounts_mom_pct") is not None else "-",
            )
            + _static_kpi_card_html(
                "Funded vs Last Month", f"&#8377;{data['prev_funded_cr']:,.2f} Cr → &#8377;{data['funded_cr']:,.2f} Cr",
                f"{data['funded_mom_pct']:+.2f}% MoM" if data.get("funded_mom_pct") is not None else "-",
            )
        )
        st.markdown(f'<div class="kpi-row">{mom_cards}</div>', unsafe_allow_html=True)
    else:
        st.info("No loans originated last month found in this file -- month-over-month new business trend unavailable.")

    seg_df = data.get("segment", pd.DataFrame())
    if not seg_df.empty:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div style="font-size:13px;font-weight:700;color:#111;margin-bottom:6px;">By Segment</div>', unsafe_allow_html=True)
        _seg_ratio_cols = {"Avg Ticket (L)": ("Funded (Cr)", "Accounts", 100)}
        st.dataframe(_safe_df(append_total_row(seg_df, ratio_cols=_seg_ratio_cols)), use_container_width=True, hide_index=True)
        _dl_btn(seg_df, "new_advances_segment.xlsx", "dl_new_advances_segment")


# ── Section 2: New Advances Trend ─────────────────────────────────────────────

def _render_new_advances_trend(
    df_curr: pd.DataFrame, curr_month: str,
    data_version: int, region: str, branch: str, status: str, segment: tuple,
) -> None:
    _section("Section 2  -  New Advances Trend", margin_top="24px")
    st.caption(
        "Accounts + funded amount by disbursement month, across this file's full Ag_Date history -- "
        "not just this reporting month. No previous month file needed."
    )

    # This selectbox renders in the main content area, not the sidebar, so it
    # doesn't get ui/styles.py's `[data-testid="stSidebar"] .stSelectbox`
    # white-text rule -- without this, the dark selectbox background leaves
    # the "Trend window" option text black-on-black. Same fix, same "#fff"
    # color, as ui/tabs/migration.py's drill-down and ui/tabs/ai_query.py's
    # own main-content selectbox.
    _style_main_content_selectbox("#fff")
    col_win, col_gran = st.columns([1, 2])
    with col_win:
        window_choice = st.selectbox(
            "Trend window", NEW_ADVANCES_TREND_MONTH_OPTIONS,
            index=NEW_ADVANCES_TREND_MONTH_OPTIONS.index(NEW_ADVANCES_TREND_DEFAULT_MONTHS),
            key="new_adv_trend_window",
        )
    with col_gran:
        granularity = st.radio(
            "Group by", _GRANULARITY_OPTIONS,
            horizontal=True, key="new_adv_trend_gran", index=1,
        )
    months = None if window_choice == "All" else int(window_choice)

    plot_df = _cached_new_advances_trend(
        df_curr, data_version, region, branch, status, segment, curr_month, months, granularity,
    )
    if plot_df.empty:
        st.info("No Ag_Date history found for a trend view.")
        return

    _chart_card(compute_new_advances_trend_chart(plot_df, granularity))
    st.markdown("<br>", unsafe_allow_html=True)
    # plot_df is already chronologically ascending (roll_new_advances_trend's
    # own sort key) -- reverse it for "most recent first" display rather than
    # re-sorting by the "Month" string, which breaks for non-monthly labels
    # like "Q1-2026" or "FY25-26" (alphabetical != chronological there).
    _trend_ratio_cols = {"Avg Ticket (L)": ("Funded (Cr)", "Accounts", 100)}
    st.dataframe(_safe_df(append_total_row(plot_df.iloc[::-1], ratio_cols=_trend_ratio_cols)), use_container_width=True, hide_index=True)
    _dl_btn(plot_df, "new_advances_trend.xlsx", "dl_new_advances_trend")


# ── Section 3: New Advances by Region / Branch / Executive ───────────────────

def _render_new_advances_by_dimension(data: dict) -> None:
    _section("Section 3  -  New Advances by Region / Branch / Executive", margin_top="24px")
    st.caption("This month's new advances at each grain, with MoM vs that same entity's own advances last month.")

    dim_tabs_avail = []
    if not data.get("region", pd.DataFrame()).empty:    dim_tabs_avail.append(("Region",    "region"))
    if not data.get("branch", pd.DataFrame()).empty:    dim_tabs_avail.append(("Branch",    "branch"))
    if not data.get("executive", pd.DataFrame()).empty: dim_tabs_avail.append(("Executive", "executive"))

    if not dim_tabs_avail:
        st.info("No dimension data found.")
        return

    sub_tabs = st.tabs([t[0] for t in dim_tabs_avail])
    for tab, (label, key) in zip(sub_tabs, dim_tabs_avail):
        with tab:
            df = data[key]
            _dim_ratio_cols = {"Avg Ticket (L)": ("Funded (Cr)", "Accounts This Month", 100)}
            st.dataframe(_safe_df(append_total_row(df, ratio_cols=_dim_ratio_cols)), use_container_width=True, hide_index=True)
            _dl_btn(df, f"new_advances_{key}.xlsx", f"dl_new_advances_{key}")


# ── Section 4: Disbursement Vintage ───────────────────────────────────────────

def _render_disbursement_vintage(
    vintage_df: pd.DataFrame,
    data_version: int, region: str, branch: str, status: str, segment: tuple,
) -> None:
    _section("Section 4  -  Disbursement Vintage", margin_top="24px")
    st.caption(
        "Rising NPA% on older cohorts = expected ageing. "
        "Spike on a specific month = sourcing quality issue that month  -  collections can't fix it, credit can stop repeating it."
    )

    if vintage_df.empty:
        st.info("No disbursement vintage data found in this file.")
        return

    granularity = st.radio(
        "Group by", _GRANULARITY_OPTIONS,
        horizontal=True, key="vintage_gran", index=1,
    )
    plot_df = _cached_vintage_rollup(vintage_df, data_version, region, branch, status, segment, granularity)
    fig_v = build_vintage_chart(plot_df)
    if fig_v.data:
        _chart_card(fig_v)
        st.markdown("<br>", unsafe_allow_html=True)
    _render_product_table(plot_df.drop(columns=["NPA Count", "SMA-2 Count"], errors="ignore"))
    _dl_btn(plot_df, "disbursement_vintage.xlsx", "dl_vintage")


# ── Main render entry point ───────────────────────────────────────────────────

def render_business_tab(
    new_advances: dict,
    df_curr: pd.DataFrame,
    curr_month: str,
    dimension_data: dict,
    vintage_df: pd.DataFrame,
    data_version: int, region: str, branch: str, status: str, segment: tuple,
) -> None:
    _render_new_advances(new_advances or {})
    _divider()

    _render_new_advances_trend(df_curr, curr_month, data_version, region, branch, status, segment)
    _divider()

    _render_new_advances_by_dimension(dimension_data or {})
    _divider()

    _render_disbursement_vintage(
        vintage_df if vintage_df is not None else pd.DataFrame(),
        data_version, region, branch, status, segment,
    )
