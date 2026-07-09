import pandas as pd
import streamlit as st

from ui.components import _static_kpi_card_html, _chart_card, _empty_state, _style_main_content_selectbox, _npa_pct_color


def _filter_options(df: pd.DataFrame, col: str) -> list[str]:
    """"All" plus every distinct value in `col`, or just ["All"] if the
    column isn't present (e.g. an older LCC extract missing MNT NAME)."""
    if col not in df.columns:
        return ["All"]
    return ["All"] + sorted(df[col].dropna().unique().tolist())


@st.cache_data(show_spinner=False)
def _cached_filtered_roll_rate(_df_curr_f: pd.DataFrame, _df_prev_f: pd.DataFrame, data_version: int, drilldown_key: str):
    """Same cached-recompute pattern app.py's own `_cached_roll_rate` uses for
    the unfiltered case -- without this, a drill-down filter recomputes the
    full roll-rate matrix (a set-build + merge over every matched account) on
    EVERY Streamlit rerun, including reruns triggered by unrelated widgets
    elsewhere on the page, not just when the filter selection actually changes.

    _df_curr_f/_df_prev_f are underscore-prefixed (never hashed by Streamlit --
    hashing the full sliced DataFrame every rerun was the actual cost this
    cache was meant to avoid); data_version + drilldown_key (which already
    captures both the sidebar's filter selection AND this tab's own local
    Region/Branch/Executive drill-down) are the cheap, explicit stand-ins."""
    from analysis.roll_rate import compute_roll_rate_matrix
    return compute_roll_rate_matrix(_df_curr_f, _df_prev_f)


def _bucket_summary_table_html(df: pd.DataFrame) -> str:
    headers = ["Bucket", "Accounts", "Roll Fwd%", "Stable%", "Roll Bwd%"]
    th = "".join(
        f'<th style="background:#111;color:#FFC000;padding:8px 12px;font-size:11px;'
        f'text-align:{"left" if h == "Bucket" else "right"};white-space:nowrap;">{h}</th>'
        for h in headers
    )
    rows_html = ""
    for _, row in df.iterrows():
        fwd, stable, bwd = row["Roll Fwd%"], row["Stable%"], row["Roll Bwd%"]
        # Roll Fwd% is high=bad, same direction _npa_pct_color already encodes -- reuse it
        # (thresholds widened to 20/10 since a bucket-level forward-roll rate runs higher
        # than a portfolio NPA%). Roll Bwd% is high=GOOD (the opposite direction), so it
        # can't reuse _sma2_pct_color -- that helper colors a HIGH value red, which would
        # be backwards here; kept as its own inline threshold instead of a shared helper.
        fwd_color = _npa_pct_color(fwd, hi=20, mid=10)
        bwd_color = "#16a34a" if bwd >= 10 else ("#d97706" if bwd >= 5 else "#374151")
        rows_html += (
            f'<tr style="border-bottom:1px solid #f0f0f0;">'
            f'<td style="padding:8px 12px;font-size:12px;font-weight:700;">{row["Bucket"]}</td>'
            f'<td style="padding:8px 12px;font-size:12px;text-align:right;">{row["Accounts"]:,}</td>'
            f'<td style="padding:8px 12px;font-size:12px;text-align:right;font-weight:700;color:{fwd_color};">{fwd:.1f}%</td>'
            f'<td style="padding:8px 12px;font-size:12px;text-align:right;color:#6b7280;">{stable:.1f}%</td>'
            f'<td style="padding:8px 12px;font-size:12px;text-align:right;font-weight:700;color:{bwd_color};">{bwd:.1f}%</td>'
            f'</tr>'
        )
    return (
        f'<div style="overflow-x:auto;border-radius:8px;border:1px solid #e5e7eb;">'
        f'<table style="width:100%;border-collapse:collapse;font-family:Inter,sans-serif;">'
        f'<thead><tr>{th}</tr></thead><tbody>{rows_html}</tbody>'
        f'</table></div>'
    )


def render_migration_tab(
    df_curr: pd.DataFrame,
    df_prev: pd.DataFrame,
    rr_matrix,
    rr_meta: dict | None,
    data_version: int = 0,
    sidebar_filter_key: str = "",
) -> None:
    if rr_meta is None:
        _empty_state(
            "📈", "Previous month data not loaded",
            "Upload a previous month LCC file alongside the current month file,<br>"
            "then click <strong>Generate Dashboard</strong> to see bucket migration and roll-rate analysis.",
        )
        return

    from analysis.roll_rate import compute_bucket_roll_summary, build_roll_rate_heatmap

    # ── Drill-down filters (local to this tab -- narrows the migration view
    # without touching the sidebar's global filter used by every other tab) ──
    st.markdown('<div class="section-label">Drill Down</div>', unsafe_allow_html=True)
    # These selectboxes render in the main content area, not the sidebar, so they
    # don't get ui/styles.py's `[data-testid="stSidebar"] .stSelectbox` white-text
    # rule -- without this, the dark selectbox background leaves the "All"/option
    # text nearly unreadable. MUST use the same color ui/tabs/ai_query.py's/
    # business.py's own main-content selectboxes use, for visual consistency
    # (this styles every stSelectbox on the page, not just this tab's -- only
    # the active tab's render code runs per rerun, so this can no longer leak
    # into an inactive tab, but a mismatched color would still look inconsistent
    # if it ever differed from the other tabs' calls).
    _style_main_content_selectbox("#fff")
    f_col1, f_col2, f_col3 = st.columns(3)

    with f_col1:
        sel_region = st.selectbox("Region", _filter_options(df_curr, "RegionName"), key="mig_region")

    df_for_branch = df_curr if sel_region == "All" else df_curr[df_curr["RegionName"] == sel_region]
    with f_col2:
        sel_branch = st.selectbox("Branch", _filter_options(df_for_branch, "Unit"), key="mig_branch")

    df_for_exec = df_for_branch if sel_branch == "All" else df_for_branch[df_for_branch["Unit"] == sel_branch]
    with f_col3:
        sel_exec = st.selectbox("Executive", _filter_options(df_for_exec, "MNT NAME"), key="mig_exec")

    has_filter = sel_region != "All" or sel_branch != "All" or sel_exec != "All"

    if has_filter:
        def _apply(df: pd.DataFrame) -> pd.DataFrame:
            # One combined boolean mask, one indexing op -- not up to 3
            # sequential boolean-index copies for the up-to-3 active filters.
            mask = pd.Series(True, index=df.index)
            if sel_region != "All" and "RegionName" in df.columns:
                mask &= df["RegionName"] == sel_region
            if sel_branch != "All" and "Unit" in df.columns:
                mask &= df["Unit"] == sel_branch
            if sel_exec != "All" and "MNT NAME" in df.columns:
                mask &= df["MNT NAME"] == sel_exec
            return df[mask]

        df_curr_f = _apply(df_curr)
        df_prev_f = _apply(df_prev) if df_prev is not None else pd.DataFrame()
        if len(df_curr_f) == 0:
            st.warning("No accounts match this Region / Branch / Executive combination.")
            return
        # Combines the sidebar's own filter selection (already baked into
        # df_curr's content, but not otherwise visible to this cache key) with
        # this tab's own local drill-down -- both determine df_curr_f's content.
        drilldown_key = f"{sidebar_filter_key}|{sel_region}|{sel_branch}|{sel_exec}"
        rr_matrix, rr_meta = _cached_filtered_roll_rate(df_curr_f, df_prev_f, data_version, drilldown_key)

    # ── KPI row ──────────────────────────────────────────────────────────────
    st.markdown('<div class="section-label" style="margin-top:16px;">Roll-Rate Summary</div>', unsafe_allow_html=True)
    rr_kpis = [
        ("Roll-Forward Rate",  rr_meta["roll_forward_rate"],  "%", "#dc2626", "Accounts that worsened bucket"),
        ("Roll-Backward Rate", rr_meta["roll_backward_rate"], "%", "#16a34a", "Delinquent accounts returned to STD"),
        ("NPA Formation",      rr_meta["npa_formation_rate"], "%", "#991b1b", "Non-NPA accounts that became NPA"),
        ("Matched Accounts",   rr_meta["matched_count"],      "",  "#111827", "Accounts in both months"),
    ]
    for col, (label, val, unit, color, tip) in zip(st.columns(4), rr_kpis):
        with col:
            st.markdown(
                _static_kpi_card_html(label, f"{val:,.1f}{unit}", tip, color=color, value_style="font-size:24px;"),
                unsafe_allow_html=True,
            )

    # ── Heatmap ──────────────────────────────────────────────────────────────
    st.markdown('<div class="section-label" style="margin-top:20px;">Bucket Migration Matrix</div>', unsafe_allow_html=True)
    fig_rr = build_roll_rate_heatmap(rr_matrix)
    _chart_card(fig_rr)
    st.caption(
        f"{rr_meta['matched_count']:,} matched accounts | "
        f"{rr_meta['new_entries']:,} new this month | "
        f"{rr_meta['exits']:,} closed/exited"
    )

    # ── Bucket-wise Roll Forward / Stable / Roll Backward % table ─────────────
    st.markdown('<div class="section-label" style="margin-top:20px;">Bucket-wise Roll Rates</div>', unsafe_allow_html=True)
    bucket_summary = compute_bucket_roll_summary(rr_matrix)
    if bucket_summary.empty or bucket_summary["Accounts"].sum() == 0:
        st.caption("No matched accounts for this selection.")
    else:
        st.markdown(_bucket_summary_table_html(bucket_summary), unsafe_allow_html=True)
        st.caption(
            "Per previous-month bucket: % that rolled forward (worsened), stayed stable, "
            "or rolled backward (improved) this month."
        )
