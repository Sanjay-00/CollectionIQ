from io import BytesIO

import pandas as pd
import streamlit as st

from utils import (
    build_branch_bar_chart,
    build_closing_pc_chart,
    build_html_export,
    build_status_bar_chart,
    fmt_value,
)

_KIND = {
    "Month Demand": "money", "Total Collection": "money", "Collection %": "pct",
    "Strike %": "pct", "NPA %": "pct", "Hard Bucket %": "pct",
    "Count": "count", "SOH": "money", "LCC%": "pct", "CMD %": "pct",
}
_KPI_TOP      = ["Month Demand", "Total Collection", "Collection %", "Strike %", "NPA %", "Hard Bucket %"]
_KPI_BOT      = ["Count", "CMD %"]
_KPI_EXPOSURE = ["SOH"]
_INVERSE_MOM  = {"NPA %", "Hard Bucket %"}


def _kpi_card(label: str, value: str, mom: float) -> str:
    arrow = "▲" if mom >= 0 else "▼"
    cls   = ("kpi-mom-down" if mom >= 0 else "kpi-mom-up") if label in _INVERSE_MOM \
            else ("kpi-mom-up" if mom >= 0 else "kpi-mom-down")
    return (
        f'<div class="kpi-card">'
        f'<div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div>'
        f'<div class="kpi-mom">MoM <span class="{cls}">{arrow} {abs(mom):.2f}%</span></div>'
        f'</div>'
    )


def _kpi_row(keys: list, metrics: dict) -> None:
    html = "".join(
        _kpi_card(k, fmt_value(metrics[k][0], _KIND[k]), metrics[k][1])
        for k in keys if k in metrics and k in _KIND
    )
    st.markdown(f'<div class="kpi-row">{html}</div>', unsafe_allow_html=True)


def render_dashboard_tab(
    df_curr: pd.DataFrame,
    df_prev: pd.DataFrame,
    metrics: dict,
    curr_month: str,
    sel_region: str,
    sel_branch: str,
    sel_status: str,
    alerts: list,
    scorecard_df,
    rr_meta: dict | None,
) -> None:
    # ── KPIs ────────────────────────────────────────────────────────────────
    st.markdown('<div class="section-label">Key Performance Indicators</div>', unsafe_allow_html=True)
    _kpi_row(_KPI_TOP, metrics)

    # ── Charts ──────────────────────────────────────────────────────────────
    st.markdown('<div class="section-label">Portfolio Analysis</div>', unsafe_allow_html=True)
    fig_status  = build_status_bar_chart(df_curr)
    fig_branch  = build_branch_bar_chart(df_curr)
    fig_closing = build_closing_pc_chart(df_curr)

    col_bar, col_hbar, col_lcc = st.columns([2, 2, 1])
    with col_bar:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.plotly_chart(fig_status, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with col_hbar:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.plotly_chart(fig_branch, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with col_lcc:
        lcc_val   = fmt_value(metrics["LCC%"][0], "pct")
        lcc_mom   = metrics["LCC%"][1]
        lcc_arrow = "▲" if lcc_mom >= 0 else "▼"
        lcc_cls   = "kpi-mom-up" if lcc_mom >= 0 else "kpi-mom-down"
        st.markdown(f"""
        <div class="kpi-card" style="height:100%;display:flex;flex-direction:column;
             justify-content:center;align-items:center;text-align:center;margin-top:0;">
          <div class="kpi-label">LCC %</div>
          <div style="font-size:36px;font-weight:800;color:#111;line-height:1.1;">{lcc_val}</div>
          <div class="kpi-mom">MoM <span class="{lcc_cls}">{lcc_arrow} {abs(lcc_mom):.2f}%</span></div>
        </div>
        """, unsafe_allow_html=True)

    col_trend, col_bot = st.columns([3, 2])
    with col_trend:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.plotly_chart(fig_closing, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with col_bot:
        st.markdown("<br>", unsafe_allow_html=True)
        _kpi_row(_KPI_BOT, metrics)
        _kpi_row(_KPI_EXPOSURE, metrics)

    # ── HTML export ─────────────────────────────────────────────────────────
    st.markdown('<div style="border-top:1px solid #e5e7eb;margin:24px 0 16px 0;"></div>', unsafe_allow_html=True)
    col_dl, _ = st.columns([1, 3])
    with col_dl:
        filters_applied = {
            "Region": sel_region, "Branch": sel_branch,
            "Loan Status": sel_status, "Year Month": str(curr_month),
        }
        html_content = build_html_export(
            df_curr, df_prev, metrics, fig_status, fig_branch, fig_closing,
            filters_applied, curr_month=curr_month, alerts=alerts,
            scorecard_df=scorecard_df,
            roll_rate_meta=rr_meta,
        )
        st.download_button(
            label="⬇  Download Dashboard as HTML",
            data=html_content.encode("utf-8"),
            file_name=f"shriram_dashboard_{curr_month}.html",
            mime="text/html",
            use_container_width=True,
        )
