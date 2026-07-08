from io import BytesIO

import pandas as pd
import streamlit as st

from utils import build_html_export, fmt_value
from ui.components import _kpi_card_html, _chart_card, _divider, _cached_html_export

_KIND = {
    "Month Demand": "money", "Total Collection": "money", "Collection %": "pct",
    "Strike %": "pct", "NPA %": "pct", "Hard Bucket %": "pct", "SMA-2 %": "pct",
    "Count": "count", "SOH": "money", "LCC%": "pct", "CMD %": "pct",
}
_KPI_TOP      = ["Month Demand", "Total Collection", "Collection %", "Strike %", "NPA %", "Hard Bucket %"]
_KPI_BOT      = ["Count", "CMD %"]
_KPI_EXPOSURE = ["SOH"]
_INVERSE_MOM  = {"NPA %", "Hard Bucket %", "SMA-2 %"}


def _kpi_row(keys: list, metrics: dict, count_deltas: dict | None = None) -> None:
    count_deltas = count_deltas or {}
    html = "".join(
        _kpi_card_html(
            k, fmt_value(metrics[k][0], _KIND[k]), metrics[k][1],
            inverse=k in _INVERSE_MOM, zero_delta_bad=True,
            count_delta=count_deltas.get(k),
        )
        for k in keys if k in metrics and k in _KIND
    )
    st.markdown(f'<div class="kpi-row">{html}</div>', unsafe_allow_html=True)


def _npa_count_delta(df_curr: pd.DataFrame, df_prev: pd.DataFrame) -> int | None:
    """Raw NPA account-count movement, used to break NPA %'s zero-delta tie."""
    if "curr_bucket" not in df_curr.columns or df_prev.empty or "curr_bucket" not in df_prev.columns:
        return None
    return int((df_curr["curr_bucket"] == "NPA").sum()) - int((df_prev["curr_bucket"] == "NPA").sum())


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
    fig_status,
    fig_branch,
    fig_closing,
    data_version: int = 0,
    segment: tuple = (),
) -> None:
    # ── KPIs ────────────────────────────────────────────────────────────────
    st.markdown('<div class="section-label">Key Performance Indicators</div>', unsafe_allow_html=True)
    _kpi_row(_KPI_TOP, metrics, count_deltas={"NPA %": _npa_count_delta(df_curr, df_prev)})

    # ── Charts ──────────────────────────────────────────────────────────────
    # fig_status/fig_branch/fig_closing are now built once in app.py's
    # cached pipeline (_cached_dashboard_charts), not rebuilt here on every
    # rerun -- see that function's docstring for why.
    st.markdown('<div class="section-label">Portfolio Analysis</div>', unsafe_allow_html=True)

    col_bar, col_hbar, col_lcc = st.columns([2, 2, 1])
    with col_bar:
        _chart_card(fig_status)
    with col_hbar:
        _chart_card(fig_branch)
    with col_lcc:
        lcc_val   = fmt_value(metrics["LCC%"][0], "pct")
        lcc_mom   = metrics["LCC%"][1]
        lcc_arrow = "▲" if lcc_mom >= 0 else "▼"
        lcc_cls   = "kpi-mom-up" if lcc_mom >= 0 else "kpi-mom-down"
        sma2_val  = fmt_value(metrics.get("SMA-2 %", (0, 0))[0], "pct")
        sma2_mom  = metrics.get("SMA-2 %", (0, 0))[1]
        sma2_arrow = "▲" if sma2_mom >= 0 else "▼"
        sma2_cls  = "kpi-mom-down" if sma2_mom >= 0 else "kpi-mom-up"
        st.markdown(f"""
        <div class="kpi-card" style="display:flex;flex-direction:column;
             justify-content:center;align-items:center;text-align:center;margin-top:0;margin-bottom:12px;">
          <div class="kpi-label">LCC %</div>
          <div style="font-size:36px;font-weight:800;color:#111;line-height:1.1;">{lcc_val}</div>
          <div class="kpi-mom">MoM <span class="{lcc_cls}">{lcc_arrow} {abs(lcc_mom):.2f}%</span></div>
        </div>
        <div class="kpi-card" style="display:flex;flex-direction:column;
             justify-content:center;align-items:center;text-align:center;margin-top:0;">
          <div class="kpi-label">SMA-2 %</div>
          <div style="font-size:36px;font-weight:800;color:#ef4444;line-height:1.1;">{sma2_val}</div>
          <div class="kpi-mom">MoM <span class="{sma2_cls}">{sma2_arrow} {abs(sma2_mom):.2f}%</span></div>
        </div>
        """, unsafe_allow_html=True)

    col_trend, col_bot = st.columns([3, 2])
    with col_trend:
        _chart_card(fig_closing)
    with col_bot:
        st.markdown("<br>", unsafe_allow_html=True)
        _kpi_row(_KPI_BOT, metrics)
        _kpi_row(_KPI_EXPOSURE, metrics)

    # ── HTML export ─────────────────────────────────────────────────────────
    _divider("24px 0 16px 0")
    col_dl, _ = st.columns([1, 3])
    with col_dl:
        filters_applied = {
            "Region": sel_region, "Branch": sel_branch,
            "Loan Status": sel_status, "Year Month": str(curr_month),
        }
        _html_cache_key = (data_version, sel_region, sel_branch, sel_status, segment)
        html_content = _cached_html_export(
            _html_cache_key, build_html_export,
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
            width='stretch',
        )
