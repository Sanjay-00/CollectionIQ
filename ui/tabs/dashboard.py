
import pandas as pd
import streamlit as st

from utils import build_html_export, fmt_value
from ui.components import _kpi_card_html, _divider
from ui.action_overview import render_summary

_KIND = {
    "Month Demand": "money", "Total Collection": "money", "Collection %": "pct",
    "Strike %": "pct", "NPA %": "pct", "NPA % (SOH)": "pct", "Hard Bucket %": "pct", "SMA-2 %": "pct",
    "Count": "count", "SOH": "money", "LCC%": "pct", "CMD %": "pct", "Delinquency %": "pct",
}
# Scoreboard rows. CMD % stays in the HTML export but off the screen until
# its business definition is confirmed.
_KPI_COLLECTIONS = ["Month Demand", "Total Collection", "Collection %", "Strike %", "LCC%"]
_KPI_RISK        = ["Count", "SOH", "Delinquency %", "SMA-2 %", "NPA %", "NPA % (SOH)", "Hard Bucket %"]
_INVERSE_MOM     = {"NPA %", "NPA % (SOH)", "Hard Bucket %", "SMA-2 %", "Delinquency %"}


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
    date_from=None,
    alerts_prev: list | None = None,
) -> None:
    # ── Scoreboard: collections, then book and risk ──────────────────────────
    st.markdown('<div class="section-label">Collections</div>', unsafe_allow_html=True)
    _kpi_row(_KPI_COLLECTIONS, metrics)
    st.markdown('<div class="section-label" style="margin-top:14px;">Book and Risk</div>', unsafe_allow_html=True)
    _kpi_row(_KPI_RISK, metrics, count_deltas={"NPA %": _npa_count_delta(df_curr, df_prev)})

    # ── Summary of the whole portfolio, each section linking to its tab ──────
    filter_key = f"{sel_region}|{sel_branch}|{sel_status}|{','.join(sorted(segment))}|{date_from}"
    render_summary(df_curr, df_prev, data_version, filter_key, curr_month, alerts, alerts_prev or [])

    # ── HTML export ─────────────────────────────────────────────────────────
    _divider("24px 0 16px 0")
    col_dl, _ = st.columns([1, 3])
    with col_dl:
        filters_applied = {
            "Region": sel_region, "Branch": sel_branch,
            "Loan Status": sel_status, "Year Month": str(curr_month),
        }
        # Built on click (the charts live only in this export now).
        def _export() -> bytes:
            return build_html_export(
                df_curr, df_prev, metrics, fig_status, fig_branch, fig_closing,
                filters_applied, curr_month=curr_month, alerts=alerts,
                scorecard_df=scorecard_df, roll_rate_meta=rr_meta,
            ).encode("utf-8")
        st.download_button(
            label="⬇  Download Dashboard as HTML",
            data=_export,
            file_name=f"collectioniq_dashboard_{curr_month}.html",
            mime="text/html",
            width='stretch',
        )
