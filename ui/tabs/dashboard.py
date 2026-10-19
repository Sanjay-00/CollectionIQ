
import pandas as pd
import streamlit as st

from utils import fmt_value
from ui.components import _kpi_card_html
from ui.action_overview import render_summary

_KIND = {
    "Month Demand": "money", "Total Collection": "money", "Collection %": "pct",
    "Strike %": "pct", "NPA %": "pct", "NPA % (SOH)": "pct", "Hard Bucket %": "pct", "SMA-2 %": "pct",
    "Count": "count", "SOH": "money", "LCC%": "pct", "CMD %": "pct", "Delinquency %": "pct",
}
# Scoreboard rows. CMD % is computed (utils.compute_metrics) but not shown
# until its business definition is confirmed.
_KPI_COLLECTIONS = ["Month Demand", "Total Collection", "Collection %", "Strike %", "LCC%"]
_KPI_RISK        = ["Count", "SOH", "Delinquency %", "SMA-2 %", "NPA %", "NPA % (SOH)", "Hard Bucket %"]
_INVERSE_MOM     = {"NPA %", "NPA % (SOH)", "Hard Bucket %", "SMA-2 %", "Delinquency %"}


def _change(k: str, metrics: dict, prev_metrics: dict | None) -> tuple:
    """(change, unit): a % metric moves in points ("▲ 0.64 pts", as on every
    other tab and the report); an amount or count as a % change."""
    if _KIND[k] == "pct":
        before = (prev_metrics or {}).get(k)
        return (None if before is None else round(metrics[k][0] - before[0], 2)), " pts"
    return metrics[k][1], "%"


def _kpi_row(keys: list, metrics: dict, count_deltas: dict | None = None, prev_metrics: dict | None = None) -> None:
    count_deltas = count_deltas or {}
    cards = []
    for k in keys:
        if k not in metrics or k not in _KIND:
            continue
        change, unit = _change(k, metrics, prev_metrics)
        cards.append(_kpi_card_html(k, fmt_value(metrics[k][0], _KIND[k]), change, unit=unit,
                                    inverse=k in _INVERSE_MOM, zero_delta_bad=True, count_delta=count_deltas.get(k)))
    html = "".join(cards)
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
    prev_metrics: dict | None = None,
    data_version: int = 0,
    filter_key: str = "",
    alerts_prev: list | None = None,
) -> None:
    # ── Scoreboard: collections, then book and risk ──────────────────────────
    st.markdown('<div class="section-label">Collections</div>', unsafe_allow_html=True)
    _kpi_row(_KPI_COLLECTIONS, metrics, prev_metrics=prev_metrics)
    st.markdown('<div class="section-label" style="margin-top:14px;">Book and Risk</div>', unsafe_allow_html=True)
    _kpi_row(_KPI_RISK, metrics, count_deltas={"NPA %": _npa_count_delta(df_curr, df_prev)}, prev_metrics=prev_metrics)

    # ── Summary of the whole portfolio, each section linking to its tab ──────
    render_summary(df_curr, df_prev, data_version, filter_key, curr_month, alerts, alerts_prev or [])
