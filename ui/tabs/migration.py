import pandas as pd
import streamlit as st

from utils import fmt_value
from ui.components import _static_kpi_card_html, _chart_card, _empty_state


def render_migration_tab(
    df_curr: pd.DataFrame,
    df_prev_raw: pd.DataFrame,
    rr_matrix,
    rr_meta: dict | None,
) -> None:
    if len(df_prev_raw) == 0 or rr_meta is None:
        _empty_state(
            "📈", "Previous month data not loaded",
            "Upload a previous month LCC file alongside the current month file,<br>"
            "then click <strong>Generate Dashboard</strong> to see bucket migration and roll-rate analysis.",
        )
        return

    from analysis.roll_rate import build_roll_rate_heatmap

    # ── KPI row ──────────────────────────────────────────────────────────────
    st.markdown('<div class="section-label">Roll-Rate Summary</div>', unsafe_allow_html=True)
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
