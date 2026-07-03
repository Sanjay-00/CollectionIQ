import pandas as pd
import streamlit as st

from ui.components import _dl_btn, _safe_df, _static_kpi_card_html, _empty_state


def render_scorecard_tab(df_curr: pd.DataFrame, scorecard_df) -> None:
    if "MNT NAME" not in df_curr.columns:
        _empty_state(
            "👤", "MNT NAME column not found",
            "The Executive Scorecard requires a field executive column (MNT NAME) in your LCC extract.",
        )
        return

    if scorecard_df is None or len(scorecard_df) == 0:
        st.info("Not enough data per executive to build scorecard (minimum 5 accounts required).")
        return

    from analysis.executive_scorecard import build_scorecard_table_html, rank_by_metric

    avg_coll   = scorecard_df["Collection %"].mean()
    avg_strike = scorecard_df["Strike Rate %"].mean()
    avg_npa    = scorecard_df["NPA %"].mean() if "NPA %" in scorecard_df.columns else 0.0

    # ── Summary KPIs ────────────────────────────────────────────────────────
    st.markdown('<div class="section-label">Performance Summary</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(_static_kpi_card_html("Avg Collection %", f"{avg_coll:.1f}%", f"{len(scorecard_df)} executives ranked"), unsafe_allow_html=True)
    with c2:
        strike_color = "#16a34a" if avg_strike >= 70 else "#d97706" if avg_strike >= 50 else "#dc2626"
        st.markdown(_static_kpi_card_html("Avg Strike Rate %", f"{avg_strike:.1f}%", "EMI obligation cleared this month", color=strike_color), unsafe_allow_html=True)
    with c3:
        npa_color = "#16a34a" if avg_npa < 5 else "#d97706" if avg_npa < 10 else "#dc2626"
        st.markdown(_static_kpi_card_html("Avg NPA %", f"{avg_npa:.1f}%", "Across all executives", color=npa_color), unsafe_allow_html=True)

    # ── Rank-by selector  -  Collection % is the default; Strike % re-ranks the
    # same executives independently, so a lead can also see who's actually
    # current on installment obligation this month, not just who collected
    # the most (which can be padded by reserve/settlement-driven recoveries).
    st.markdown('<div class="section-label" style="margin-top:24px;">Full Rankings</div>', unsafe_allow_html=True)
    rank_metric = st.radio(
        "Rank executives by", ["Collection %", "Strike Rate %"],
        horizontal=True, key="scorecard_rank_metric",
    )
    ranked_df = scorecard_df if rank_metric == "Collection %" else rank_by_metric(scorecard_df, "Strike Rate %")

    top_count = (ranked_df["Tier"] == "top").sum()
    bot_count = (ranked_df["Tier"] == "bottom").sum()
    rc1, rc2 = st.columns(2)
    with rc1:
        st.markdown(_static_kpi_card_html("Top Performers", top_count, f"Top 25% by {rank_metric.lower()}", color="#16a34a"), unsafe_allow_html=True)
    with rc2:
        st.markdown(_static_kpi_card_html("Need Attention", bot_count, f"Bottom 25% by {rank_metric.lower()}", color="#dc2626"), unsafe_allow_html=True)

    with st.expander(f"View Executive Performance Table ({len(ranked_df)} executives)", expanded=True):
        st.markdown(build_scorecard_table_html(ranked_df), unsafe_allow_html=True)
        _dl_btn(
            ranked_df.drop(columns=["Tier"], errors="ignore"),
            "executive_scorecard.xlsx", "dl_scorecard",
        )
