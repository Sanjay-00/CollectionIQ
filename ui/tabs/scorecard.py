import pandas as pd
import streamlit as st

from ui.components import _dl_btn, _safe_df


def render_scorecard_tab(df_curr: pd.DataFrame, scorecard_df) -> None:
    if "MNT NAME" not in df_curr.columns:
        st.markdown("""
        <div class="empty-state">
          <div class="empty-state-icon">👤</div>
          <div class="empty-state-title">MNT NAME column not found</div>
          <div class="empty-state-sub">The Executive Scorecard requires a field executive column (MNT NAME) in your LCC extract.</div>
        </div>
        """, unsafe_allow_html=True)
        return

    if scorecard_df is None or len(scorecard_df) == 0:
        st.info("Not enough data per executive to build scorecard (minimum 5 accounts required).")
        return

    from analysis.executive_scorecard import build_scorecard_table_html

    top_count  = (scorecard_df["Tier"] == "top").sum()
    bot_count  = (scorecard_df["Tier"] == "bottom").sum()
    avg_coll   = scorecard_df["Collection %"].mean()
    avg_strike = scorecard_df["Strike Rate %"].mean()
    avg_npa    = scorecard_df["NPA %"].mean() if "NPA %" in scorecard_df.columns else 0.0

    # ── Summary KPIs ────────────────────────────────────────────────────────
    st.markdown('<div class="section-label">Performance Summary</div>', unsafe_allow_html=True)
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(f"""
        <div class="kpi-card" style="border-top-color:#16a34a;">
          <div class="kpi-label">Top Performers</div>
          <div class="kpi-value" style="color:#16a34a;">{top_count}</div>
          <div class="kpi-mom">Top 25% by collection %</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="kpi-card" style="border-top-color:#dc2626;">
          <div class="kpi-label">Need Attention</div>
          <div class="kpi-value" style="color:#dc2626;">{bot_count}</div>
          <div class="kpi-mom">Bottom 25% by collection %</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="kpi-card">
          <div class="kpi-label">Avg Collection %</div>
          <div class="kpi-value">{avg_coll:.1f}%</div>
          <div class="kpi-mom">{len(scorecard_df)} executives ranked</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        strike_color = "#16a34a" if avg_strike >= 70 else "#d97706" if avg_strike >= 50 else "#dc2626"
        st.markdown(f"""
        <div class="kpi-card" style="border-top-color:{strike_color};">
          <div class="kpi-label">Avg Strike Rate %</div>
          <div class="kpi-value" style="color:{strike_color};">{avg_strike:.1f}%</div>
          <div class="kpi-mom">EMI obligation cleared this month</div>
        </div>""", unsafe_allow_html=True)
    with c5:
        npa_color = "#16a34a" if avg_npa < 5 else "#d97706" if avg_npa < 10 else "#dc2626"
        st.markdown(f"""
        <div class="kpi-card" style="border-top-color:{npa_color};">
          <div class="kpi-label">Avg NPA %</div>
          <div class="kpi-value" style="color:{npa_color};">{avg_npa:.1f}%</div>
          <div class="kpi-mom">Across all executives</div>
        </div>""", unsafe_allow_html=True)

    # ── Scorecard table ──────────────────────────────────────────────────────
    st.markdown('<div class="section-label" style="margin-top:24px;">Full Rankings</div>', unsafe_allow_html=True)
    with st.expander(f"View Executive Performance Table ({len(scorecard_df)} executives)", expanded=True):
        st.markdown(build_scorecard_table_html(scorecard_df), unsafe_allow_html=True)
        _dl_btn(
            scorecard_df.drop(columns=["Tier"], errors="ignore"),
            "executive_scorecard.xlsx", "dl_scorecard",
        )
