import pandas as pd
import streamlit as st


def render_sidebar(df_curr_raw: pd.DataFrame, curr_month: str) -> tuple[str, str, str]:
    """Render sidebar filter controls. Returns (sel_region, sel_branch, sel_status)."""
    with st.sidebar:
        st.markdown('<div class="filter-header">⚙ FILTERS</div>', unsafe_allow_html=True)

        st.markdown(f"""
        <div style="background:#2a2a2a;border-radius:8px;padding:10px 14px;margin-bottom:16px;">
          <div style="font-size:10px;color:#888;font-weight:600;text-transform:uppercase;letter-spacing:1px;">Reporting Month</div>
          <div style="font-size:18px;font-weight:800;color:#FFC000;margin-top:2px;">{curr_month}</div>
        </div>
        """, unsafe_allow_html=True)

        regions = ["All"] + sorted(df_curr_raw["RegionName"].dropna().unique().tolist())
        sel_region = st.selectbox("Region", regions)

        # Reset branch when region changes
        if st.session_state.get("_prev_region") != sel_region:
            st.session_state["_sel_branch"] = "All"
            st.session_state["_prev_region"] = sel_region

        df_for_branch = df_curr_raw if sel_region == "All" else df_curr_raw[df_curr_raw["RegionName"] == sel_region]
        branches = ["All"] + sorted(df_for_branch["Unit"].dropna().unique().tolist())
        default_branch = st.session_state.get("_sel_branch", "All")
        branch_idx = branches.index(default_branch) if default_branch in branches else 0
        sel_branch = st.selectbox("Branch", branches, index=branch_idx)
        st.session_state["_sel_branch"] = sel_branch

        statuses = ["All"] + sorted(df_curr_raw["Loan Status"].dropna().unique().tolist())
        sel_status = st.selectbox("Loan Status", statuses)

        st.markdown('<div style="border-top:1px solid #222;margin:14px 0;"></div>', unsafe_allow_html=True)
        st.markdown(
            f'<div style="font-size:11px;color:#555;text-align:center;">'
            f'{len(df_curr_raw):,} total records loaded</div>',
            unsafe_allow_html=True,
        )
        st.markdown("<div style='margin-top:12px;'></div>", unsafe_allow_html=True)
        if st.button("Clear cache & reload", width='stretch', key="clear_cache_btn"):
            st.cache_data.clear()
            for _k in ["df_curr_raw", "df_prev_raw", "ai_result", "report_result",
                       "_last_filter_key", "_sample_loaded", "_sel_branch", "_prev_region"]:
                st.session_state.pop(_k, None)
            st.rerun()

    return sel_region, sel_branch, sel_status
