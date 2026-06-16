import os

import pandas as pd
import streamlit as st

from ui.components import _send_report_email


def render_report_tab(
    df_curr: pd.DataFrame,
    df_prev: pd.DataFrame,
    curr_month: str,
    prev_month: str | None,
    sel_region: str,
    sel_branch: str,
    sel_status: str,
    scorecard_df,
    rr_meta: dict | None,
) -> None:
    from report_agent.graph import run_report

    st.markdown("""
    <div class="ai-panel">
      <div class="ai-title">Monthly Portfolio Intelligence Report</div>
      <div class="ai-subtitle">
        Generates a board-ready HTML report with AI executive narrative, branch rankings,
        field executive scorecard, bucket migration analysis, and 5 prioritized action items.
        Download as HTML or send via email.
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Section toggles ──────────────────────────────────────────────────────
    st.markdown('<div class="section-label" style="margin-top:20px;">Report Sections</div>', unsafe_allow_html=True)
    has_prev   = len(df_prev) > 0
    rpt_c1, rpt_c2, rpt_c3 = st.columns(3)
    with rpt_c1:
        inc_health = st.checkbox("Portfolio Health",    value=True, key="rpt_health")
        inc_flags  = st.checkbox("Risk Flags",          value=True, key="rpt_flags")
    with rpt_c2:
        inc_migrate = st.checkbox(
            "Bucket Migration", value=has_prev, key="rpt_migrate",
            disabled=not has_prev, help="Upload previous month file to enable",
        )
        inc_branch = st.checkbox("Branch Performance",  value=True, key="rpt_branch")
    with rpt_c3:
        inc_exec   = st.checkbox("Executive Rankings",  value=True, key="rpt_exec")
        inc_ai     = st.checkbox("AI Summary",          value=True, key="rpt_ai",
                                 help="Uncheck to skip Gemini and generate a faster, pandas-only report")

    # ── Email (optional) ─────────────────────────────────────────────────────
    smtp_ok = bool(os.environ.get("SMTP_HOST", ""))
    if smtp_ok:
        rpt_email_to = st.text_input(
            "Send report to (email address)",
            placeholder="manager@shriram.com, head@shriram.com",
            key="rpt_email_to",
            help="Separate multiple addresses with a comma",
        )
    else:
        rpt_email_to = ""
        st.caption("Email not configured — add SMTP_HOST / SMTP_USER / SMTP_PASS to .env to enable.")

    # ── Buttons ──────────────────────────────────────────────────────────────
    _gc, _sc, _ = st.columns([2, 1, 3])
    with _gc:
        rpt_btn = st.button("Generate Monthly Report", type="primary", key="rpt_generate", width='stretch')
    with _sc:
        rpt_send_btn = st.button(
            "📧 Send", key="rpt_send_only", width='stretch',
            disabled=not (smtp_ok and st.session_state.get("report_result", {}).get("html_report")),
        )

    # ── Send existing report ─────────────────────────────────────────────────
    if rpt_send_btn:
        _cached = st.session_state.get("report_result", {})
        if not rpt_email_to.strip():
            st.warning("Enter an email address above.")
        else:
            with st.spinner("Sending report..."):
                _ok, _err = _send_report_email(_cached["html_report"], rpt_email_to.strip(), curr_month)
            if _ok:
                st.success(f"Report sent to: {rpt_email_to}")
            else:
                st.error(f"Failed: {_err}")

    # ── Generate ─────────────────────────────────────────────────────────────
    if rpt_btn:
        enabled_sections = []
        if st.session_state.get("rpt_health"):  enabled_sections.append("portfolio_health")
        if st.session_state.get("rpt_flags"):   enabled_sections.append("risk_flags")
        if st.session_state.get("rpt_migrate"): enabled_sections.append("bucket_migration")
        if st.session_state.get("rpt_branch"):  enabled_sections.append("branch_performance")
        if st.session_state.get("rpt_exec"):    enabled_sections.append("executive_rankings")

        _skip_ai = not st.session_state.get("rpt_ai", True)
        _spinner_msg = "Generating report (pandas only)..." if _skip_ai else "Running Portfolio Intelligence Agent (30–60 seconds)..."
        with st.spinner(_spinner_msg):
            _rpt_result = run_report(
                df_curr=df_curr, df_prev=df_prev,
                curr_month=curr_month, prev_month=prev_month,
                enabled_sections=enabled_sections,
                filters_applied={"Region": sel_region, "Branch": sel_branch, "Loan Status": sel_status},
                email_to=rpt_email_to,
                skip_ai=_skip_ai,
            )
        st.session_state["report_result"] = _rpt_result

    # ── Display result ────────────────────────────────────────────────────────
    _rpt = st.session_state.get("report_result")
    if not _rpt:
        return

    if _rpt.get("error"):
        st.error(f"Report generation failed: {_rpt['error']}")
        return

    if not _rpt.get("html_report"):
        return

    st.success("Report generated successfully.")
    if _rpt.get("ai_skipped") and not _rpt.get("skip_ai"):
        st.warning("AI summary could not be generated (Gemini unavailable). Report sent without it.")
    if _rpt.get("email_sent"):
        st.info(f"Report emailed to: {_rpt.get('email_to', '')}")
    elif _rpt.get("email_error") and os.environ.get("SMTP_HOST", "") and _rpt.get("email_to", ""):
        st.warning(f"Email failed: {_rpt['email_error']}")

    rpt_dl_col, _ = st.columns([1, 3])
    with rpt_dl_col:
        st.download_button(
            label="⬇  Download Intelligence Report (HTML)",
            data=_rpt["html_report"].encode("utf-8"),
            file_name=f"portfolio_intelligence_{curr_month}.html",
            mime="text/html",
            width='stretch',
        )

    if _rpt.get("executive_narrative"):
        bullets = [l.strip().lstrip("- ").strip() for l in _rpt["executive_narrative"].split("\n") if l.strip()]
        narrative_html = "".join(
            f'<div style="display:flex;gap:8px;padding:5px 0;border-bottom:1px solid #1e293b;">'
            f'<span style="color:#FFC000;font-size:14px;font-weight:900;flex-shrink:0;">&#8226;</span>'
            f'<span style="color:#c9d1d9;font-size:13px;line-height:1.6;">{b}</span>'
            f'</div>'
            for b in bullets
        )
        st.markdown(f"""
        <div class="obs-card">
          <div class="obs-title">AI Executive Narrative</div>
          <div style="margin-top:8px;">{narrative_html}</div>
        </div>""", unsafe_allow_html=True)

    if _rpt.get("action_plan"):
        lines = [l.strip() for l in _rpt["action_plan"].split("\n") if l.strip()][:5]
        action_html = "".join(
            f'<div style="padding:8px 0;border-bottom:1px solid #21262d;font-size:13px;color:#8b949e;">{l}</div>'
            for l in lines
        )
        st.markdown(f"""
        <div class="obs-card" style="margin-top:12px;">
          <div class="obs-title">Prioritized Action Plan</div>
          {action_html}
        </div>""", unsafe_allow_html=True)
