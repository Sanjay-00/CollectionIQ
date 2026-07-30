import os

import pandas as pd
import streamlit as st

from ui.components import _send_report_email, _esc


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
        Generates a board-ready HTML report covering nearly the full analysis toolkit:
        verdict-first AI narrative, risk signals, month-over-month NPA/SMA-2 movement,
        embedded charts, region/segment/branch breakdowns, at-risk and fleet exposure lists,
        repossession candidates, good-customer retention targets, new advances (originations)
        breakdowns, and executive/branch leaderboards. Download as HTML or send via email.
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Section toggles ──────────────────────────────────────────────────────
    # Core (default ON, always visible) vs Additional (default OFF, tucked
    # into a collapsed expander) -- a flat 22-checkbox grid was the actual
    # source of "the report feels too long," not just needing better columns.
    # Core = what every monthly board report needs; Additional = opt-in
    # deep-dive detail a leader checks only when this specific report calls
    # for it.
    st.markdown('<div class="section-label" style="margin-top:20px;">Report Sections</div>', unsafe_allow_html=True)
    has_prev = len(df_prev) > 0

    def _col_caption(text: str) -> None:
        st.markdown(
            f'<div style="font-size:10px;font-weight:700;color:#9ca3af;'
            f'text-transform:uppercase;letter-spacing:1px;margin-bottom:6px;">{text}</div>',
            unsafe_allow_html=True,
        )

    with st.container(border=True):
        st.markdown(
            '<div style="font-size:13px;font-weight:800;color:#111827;margin-bottom:2px;">🎯 Core Sections</div>'
            '<div style="font-size:11px;color:#6b7280;margin-bottom:16px;">'
            'Included in every report by default : uncheck anything you don\'t need this month.</div>',
            unsafe_allow_html=True,
        )
        core_c1, core_c2, core_c3 = st.columns(3)
        with core_c1:
            _col_caption("Health & Verdict")
            inc_health  = st.checkbox("Portfolio Health",    value=True, key="rpt_health")
            inc_verdict = st.checkbox("Good vs Bad Verdict", value=True, key="rpt_verdict")
            inc_flags   = st.checkbox("Risk Flags",          value=True, key="rpt_flags")
        with core_c2:
            _col_caption("Movement & Region")
            inc_movement = st.checkbox("NPA & SMA-2 Movement", value=True, key="rpt_movement",
                                        help="This-month vs last-month NPA/SMA-2 counts, deltas, and %change - portfolio, region, branch, and executive")
            inc_migrate  = st.checkbox(
                "Bucket Migration", value=has_prev, key="rpt_migrate",
                disabled=not has_prev, help="Upload previous month file to enable",
            )
            inc_region   = st.checkbox("Region Scorecard", value=True, key="rpt_region")
        with core_c3:
            _col_caption("Business & Leaders")
            inc_new_advances = st.checkbox("New Advances (Business)", value=True, key="rpt_new_advances",
                                            help="New business funded this reporting month - accounts, funded amount, segment breakdown, MoM comparison")
            inc_branch = st.checkbox("Branch Performance", value=True, key="rpt_branch")
            inc_exec   = st.checkbox("Executive Rankings", value=True, key="rpt_exec")

    _ADDITIONAL_KEYS = [
        "rpt_indicators", "rpt_product", "rpt_overdue_demand",
        "rpt_quadrant", "rpt_concentration",
        "rpt_new_advances_trend", "rpt_new_advances_dim",
        "rpt_top_accounts", "rpt_fleet", "rpt_repo", "rpt_good",
        "rpt_recovery", "rpt_exec_strike", "rpt_ai",
    ]
    with st.expander(f"➕ Additional Sections  ({len(_ADDITIONAL_KEYS)} more available)", expanded=False):
        st.caption("✍️ Deep-dive detail: check whatever this specific report needs.")

        # Select All / Clear All MUST render (and write to st.session_state)
        # before ANY of _ADDITIONAL_KEYS' checkboxes are instantiated below --
        # Streamlit raises StreamlitAPIException ("st.session_state.X cannot
        # be modified after the widget with key X is instantiated") if you
        # write to a widget's session_state key after that widget has already
        # been created in the same script run. This used to sit inside add_c1,
        # AFTER 3 of the checkboxes it modifies (rpt_indicators, rpt_product,
        # rpt_overdue_demand) -- clicking either button raised that exception
        # on the very first key in the loop and silently updated nothing, for
        # ALL 14 checkboxes, not just those 3 (real bug, reported by users).
        # No st.rerun() needed either: since these run before the checkboxes
        # exist yet this pass, the checkboxes below pick up the new values in
        # this SAME script run.
        add_bulk_c1, add_bulk_c2, _add_bulk_spacer = st.columns([1, 1, 3])
        with add_bulk_c1:
            if st.button("Select All", key="rpt_additional_select_all"):
                for k in _ADDITIONAL_KEYS:
                    st.session_state[k] = True
        with add_bulk_c2:
            if st.button("Clear All", key="rpt_additional_clear_all"):
                for k in _ADDITIONAL_KEYS:
                    st.session_state[k] = False

        add_c1, add_c2, add_c3, add_c4, add_c5 = st.columns(5)
        with add_c1:
            _col_caption("Risk & Segments")
            inc_indicators = st.checkbox("Risk Indicators", value=False, key="rpt_indicators",
                                          help="Early-warning signals: SMA-1 pool, fresh NPA formation, chronic defaulters, non-starters, co-lending risk")
            inc_product = st.checkbox("Segment NPA Breakdown", value=False, key="rpt_product")
            inc_overdue_demand = st.checkbox("Overdue vs Month Demand", value=False, key="rpt_overdue_demand")
        with add_c2:
            _col_caption("Charts")
            inc_quadrant = st.checkbox("Branch Quadrant Chart", value=False, key="rpt_quadrant",
                                        help="Collection% vs NPA% scatter, bubble = SOH")
            inc_concentration = st.checkbox("Concentration Map", value=False, key="rpt_concentration")
        with add_c3:
            _col_caption("New Advances Detail")
            inc_new_advances_trend = st.checkbox("New Advances Trend (36mo)", value=False, key="rpt_new_advances_trend",
                                                  help="New advances by month, last 36 months")
            inc_new_advances_dim = st.checkbox("New Advances by Region/Branch/Exec", value=False, key="rpt_new_advances_dim",
                                                help="Top 5 Regions, Branches, and Executives by new advances this month")
        with add_c4:
            _col_caption("Account Lists")
            inc_top_acc = st.checkbox("Top At-Risk Accounts", value=False, key="rpt_top_accounts")
            inc_fleet   = st.checkbox("Fleet Exposure", value=False, key="rpt_fleet")
            inc_repo    = st.checkbox("Repossession Candidates", value=False, key="rpt_repo")
            inc_good    = st.checkbox("Good Customers", value=False, key="rpt_good")
        with add_c5:
            _col_caption("Executive Extras & AI")
            inc_recovery = st.checkbox("Executive Recovery", value=False, key="rpt_recovery",
                                        help="Rescued vs slipped accounts per executive - behavior signal, distinct from collection%")
            inc_exec_strike = st.checkbox("Executive Rankings (Strike %)", value=False, key="rpt_exec_strike",
                                           help="Same executives, ranked by Strike % instead of Collection % - who's actually current on installment obligation this month")
            inc_ai = st.checkbox("AI Summary", value=False, key="rpt_ai",
                                  help="Uncheck to skip Gemini and generate a faster, pandas-only report")

    # ── Email (optional) ─────────────────────────────────────────────────────
    smtp_ok = bool(os.environ.get("SMTP_HOST", ""))
    if smtp_ok:
        rpt_email_to = st.text_input(
            "Send report to (email address)",
            placeholder="manager@company.com, head@company.com",
            key="rpt_email_to",
            help="Separate multiple addresses with a comma",
        )
    else:
        rpt_email_to = ""
        st.caption("Email not configured  -  add SMTP_HOST / SMTP_USER / SMTP_PASS to .env to enable.")

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
        if st.session_state.get("rpt_health"):        enabled_sections.append("portfolio_health")
        if st.session_state.get("rpt_verdict"):       enabled_sections.append("verdict")
        if st.session_state.get("rpt_flags"):         enabled_sections.append("risk_flags")
        if st.session_state.get("rpt_indicators"):    enabled_sections.append("risk_indicators")
        if st.session_state.get("rpt_migrate"):       enabled_sections.append("bucket_migration")
        if st.session_state.get("rpt_movement"):      enabled_sections.append("npa_sma2_movement")
        if st.session_state.get("rpt_quadrant"):      enabled_sections.append("branch_quadrant")
        if st.session_state.get("rpt_concentration"): enabled_sections.append("concentration")
        if st.session_state.get("rpt_region"):        enabled_sections.append("region_scorecard")
        if st.session_state.get("rpt_overdue_demand"): enabled_sections.append("overdue_demand")
        if st.session_state.get("rpt_new_advances"):  enabled_sections.append("new_advances")
        if st.session_state.get("rpt_new_advances_trend"): enabled_sections.append("new_advances_trend")
        if st.session_state.get("rpt_new_advances_dim"):   enabled_sections.append("new_advances_by_dimension")
        if st.session_state.get("rpt_product"):       enabled_sections.append("product_analysis")
        if st.session_state.get("rpt_top_accounts"):  enabled_sections.append("top_accounts")
        if st.session_state.get("rpt_fleet"):         enabled_sections.append("fleet_exposure")
        if st.session_state.get("rpt_repo"):          enabled_sections.append("repossession")
        if st.session_state.get("rpt_good"):          enabled_sections.append("good_customers")
        if st.session_state.get("rpt_branch"):        enabled_sections.append("branch_performance")
        if st.session_state.get("rpt_recovery"):      enabled_sections.append("executive_recovery")
        if st.session_state.get("rpt_exec"):          enabled_sections.append("executive_rankings")
        if st.session_state.get("rpt_exec_strike"):   enabled_sections.append("executive_strike_rankings")

        _skip_ai = not st.session_state.get("rpt_ai", True)
        _spinner_msg = "Generating report (pandas only)..." if _skip_ai else "Running Portfolio Intelligence Agent (30 - 60 seconds)..."
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
            f'<span style="color:#c9d1d9;font-size:13px;line-height:1.6;">{_esc(b)}</span>'
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
            f'<div style="padding:8px 0;border-bottom:1px solid #21262d;font-size:13px;color:#8b949e;">{_esc(l)}</div>'
            for l in lines
        )
        st.markdown(f"""
        <div class="obs-card" style="margin-top:12px;">
          <div class="obs-title">Prioritized Action Plan</div>
          {action_html}
        </div>""", unsafe_allow_html=True)
