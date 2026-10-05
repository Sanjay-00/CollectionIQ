"""Report tab: the monthly report (report_agent/story.py) as HTML, PDF and an
Excel annex; optional detail sections; email; and branch packs (one report
per branch, zipped)."""
import os

import pandas as pd
import streamlit as st

from ui.components import _esc, _send_report_email

_PRESET_HELP = {
    "Regional": "For regional managers (default): whole picture plus branches, roll rates by branch, "
                "units needing attention, why it's happening, alerts and priorities.",
    "Leadership": "For leaders: the whole portfolio on 2 to 3 pages.",
    "Branch": "For one branch manager: their executives, roll rates and call lists. Pick the branch in the sidebar.",
}
_XLSX = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
_PACK_SECONDS_PER_BRANCH = 1.2      # rough, for the "this will take about..." note


def render_report_tab(
    df_curr: pd.DataFrame,
    df_prev: pd.DataFrame,
    curr_month: str,
    prev_month: str | None,
    sel_region: str,
    sel_branch: str,
    sel_status: str,
    scorecard_df=None,
    rr_meta: dict | None = None,
) -> None:
    from report_agent.story import OPTIONAL_SECTIONS, PRESETS

    st.markdown(
        '<div class="ai-panel"><div class="ai-title">Monthly Report</div><div class="ai-subtitle">'
        "One report, three formats: HTML (email-safe), PDF and an Excel annex with every table in full and "
        "every call list with its proof columns. Built from the same numbers as the Dashboard."
        "</div></div>", unsafe_allow_html=True)

    c1, c2 = st.columns([2, 3])
    with c1:
        preset = st.radio("Who is it for?", PRESETS, index=0, key="rpt2_preset")
    with c2:
        st.markdown(f'<div style="margin-top:28px;font-size:13px;color:#374151;">{_esc(_PRESET_HELP[preset])}</div>',
                    unsafe_allow_html=True)
    labels = {label: key for key, (label, _) in OPTIONAL_SECTIONS.items()}
    picked = st.multiselect("Add detail sections (optional)", list(labels), default=[], key="rpt2_extras",
                            help="Each adds its top rows to the report; the full lists go in the Excel annex.")
    extras = [labels[x] for x in picked]
    use_ai = st.checkbox("Add an AI-written opening paragraph (optional)", value=False, key="rpt2_ai",
                         help="Gemini rewrites the summary sentences into a paragraph. Every number it uses is "
                              "checked against the data; if any doesn't match, the paragraph is dropped.")
    filters = {"Region": sel_region, "Branch": sel_branch, "Loan Status": sel_status}

    if preset == "Branch" and sel_branch in ("All", "", None):
        st.info("Pick one branch in the sidebar's Branch filter for a single Branch report, "
                "or make a pack for every branch below.")
    elif st.button("Generate report", type="primary", key="rpt2_generate"):
        _generate(df_curr, df_prev, curr_month, prev_month, preset, filters, extras, use_ai)

    _show_result()
    _branch_packs(df_curr, df_prev, curr_month, prev_month, filters, extras)


def _generate(df_curr, df_prev, curr_month, prev_month, preset, filters, extras, use_ai) -> None:
    from report_agent import render
    from report_agent.story import build_report
    from smart_alerts import run_all_alerts
    with st.spinner("Building the report..."):
        try:
            alerts = run_all_alerts(df_curr, as_of=curr_month)
            alerts_prev = run_all_alerts(df_prev, as_of=prev_month) if len(df_prev) else []
            model = build_report(df_curr, df_prev, curr_month, prev_month, preset, filters, alerts, alerts_prev, extras)
        except ValueError as e:
            st.warning(str(e))
            return
        ai_note = ""
        if use_ai:
            from report_agent.ai_summary import polish
            model["ai_summary"], ai_note = polish(model)
        st.session_state["rpt2"] = {
            "preset": preset, "month": curr_month, "ai_note": ai_note,
            "html": render.to_html(model), "pdf": render.to_pdf(model), "xlsx": render.to_excel(model),
        }


def _show_result() -> None:
    out = st.session_state.get("rpt2")
    if not out:
        return
    if out["ai_note"]:
        (st.success if out["ai_note"].startswith("AI summary added") else st.warning)(out["ai_note"])
    base = f"{out['preset'].lower()}_report_{out['month']}".replace(" ", "_")
    d1, d2, d3, _ = st.columns([1, 1, 1, 2])
    d1.download_button("⬇ HTML", out["html"].encode("utf-8"), f"{base}.html", "text/html",
                       key="rpt2_dl_html", width="stretch")
    d2.download_button("⬇ PDF", out["pdf"], f"{base}.pdf", "application/pdf", key="rpt2_dl_pdf", width="stretch")
    d3.download_button("⬇ Excel annex", out["xlsx"], f"{base}.xlsx", _XLSX, key="rpt2_dl_xlsx", width="stretch")

    if os.environ.get("SMTP_HOST", ""):
        e1, e2 = st.columns([3, 1])
        to = e1.text_input("Send the HTML report to", placeholder="manager@company.com, head@company.com",
                           key="rpt2_email_to", help="Separate several addresses with a comma.")
        e2.markdown('<div style="height:28px;"></div>', unsafe_allow_html=True)
        if e2.button("📧 Send", key="rpt2_send", width="stretch"):
            if not to.strip():
                st.warning("Enter an email address.")
            else:
                ok, err = _send_report_email(out["html"], to.strip(), out["month"])
                (st.success(f"Sent to {to}") if ok else st.error(f"Failed: {err}"))
    else:
        st.caption("Email sending is off: add SMTP_HOST / SMTP_USER / SMTP_PASS to .env to turn it on.")

    with st.expander("Preview", expanded=True):
        st.html(out["html"])


def _branch_packs(df_curr, df_prev, curr_month, prev_month, filters, extras) -> None:
    from report_agent.packs import branches_in, build_branch_packs
    branches = branches_in(df_curr)
    if not branches:
        return
    with st.expander(f"Branch packs: a Branch report for each of the {len(branches)} branches in view", expanded=False):
        secs = round(len(branches) * _PACK_SECONDS_PER_BRANCH)
        st.caption(f"One PDF (and Excel annex with call lists) per branch, zipped, for regional managers to send "
                   f"to each branch manager. About {secs} seconds"
                   + ("; filter one region in the sidebar to make it faster." if len(branches) > 15 else "."))
        with_excel = st.checkbox("Include each branch's Excel annex", value=True, key="rpt2_pack_xlsx")
        if st.button("Make branch packs", key="rpt2_make_packs"):
            bar = st.progress(0.0, text="Starting...")
            data, done = build_branch_packs(
                df_curr, df_prev, curr_month, prev_month, filters, extras, with_excel,
                progress=lambda i, n, b: bar.progress(i / n, text=f"{i} of {n}: {b}"))
            bar.empty()
            st.session_state["rpt2_packs"] = {"zip": data, "n": len(done), "month": curr_month}
        packs = st.session_state.get("rpt2_packs")
        if packs:
            st.download_button(f"⬇ Branch packs ({packs['n']} branches, ZIP)", packs["zip"],
                               f"branch_packs_{packs['month']}.zip", "application/zip", key="rpt2_dl_packs")
