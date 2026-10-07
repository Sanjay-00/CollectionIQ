"""Report tab: the monthly report (report_agent/story.py) as HTML, PDF and an
Excel annex; optional detail sections; email; and branch packs (one report
per branch, zipped)."""
import os

import pandas as pd
import streamlit as st

from ui.components import _esc, _send_report_email, section_label

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
    sel_zone: str = "All",
) -> None:
    from report_agent.story import MAIN_SECTIONS, OPTIONAL_SECTIONS, PRESETS

    # Streamlit forgets a widget's value when its tab isn't shown, so the picks
    # are kept in plain session keys (_rpt2_last_preset, _rpt2_ticks) and restored.
    last = st.session_state.get("_rpt2_last_preset")
    if "rpt2_preset" not in st.session_state and last in PRESETS:
        st.session_state["rpt2_preset"] = last
    filters = {"Zone": sel_zone, "Region": sel_region, "Branch": sel_branch, "Loan Status": sel_status}

    with st.container(border=True, key="rpt2_card"):
        st.markdown('<div class="rpt2-hero"><div class="rpt2-hero-title">Monthly Report</div>'
                    '<div class="rpt2-hero-sub">HTML (email safe), PDF and an Excel annex with every table in full, '
                    "built from the same numbers as the Dashboard.</div></div>", unsafe_allow_html=True)
        st.markdown(_step(1, "Who is it for?"), unsafe_allow_html=True)
        preset = st.segmented_control("Who is it for?", PRESETS, key="rpt2_preset",
                                      default=None if "rpt2_preset" in st.session_state else PRESETS[0],
                                      label_visibility="collapsed") or last or PRESETS[0]
        st.caption(_PRESET_HELP[preset])
        # A new preset ticks its own sections (the detail sections stay as picked).
        if st.session_state.get("_rpt2_last_preset") != preset:
            st.session_state["_rpt2_last_preset"] = preset
            _tick_preset(preset)

        saved = st.session_state.get("_rpt2_ticks", {})
        n_main = sum(bool(st.session_state.get(f"rpt2_sec_{k}", saved.get(f"rpt2_sec_{k}"))) for k in MAIN_SECTIONS)
        n_extra = sum(bool(st.session_state.get(f"rpt2_extra_{k}", saved.get(f"rpt2_extra_{k}")))
                      for k in OPTIONAL_SECTIONS)
        scope = " · ".join(f"{k}: {_esc(v)}" for k, v in filters.items() if v not in (None, "", "All")) or "Whole portfolio"
        st.markdown(
            f'<div class="rpt2-summary"><span><b>Scope</b> {scope}</span>'
            f'<span><b>Sections</b> {n_main} main' + (f" + {n_extra} detail" if n_extra else "") + "</span>"
            '<span><b>Formats</b> HTML · PDF · Excel</span></div>', unsafe_allow_html=True)

        st.markdown(_step(2, "What goes in"), unsafe_allow_html=True)
        with st.expander(f"Sections ({n_main} of {len(MAIN_SECTIONS)} ticked)", expanded=False):
            _checkbox_grid({k: label for k, label in MAIN_SECTIONS.items()}, "rpt2_sec_")
            st.button(f"Back to the {preset} sections", key="rpt2_defaults", on_click=_tick_preset, args=(preset,))
        with st.expander(f"Add detail sections ({n_extra} of {len(OPTIONAL_SECTIONS)} ticked)", expanded=False):
            st.caption("Each adds its top rows to the report; the full lists go in the Excel annex.")
            b1, b2, _ = st.columns([1, 1, 3])
            b1.button("Select all", key="rpt2_extra_all", on_click=_set_extras, args=(True,), width="stretch")
            b2.button("Clear all", key="rpt2_extra_none", on_click=_set_extras, args=(False,), width="stretch")
            _checkbox_grid({k: label for k, (label, _) in OPTIONAL_SECTIONS.items()}, "rpt2_extra_")

        main = [k for k in MAIN_SECTIONS if st.session_state.get(f"rpt2_sec_{k}")]
        extras = [k for k in OPTIONAL_SECTIONS if st.session_state.get(f"rpt2_extra_{k}")]
        st.markdown(_step(3, "Generate"), unsafe_allow_html=True)
        a, b = st.columns([3, 1], vertical_alignment="center")
        use_ai = a.checkbox("Add an AI-written opening paragraph", value=False, key="rpt2_ai",
                            help="Gemini rewrites the summary sentences into a paragraph. Every number it uses is "
                                 "checked against the data; if any doesn't match, the paragraph is dropped.")
        if preset == "Branch" and sel_branch in ("All", "", None):
            a.info("A Branch report is for one branch: pick it in the sidebar's Branch filter, "
                   "or make a pack for every branch below.")
        elif not main and not extras:
            a.info("Tick at least one section.")
        elif b.button("Generate report", type="primary", key="rpt2_generate", width="stretch"):
            _generate(df_curr, df_prev, curr_month, prev_month, preset, filters, main + extras, use_ai)

    st.markdown("""<style>
    .st-key-rpt2_card { background: #ffffff !important; border-radius: 12px !important;
                        border-bottom: 3px solid #FFC000 !important; }
    .rpt2-hero { background: #0d0d0d; border-left: 5px solid #FFC000; border-radius: 10px;
                 padding: 14px 18px; margin-bottom: 6px; }
    .rpt2-hero-title { color: #FFC000; font-size: 18px; font-weight: 800; letter-spacing: 0.3px; }
    .rpt2-hero-sub { color: #d1d5db; font-size: 12.5px; margin-top: 3px; }
    .rpt2-step { display: flex; align-items: center; gap: 8px; font-size: 13.5px; font-weight: 700;
                 color: #111827; margin: 6px 0 2px 0; }
    .rpt2-step span { background: #FFC000; color: #000; border-radius: 50%; width: 22px; height: 22px;
                      display: inline-flex; align-items: center; justify-content: center;
                      font-size: 12px; font-weight: 800; }
    .rpt2-summary { display: flex; flex-wrap: wrap; gap: 8px 26px; font-size: 12.5px; color: #ffffff;
                    background: #1a1a1a; border-radius: 8px; padding: 9px 14px; margin: 8px 0 4px 0; }
    .rpt2-summary b { color: #FFC000; font-weight: 700; margin-right: 5px; text-transform: uppercase;
                      font-size: 11px; letter-spacing: 0.6px; }
    .st-key-rpt2_card [data-testid="stExpander"] details { border-left: 4px solid #FFC000 !important; }
    .st-key-rpt2_generate button { color: #000 !important; font-weight: 800 !important;
                                   border: 2px solid #000 !important; }
    .st-key-rpt2_generate button:hover { background: #000 !important; color: #FFC000 !important; }
    </style>""", unsafe_allow_html=True)

    _show_result()
    _branch_packs(df_curr, df_prev, curr_month, prev_month, filters, extras)


def _step(n: int, title: str) -> str:
    return f'<div class="rpt2-step"><span>{n}</span>{title}</div>'


def _tick(widget_key: str, on: bool) -> None:
    st.session_state[widget_key] = on
    st.session_state.setdefault("_rpt2_ticks", {})[widget_key] = on


def _tick_preset(preset: str) -> None:
    """Tick exactly the preset's own main sections."""
    from report_agent.story import MAIN_SECTIONS, PRESET_SECTIONS
    for key in MAIN_SECTIONS:
        _tick(f"rpt2_sec_{key}", key in PRESET_SECTIONS[preset])


def _set_extras(on: bool) -> None:
    from report_agent.story import OPTIONAL_SECTIONS
    for key in OPTIONAL_SECTIONS:
        _tick(f"rpt2_extra_{key}", on)


def _checkbox_grid(options: dict, prefix: str, columns: int = 3) -> None:
    """One checkbox per option ({key: label}), in columns. The ticks are also
    kept in _rpt2_ticks, so they survive a visit to another tab."""
    saved = st.session_state.setdefault("_rpt2_ticks", {})
    cols = st.columns(columns)
    for i, (key, label) in enumerate(options.items()):
        wkey = f"{prefix}{key}"
        if wkey not in st.session_state:
            st.session_state[wkey] = saved.get(wkey, False)
        saved[wkey] = cols[i % columns].checkbox(label, key=wkey)


def _generate(df_curr, df_prev, curr_month, prev_month, preset, filters, sections, use_ai) -> None:
    from report_agent import render
    from report_agent.story import build_report
    from smart_alerts import run_all_alerts
    with st.spinner("Building the report..."):
        try:
            alerts = run_all_alerts(df_curr, as_of=curr_month)
            alerts_prev = run_all_alerts(df_prev, as_of=prev_month) if len(df_prev) else []
            model = build_report(df_curr, df_prev, curr_month, prev_month, preset, filters, alerts, alerts_prev,
                                 sections=sections)
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
    section_label("Branch packs", "18px")
    with st.expander(f"A Branch report for each of the {len(branches)} branches in view", expanded=False):
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
