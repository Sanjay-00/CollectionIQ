"""Report tab: the monthly report (report_agent/story.py) for one reader level,
as one file per unit in view (report_agent/packs.py) in the one format picked:
Leadership (one file), Zone (one per zone), Regional (one per region), Branch
(one per branch). Several files come as one ZIP."""
import os

import pandas as pd
import streamlit as st

from ui.components import _esc, _send_report_email

_PRESET_HELP = {
    "Leadership": "For leaders: everything in view, high level. How the zones and regions compare, where loans "
                  "slip, and where to focus. Filter the sidebar to one zone or region for its own leadership view.",
    "Zone": "For zonal managers, one file per zone: its regions compared, its best and worst branches, "
            "where loans slip, and where to focus.",
    "Regional": "For regional managers, one file per region, in detail: branches and executives ranked, "
                "where loans slip and why, new business, and what to do first.",
    "Branch": "For branch managers, one file per branch: executives ranked, where loans slip, what to do first, "
              "and who to call.",
}
_UNIT_WORD = {"Zone": "zone", "Regional": "region", "Branch": "branch"}
_SECONDS_PER_FILE = {"PDF": 1.8, "HTML": 1.3, "Excel": 2.3}    # measured on the demo book, for the "about N seconds" note


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
    from report_agent.packs import FORMATS, units_in
    from report_agent.story import DEFAULT_PRESET, MAIN_SECTIONS, OPTIONAL_SECTIONS, PRESETS

    # Streamlit forgets a widget's value when its tab isn't shown, so the picks
    # are kept in plain session keys (_rpt2_last_preset, _rpt2_last_fmt,
    # _rpt2_ticks) and restored.
    last = st.session_state.get("_rpt2_last_preset")
    if "rpt2_preset" not in st.session_state and last in PRESETS:
        st.session_state["rpt2_preset"] = last
    last_fmt = st.session_state.get("_rpt2_last_fmt")
    if "rpt2_fmt" not in st.session_state and last_fmt in FORMATS:
        st.session_state["rpt2_fmt"] = last_fmt
    filters = {"Zone": sel_zone, "Region": sel_region, "Branch": sel_branch, "Loan Status": sel_status}

    with st.container(border=True, key="rpt2_card"):
        st.markdown('<div class="rpt2-hero"><div class="rpt2-hero-title">Monthly Report</div>'
                    '<div class="rpt2-hero-sub">A short visual story for each reader: what went well, what went '
                    "wrong, and where to focus, built from the same numbers as the Dashboard.</div></div>",
                    unsafe_allow_html=True)
        left, right = st.columns(2, gap="large")
        with left:
            st.markdown(_step(1, "Who is it for?"), unsafe_allow_html=True)
            preset = st.segmented_control("Who is it for?", PRESETS, key="rpt2_preset",
                                          default=None if "rpt2_preset" in st.session_state else DEFAULT_PRESET,
                                          label_visibility="collapsed") or last or DEFAULT_PRESET
            st.markdown(f'<div class="rpt2-help">{_PRESET_HELP[preset]}</div>', unsafe_allow_html=True)
        # A new level ticks its own sections (the detail sections stay as picked).
        if st.session_state.get("_rpt2_last_preset") != preset:
            st.session_state["_rpt2_last_preset"] = preset
            _tick_preset(preset)

        units = units_in(df_curr, preset)
        word = _UNIT_WORD.get(preset)
        with right:
            st.markdown(_step(2, "Which ones?"), unsafe_allow_html=True)
            if units:
                picked = _pick_units(preset, units, word)
            else:
                picked = None
                st.markdown('<div class="rpt2-help" style="margin-top:6px;">One file for everything in view. '
                            "To narrow it, filter the sidebar (zone, region or branch).</div>", unsafe_allow_html=True)
        files = len(picked) if units else 1
        files_text = (f"{files} files, one per {word} (ZIP)" if files > 1
                      else f"1 file ({_esc(picked[0])})" if units and picked else "none picked" if units else "1 file")

        left, right = st.columns(2, gap="large")
        with left:
            st.markdown(_step(3, "Format"), unsafe_allow_html=True)
            fmt = st.segmented_control("Format", list(FORMATS), key="rpt2_fmt",
                                       default=None if "rpt2_fmt" in st.session_state else "PDF",
                                       label_visibility="collapsed") or last_fmt or "PDF"
            st.session_state["_rpt2_last_fmt"] = fmt
            st.markdown('<div class="rpt2-help">' + {
                "PDF": "To print or attach.", "HTML": "To paste into an email; looks the same in Gmail and Outlook.",
                "Excel": "Every table in full, for anyone who wants to dig in."}[fmt] + "</div>", unsafe_allow_html=True)
        with right:
            st.markdown(_step(4, "What goes in"), unsafe_allow_html=True)
            # A fixed title and key: a title that changed with each tick (a live
            # count) made Streamlit treat it as a new expander and close it.
            with st.expander("Choose the sections", key="rpt2_sec_box"):
                _checkbox_grid({k: label for k, label in MAIN_SECTIONS.items()}, "rpt2_sec_", columns=2)
                st.button(f"Back to the {preset} sections", key="rpt2_defaults", on_click=_tick_preset, args=(preset,))
            with st.expander("Add detail sections", key="rpt2_extra_box"):
                st.caption("Each adds its top rows to the report; the full lists go in the Excel annex.")
                b1, b2 = st.columns(2)
                b1.button("Select all", key="rpt2_extra_all", on_click=_set_extras, args=(True,), width="stretch")
                b2.button("Clear all", key="rpt2_extra_none", on_click=_set_extras, args=(False,), width="stretch")
                _checkbox_grid({k: label for k, (label, _) in OPTIONAL_SECTIONS.items()}, "rpt2_extra_", columns=2)

        main = [k for k in MAIN_SECTIONS if st.session_state.get(f"rpt2_sec_{k}")]
        extras = [k for k in OPTIONAL_SECTIONS if st.session_state.get(f"rpt2_extra_{k}")]
        scope = " · ".join(f"{k}: {_esc(v)}" for k, v in filters.items() if v not in (None, "", "All")) or "Whole portfolio"
        st.markdown(
            f'<div class="rpt2-summary"><span><b>Scope</b> {scope}</span><span><b>Files</b> {files_text}</span>'
            f'<span><b>Sections</b> {len(main)} of {len(MAIN_SECTIONS)}'
            + (f" + {len(extras)} detail" if extras else "") + f"</span><span><b>Format</b> {fmt}</span></div>",
            unsafe_allow_html=True)

        st.markdown(_step(5, "Generate"), unsafe_allow_html=True)
        a, b = st.columns([3, 1], vertical_alignment="center")
        use_ai = a.checkbox("Add an AI-written opening paragraph", value=False, key="rpt2_ai",
                            help="Gemini rewrites the At a glance story into a paragraph (one call per file). Every "
                                 "number it uses is checked against the data; if any doesn't match, it's dropped.")
        if files > 1:
            a.caption(f"About {round(files * _SECONDS_PER_FILE[fmt])} seconds for {files} files."
                      + (" Pick fewer in step 2 to make it faster." if files > 15 else ""))
        if units and not picked:
            a.info(f"Pick at least one {word} in step 2.")
        elif not main and not extras:
            a.info("Tick at least one section.")
        elif b.button("Generate", type="primary", key="rpt2_generate", width="stretch"):
            _generate(df_curr, df_prev, curr_month, prev_month, preset, filters, main + extras, fmt, use_ai,
                      picked if units else None)

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
    .rpt2-help { font-size: 12px; color: #6b7280; margin-top: -6px; line-height: 1.45; }
    .st-key-rpt2_card [data-testid="stButtonGroup"] { margin-bottom: 0 !important; }
    .rpt2-summary { display: flex; flex-wrap: wrap; gap: 8px 26px; font-size: 12.5px; color: #ffffff;
                    background: #1a1a1a; border-radius: 8px; padding: 9px 14px; margin: 10px 0 4px 0; }
    .rpt2-summary b { color: #FFC000; font-weight: 700; margin-right: 5px; text-transform: uppercase;
                      font-size: 11px; letter-spacing: 0.6px; }
    .st-key-rpt2_card [data-testid="stExpander"] details { border-left: 4px solid #FFC000 !important; }
    .st-key-rpt2_generate button, .st-key-rpt2_download button { color: #000 !important; font-weight: 800 !important;
                                   border: 2px solid #000 !important; }
    .st-key-rpt2_generate button:hover, .st-key-rpt2_download button:hover {
        background: #000 !important; color: #FFC000 !important; }
    </style>""", unsafe_allow_html=True)

    _show_result()


def _step(n: int, title: str) -> str:
    return f'<div class="rpt2-step"><span>{n}</span>{title}</div>'


def _tick(widget_key: str, on: bool) -> None:
    st.session_state[widget_key] = on
    st.session_state.setdefault("_rpt2_ticks", {})[widget_key] = on


def _tick_preset(preset: str) -> None:
    """Tick exactly the level's own main sections."""
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


def _pick_units(preset: str, units: list[str], word: str) -> list[str]:
    """The units to make files for (all by default), kept per level across tab visits."""
    saved = st.session_state.setdefault("_rpt2_units", {})
    key = f"rpt2_units_{preset}"
    if key not in st.session_state:
        st.session_state[key] = [u for u in saved.get(preset, units) if u in units] or list(units)
    st.session_state[key] = [u for u in st.session_state[key] if u in units]     # filters may have changed
    picked = st.multiselect(f"Which {word}s?", units, key=key, label_visibility="collapsed",
                            placeholder=f"Pick one or more {word}s")
    saved[preset] = picked
    note = (f"All {len(units)} {word}s in view." if len(picked) == len(units)
            else f"{len(picked)} of {len(units)} {word}s.")
    st.markdown(f'<div class="rpt2-help" style="margin-top:2px;">{note} One file each.</div>', unsafe_allow_html=True)
    return picked


def _generate(df_curr, df_prev, curr_month, prev_month, preset, filters, sections, fmt, use_ai, only=None) -> None:
    from report_agent.packs import MIME, build_files, zip_files
    bar = st.progress(0.0, text="Building...")
    out = build_files(df_curr, df_prev, curr_month, prev_month, preset, filters, sections=sections, fmt=fmt,
                      use_ai=use_ai, only=only, progress=lambda i, n, u: bar.progress(i / n, text=f"{i} of {n}: {u}"))
    bar.empty()
    files = out["files"]
    if len(files) == 1:
        name, data = files[0]
        mime = MIME[name.rsplit(".", 1)[1]]
    else:
        name, data, mime = f"{preset.lower()}_reports_{curr_month}.zip", zip_files(files), MIME["zip"]
    st.session_state["rpt2"] = {"name": name, "data": data, "mime": mime, "n": len(files), "preset": preset,
                                "fmt": fmt, "month": curr_month, "preview": out["preview"], "notes": out["notes"]}


def _show_result() -> None:
    out = st.session_state.get("rpt2")
    if not out or "data" not in out:
        return
    for note in out["notes"][:3]:
        (st.success if "AI summary added" in note else st.warning)(note)
    label = (f"⬇ Download {out['n']} {out['preset']} reports ({out['fmt']}, ZIP)" if out["n"] > 1
             else f"⬇ Download the {out['preset']} report ({out['fmt']})")
    st.download_button(label, out["data"], out["name"], out["mime"], key="rpt2_download", type="primary")

    if out["fmt"] == "HTML" and out["n"] == 1:
        if os.environ.get("SMTP_HOST", ""):
            e1, e2 = st.columns([3, 1])
            to = e1.text_input("Send the HTML report to", placeholder="manager@company.com, head@company.com",
                               key="rpt2_email_to", help="Separate several addresses with a comma.")
            e2.markdown('<div style="height:28px;"></div>', unsafe_allow_html=True)
            if e2.button("📧 Send", key="rpt2_send", width="stretch"):
                if not to.strip():
                    st.warning("Enter an email address.")
                else:
                    ok, err = _send_report_email(out["data"].decode("utf-8"), to.strip(), out["month"])
                    (st.success(f"Sent to {to}") if ok else st.error(f"Failed: {err}"))
        else:
            st.caption("Email sending is off: add SMTP_HOST / SMTP_USER / SMTP_PASS to .env to turn it on.")

    with st.expander("Preview" + (" (the first file)" if out["n"] > 1 else ""), expanded=True, key="rpt2_preview"):
        st.html(out["preview"])
