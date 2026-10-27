"""AI Query tab: list / count / custom-grouping questions in plain English,
answered by the AI Query engine (graph.run_query: planner -> validator ->
compiler -> executor). The pipeline's steps show live while it runs and stay
on the answer ("How this answer was built"), with the validation badge and
"how I read your question" on the answer card (ui/query_answer.py).

Questions about WHY a number moved belong to the Investigator tab; it hands
list questions over here (HANDOFF_KEY), already typed in and run."""
import os

import pandas as pd
import streamlit as st

from ui.query_answer import render_clarification, render_query_result, render_steps, run_ai_query

INPUT_KEY = "ai_query_input"
HANDOFF_KEY = "_aiq_handoff"        # a question sent from the Investigator: fill the box and run it
RESULT_KEY = "ai_result"

_EXAMPLES = [
    ("Show customers who haven't paid for last 3 months", "Show customers who haven't paid for last 3 months"),
    ("Branches with the biggest NPA drop vs last month",
     "Give me all branches with previous NPA count and current NPA count, sorted descending with the biggest "
     "reduction on top"),
    ("Accounts above 2 EMI from Nov 2025 onward advances",
     "Show all accounts with arrears greater than 2 EMI from November 2025 onward advances"),
    ("Accounts that need immediate action", "Show those accounts that need immediate action"),
]


def _fill(text: str) -> None:
    """An example's on_click: fills the question box. Callbacks run before the
    page redraws, so the box's value can be set here."""
    st.session_state[INPUT_KEY] = text


def _panel() -> None:
    """The dark title panel with example questions; clicking one fills the box.
    Native buttons (code review, 2026-10-27): this used Streamlit's HTML
    component, now deprecated, with JavaScript reaching into the parent page."""
    st.markdown(
        """<style>
        .st-key-aiq_panel { background:#0d1117; border:1px solid #21262d; border-radius:12px;
                            padding:16px 20px 10px 20px; gap:6px; }
        .aiq-title { font-size:20px; font-weight:800; color:#FFC000; letter-spacing:-0.3px; }
        .aiq-sub { font-size:13px; color:#8b949e; line-height:1.7; margin-bottom:4px; }
        .st-key-aiq_panel .stButton button { background:#161b22; color:#8b949e; border:1px solid #2d333b;
                            border-radius:6px; padding:3px 12px; min-height:0; font-size:11px; font-style:italic; }
        .st-key-aiq_panel .stButton button p { font-size:11px; }
        .st-key-aiq_panel .stButton button:hover { background:#2a2a2a; color:#d0d0d0; border-color:#555; }
        </style>""",
        unsafe_allow_html=True,
    )
    with st.container(key="aiq_panel"):
        st.markdown('<div class="aiq-title">🤖 AI Query Assistant</div>'
                    '<div class="aiq-sub">Ask for any list, count or comparison in plain English. You\'ll see each '
                    'step as it runs, and every table is exact source data. Click an example to try:</div>',
                    unsafe_allow_html=True)
        with st.container(horizontal=True, gap="small"):
            for i, (short, full) in enumerate(_EXAMPLES):
                st.button(short, key=f"aiq_example_{i}", on_click=_fill, args=(full,))


def render_ai_query_tab(df_curr: pd.DataFrame, query_ctx: dict) -> None:
    """query_ctx: what run_ai_query needs (snapshot_dates, df_prev,
    precomputed_views, alerts_curr, alerts_prev, rr_meta, data_version,
    filter_key)."""
    _panel()
    handoff = st.session_state.pop(HANDOFF_KEY, None)
    if handoff:
        st.session_state[INPUT_KEY] = handoff
    question = st.text_area("Query", key=INPUT_KEY, placeholder="Type your question here...", height=90,
                            label_visibility="collapsed")
    c1, c2, c3 = st.columns([1, 1.6, 3], vertical_alignment="center")
    run = c1.button("🔍  Run Query", type="primary", width="stretch", key="aiq_run")
    summary = c2.checkbox("Generate AI summary", value=False, key="ai_gen_summary",
                          help="Off by default: one more Gemini call writes bullet-point observations under the "
                               "table. The table, KPIs and highlights don't need it.")
    c3.markdown("<div style='font-size:12px;color:#9ca3af;text-align:right;'>Powered by Gemini · planner, "
                "validator and compiler pipeline</div>", unsafe_allow_html=True)

    if run or handoff:
        if not (question or "").strip():
            st.warning("Please enter a question.")
        elif not os.environ.get("GOOGLE_API_KEY"):
            st.error("GOOGLE_API_KEY not found in .env file.")
        else:
            st.session_state[RESULT_KEY] = run_ai_query(question.strip(), df_curr, query_ctx,
                                                        skip_insights=not summary)

    result = st.session_state.get(RESULT_KEY)
    if not result:
        return
    render_steps(result, "aiq")
    if result.get("needs_clarification"):
        picked = render_clarification(result, "aiq")
        if picked:
            out_of_scope = result.get("query_title") == "Out of Scope"
            st.session_state[RESULT_KEY] = run_ai_query(picked, df_curr, query_ctx, allow_clarification=not out_of_scope,
                                                        skip_insights=not summary)
            st.rerun()
        return
    render_query_result(result, df_curr, "aiq")
