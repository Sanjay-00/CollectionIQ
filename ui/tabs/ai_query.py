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


def _panel() -> None:
    """The dark title panel with example questions; clicking one fills the box."""
    chips = "".join(
        f'<button class="chip" onclick="fill({_js(full)})">{_html(short)}</button>' for short, full in _EXAMPLES)
    st.components.v1.html(f"""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap" rel="stylesheet">
<style>
*, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ background: transparent; font-family: 'Inter', sans-serif; padding: 2px 0 0 0; }}
.panel {{ background: #0d1117; border: 1px solid #21262d; border-radius: 12px; padding: 18px 22px; }}
.hdr {{ display: flex; align-items: center; gap: 10px; margin-bottom: 6px; }}
.icon {{ font-size: 22px; line-height: 1; }}
.title {{ font-size: 20px; font-weight: 800; color: #FFC000; letter-spacing: -0.3px; }}
.sub {{ font-size: 13px; color: #8b949e; line-height: 1.7; margin-bottom: 12px; }}
.chip {{ display: inline-block; background: #161b22; color: #8b949e; border: 1px solid #2d333b; border-radius: 6px;
        padding: 5px 12px; font-size: 11px; font-style: italic; cursor: pointer; margin: 0 6px 6px 0;
        font-family: 'Inter', sans-serif; transition: background 0.15s, color 0.15s, border-color 0.15s;
        line-height: 1.4; outline: none; }}
.chip:hover {{ background: #2a2a2a; color: #d0d0d0; border-color: #555; }}
</style>
<div class="panel">
  <div class="hdr"><span class="icon">🤖</span><span class="title">AI Query Assistant</span></div>
  <div class="sub">Ask for any list, count or comparison in plain English. You'll see each step as it runs,
  and every table is exact source data. Click an example to try:</div>
  {chips}
</div>
<script>
function fill(text) {{
    var doc = window.parent.document;
    var ta  = doc.querySelector('[data-testid="stTextArea"] textarea');
    if (!ta) return;
    var set = Object.getOwnPropertyDescriptor(window.parent.HTMLTextAreaElement.prototype, 'value').set;
    set.call(ta, text);
    ta.dispatchEvent(new Event('input', {{ bubbles: true }}));
    ta.focus();
}}
</script>
""", height=168, scrolling=False)


def _js(text: str) -> str:
    """A string as a JavaScript literal inside an HTML attribute."""
    import json
    return _html(json.dumps(text))


def _html(text: str) -> str:
    import html
    return html.escape(text, quote=True)


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
