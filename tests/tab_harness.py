"""The whole Investigator tab under AppTest (tests/test_investigator_tab.py).

Data: the demo portfolio (generate_demo_data.py) through the app's own
cleaning (utils.load_and_validate, the file read skipped) and its month join
(last month's bucket and amounts carried on each loan, as app.py does). The
review brief is computed for real (run_review); the background runner, the
decision log's file and the AI router are swapped for test doubles: the
router answers each question from a fixed script, so no Gemini call is made.
AppTest runs the page in this process, so the page reads DATA from here."""
from __future__ import annotations

import time

import pandas as pd

DATA: dict = {}
CURR_MONTH, PREV_MONTH = "2026-08", "2026-07"


def build() -> dict:
    """curr, prev, brief, and the names the scripted router uses."""
    if DATA:
        return DATA
    import generate_demo_data as demo
    import utils
    from investigation.review.runner import run_review

    loans = demo.build_portfolio()
    sim = demo.simulate(loans)

    def clean(raw: pd.DataFrame) -> pd.DataFrame:
        read = utils._read_excel
        utils._read_excel = lambda file, sheet_name=0: (raw.copy(), "calamine")
        try:
            df, errors = utils.load_and_validate(None)
        finally:
            utils._read_excel = read
        assert df is not None and not errors, errors
        return df

    curr = clean(demo.add_real_world_mess(demo.snapshot(loans, sim, demo.CURR)))
    prev = clean(demo.add_real_world_mess(demo.snapshot(loans, sim, demo.PREV)))
    carry = {"curr_bucket": "prev_bucket", **{s: d for s, d in utils.PREV_CARRYOVER_COLS.items() if s in prev.columns}}
    curr = curr.merge(prev[["Loan No", *carry]].rename(columns=carry), on="Loan No", how="left")

    branch = str(curr["Unit"].value_counts().index[0])
    row = curr[curr["Unit"] == branch].iloc[0]
    DATA.update(curr=curr, prev=prev, brief=run_review(curr, prev, CURR_MONTH, PREV_MONTH),
                branch=branch, region=str(row["RegionName"]),
                executive=(str(row["MNT NAME"]), branch))
    return DATA


def route(question: str) -> dict:
    """The scripted router: what parse_turn would return for each test question."""
    d = DATA
    base = {"resolved_entity": None, "needs_clarification": False, "clarification_question": ""}
    q = question.split(": ", 1)[-1] if question.startswith("About the") else question
    if q == "q_branch":
        return {**base, "step_type": "entity_summary", "params": {"entity_type": "branch", "entity_value": d["branch"]}}
    if q == "q_exec":         # typed in lower case: stored under the file's own spelling
        name, unit = d["executive"]
        return {**base, "step_type": "entity_summary",
                "params": {"entity_type": "executive", "entity_value": [name.lower(), unit]}}
    if q == "q_clarify":
        return {**base, "step_type": None, "params": {}, "needs_clarification": True,
                "clarification_question": "Which branch do you mean?"}
    if q == "q_handoff":
        return {**base, "step_type": "portfolio_query", "params": {"question": "list the NPA loans"}}
    if q == "q_bad":          # entity_value missing: the step refuses
        return {**base, "step_type": "entity_summary", "params": {"entity_type": "branch"}}
    if q == "q_parts":
        part = lambda step, params, label: {"step_type": step, "params": params, "label": label}  # noqa: E731
        parts = [part("dimension_breakdown", {"dimension": "branch", "scope_col": "region", "scope_value": d["region"]},
                      "Branches"),
                 part("underperformer_quartile", {"metric": "NPA %", "scope_col": "region", "scope_value": d["region"]},
                      "Executives")]
        return {**base, "step_type": parts[0]["step_type"], "params": parts[0]["params"], "parts": parts}
    return {**base, "step_type": None, "params": {}, "needs_clarification": True, "clarification_question": "?"}


def install(monkeypatch, log_path: str) -> None:
    """Swap in the test doubles (undone by monkeypatch after the test)."""
    import streamlit as st

    import ui.review_panel as rp
    import ui.tabs.investigator as inv
    from investigation.review.decisions import DecisionLog
    from investigation.review.runner import Job

    job = Job("session|1|f|2026-08|2026-07", status="done", brief=DATA["brief"], finished=time.time())
    monkeypatch.setattr(rp, "start_review", lambda *a, **k: job)
    monkeypatch.setattr(inv, "start_review", lambda *a, **k: job)
    monkeypatch.setattr(rp, "decision_log", lambda: st.session_state.setdefault("LOG", DecisionLog(log_path)))

    def scripted(question, memory, allow_multi=False):
        st.session_state.setdefault("ROUTED", []).append((question, allow_multi))
        return route(question)
    monkeypatch.setattr(inv, "parse_turn", scripted)


def app() -> None:
    """The page AppTest runs."""
    import tab_harness as h
    import ui.tabs.investigator as inv
    inv.render_investigator_tab(h.DATA["curr"], h.DATA["prev"], data_version=1, filter_key="f",
                                curr_month=h.CURR_MONTH, prev_month=h.PREV_MONTH)
