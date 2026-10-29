"""The AI switch (2026-10-28): COLLECTIONIQ_AI=off in .env means no data
leaves this computer: no Gemini call and no LangSmith trace, whatever else is
configured. Everything computed in code (dashboard, reports, the review
briefing) still works; each AI feature says plainly that AI is off. For a
pilot waiting on data approval, and as the demo's fallback if the network fails."""
import os

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

import gemini_client as gc
from investigator.state import EntityMemory


@pytest.fixture
def ai_off(monkeypatch):
    """AI switched off, with a key still configured: the switch must win."""
    monkeypatch.setenv("COLLECTIONIQ_AI", "off")
    monkeypatch.setenv("GOOGLE_API_KEY", "a-real-looking-key")

    class Forbidden:
        def __init__(self, *a, **k):
            raise AssertionError("a Gemini client was created with AI switched off")
    monkeypatch.setattr(gc.genai, "Client", Forbidden)


class TestTheSwitch:
    @pytest.mark.parametrize("value, on", [("off", False), ("OFF", False), ("0", False), ("false", False),
                                           ("no", False), ("on", True), ("", True)])
    def test_reading_it(self, monkeypatch, value, on):
        monkeypatch.setenv("COLLECTIONIQ_AI", value)
        assert gc.ai_enabled() is on

    def test_on_unless_switched_off(self, monkeypatch):
        monkeypatch.delenv("COLLECTIONIQ_AI", raising=False)
        assert gc.ai_enabled() is True

    def test_no_key_is_handed_out(self, ai_off):
        assert gc.ai_key() == "" and gc.ai_unavailable_reason() == gc.AI_OFF_MESSAGE

    def test_no_client_can_be_made(self, ai_off):
        with pytest.raises(gc.AIDisabledError):
            gc.make_client("any key")

    def test_tracing_is_switched_off_too(self, ai_off, monkeypatch):
        monkeypatch.setenv("LANGSMITH_TRACING", "true")
        monkeypatch.setenv("LANGCHAIN_TRACING_V2", "true")
        gc.apply_ai_switch()
        assert os.environ["LANGSMITH_TRACING"] == "false" and os.environ["LANGCHAIN_TRACING_V2"] == "false"

    def test_tracing_left_alone_when_on(self, monkeypatch):
        monkeypatch.setenv("COLLECTIONIQ_AI", "on")
        monkeypatch.setenv("LANGSMITH_TRACING", "true")
        gc.apply_ai_switch()
        assert os.environ["LANGSMITH_TRACING"] == "true"

    def test_without_a_key_the_reason_says_so(self, monkeypatch):
        monkeypatch.setenv("COLLECTIONIQ_AI", "on")
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        assert "key" in gc.ai_unavailable_reason().lower()


class TestEveryAIFeatureStandsDown:
    """With a key present and AI off, none of them may reach Gemini."""

    def test_ai_query_planner(self, ai_off):
        import graph
        r = graph.run_query("NPA% by region", pd.DataFrame({"Loan No": ["L1"]}), skip_insights=True)
        assert gc.AI_OFF_MESSAGE in r["error"]

    def test_ai_query_summary(self, ai_off):
        from agents.insight_generator import generate_insights
        out = generate_insights("q", pd.DataFrame({"a": [1]}), {}, {})
        assert "switched off" in out

    def test_investigator_router(self, ai_off):
        from investigator import llm
        out = llm.parse_turn("why is NPA high", EntityMemory())
        assert out["needs_clarification"] and gc.AI_OFF_MESSAGE in out["clarification_question"]

    def test_investigator_narration(self, ai_off):
        from investigator import llm
        df = pd.DataFrame({"Metric": ["NPA%"], "Current": [5.0]})
        assert llm.narrate_step("entity_summary", df) == ""
        assert list(llm.narrate_step_stream("entity_summary", df)) == []

    def test_review_analyst(self, ai_off):
        from investigation.review.analyst import write_analysis
        brief = type("B", (), {"status": "ok", "findings": [object()]})()
        out = write_analysis(brief)
        assert out.status == "skipped"

    def test_report_summary(self, ai_off, monkeypatch):
        from report_agent import ai_summary
        monkeypatch.setattr(ai_summary, "facts_from", lambda model: "NPA% 5.0 in PUNE")
        text, status = ai_summary.polish({})
        assert text is None and "switched off" in status

    def test_json_helper(self, ai_off):
        with pytest.raises(gc.AIDisabledError):
            gc.generate_json_text("x", temperature=0.1)


class TestWhatThePageSays:
    def test_ai_query_explains_instead_of_erroring(self, ai_off):
        def page():
            import pandas as pd
            import streamlit as st
            from ui.tabs.ai_query import render_ai_query_tab
            st.session_state["_aiq_handoff"] = "NPA% by region"
            render_ai_query_tab(pd.DataFrame({"Loan No": ["L1"]}), {})
        at = AppTest.from_function(page).run()
        assert not at.exception and not at.error
        assert any("switched off" in i.value for i in at.info)

    def test_investigator_explains_instead_of_erroring(self, ai_off):
        def page():
            import pandas as pd
            from investigator.state import EntityMemory
            from ui.tabs.investigator import _process_new_question
            _process_new_question("why is NPA high", pd.DataFrame({"Loan No": ["L1"]}), pd.DataFrame(), EntityMemory())
        at = AppTest.from_function(page).run()
        assert not at.exception and not at.error
        assert any("switched off" in i.value for i in at.info)
