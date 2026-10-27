"""Every structured Gemini call asks for JSON mode and parses the reply in one
place (gemini_client.json_config / parse_json_object). Before, the AI Query
planner and the Investigator router asked for free text and each stripped
markdown fences with its own regex (code review, 2026-10-27)."""
import pytest

import gemini_client as gc
from config import PLANNER_TEMPERATURE


class TestHelpers:
    def test_json_config_asks_for_json(self):
        cfg = gc.json_config(0.1, system_instruction="SYS")
        assert cfg == {"temperature": 0.1, "response_mime_type": "application/json", "system_instruction": "SYS"}
        assert "response_schema" in gc.json_config(0.1, response_schema={"type": "object"})

    @pytest.mark.parametrize("text", ['{"a": 1}', '```json\n{"a": 1}\n```', '```\n{"a": 1}```', '  {"a": 1}  '])
    def test_parse_json_object_reads_bare_or_fenced_json(self, text):
        assert gc.parse_json_object(text) == {"a": 1}

    @pytest.mark.parametrize("text", ["I'm sorry, I can't help with that.", "[1, 2]", "", None, '"just a string"'])
    def test_anything_but_one_json_object_is_an_error(self, text):
        with pytest.raises(gc.JSONReplyError):
            gc.parse_json_object(text)

    def test_generate_json_text_uses_json_mode(self, monkeypatch):
        seen = {}
        monkeypatch.setenv("GOOGLE_API_KEY", "dummy")
        monkeypatch.setattr(gc, "make_client", lambda key: object())
        monkeypatch.setattr(gc, "add_token_usage", lambda r: None)

        class R:
            text = '{"ok": true}'

        def call(client, model, contents, config):
            seen.update(config=config, contents=contents)
            return R()

        monkeypatch.setattr(gc, "call_gemini_with_retry", call)
        assert gc.generate_json_text("PROMPT", temperature=0.1) == '{"ok": true}'
        assert seen["config"]["response_mime_type"] == "application/json" and seen["contents"] == "PROMPT"


class _Reply:
    def __init__(self, text):
        self.text = text


class TestCallSites:
    def test_the_ai_query_planner_asks_for_json(self, monkeypatch):
        import agents.logical_planner as lp
        monkeypatch.setenv("GOOGLE_API_KEY", "dummy")
        sent = {}
        monkeypatch.setattr(lp, "_call_gemini_hedged",
                            lambda c, m, contents, config: sent.update(config=config) or _Reply('{"intent": "loan_table"}'))
        monkeypatch.setattr(lp, "_add_token_usage", lambda *a, **k: None)
        lp.plan_logical("show me the top branches")
        assert sent["config"]["response_mime_type"] == "application/json"
        assert sent["config"]["temperature"] == PLANNER_TEMPERATURE and sent["config"]["system_instruction"]

    def test_a_planner_reply_that_is_not_an_object_is_out_of_scope(self, monkeypatch):
        import agents.logical_planner as lp
        monkeypatch.setenv("GOOGLE_API_KEY", "dummy")
        monkeypatch.setattr(lp, "_call_gemini_hedged", lambda *a, **k: _Reply("[1, 2]"))
        monkeypatch.setattr(lp, "_add_token_usage", lambda *a, **k: None)
        assert lp.plan_logical("anything")["query_title"] == "Out of Scope"

    def test_the_investigator_router_asks_for_json(self, monkeypatch):
        import investigator.llm as llm
        from investigator.state import EntityMemory
        monkeypatch.setenv("GOOGLE_API_KEY", "dummy")
        sent = {}
        monkeypatch.setattr(llm, "make_client", lambda key: object())
        monkeypatch.setattr(llm, "_add_token_usage", lambda *a, **k: None)
        monkeypatch.setattr(llm, "_call_gemini_hedged", lambda c, m, contents, config: sent.update(config=config)
                            or _Reply('{"step_type": "dimension_breakdown", "params": {"dimension": "region"}}'))
        out = llm.parse_turn("how are my regions doing", EntityMemory())
        assert sent["config"]["response_mime_type"] == "application/json" and out["step_type"] == "dimension_breakdown"

    def test_the_review_analyst_asks_for_json(self, monkeypatch):
        from investigation.review import analyst
        sent = {}
        monkeypatch.setattr(analyst, "generate_json_text", lambda prompt, **kw: sent.update(kw=kw) or "{}")
        analyst.gemini_generate("PROMPT")
        assert sent["kw"]["temperature"] == PLANNER_TEMPERATURE
