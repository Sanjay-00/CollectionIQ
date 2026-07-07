"""Regression coverage for report_agent/nodes/risk_narrator.py -- previously had
zero test coverage despite being one of only two report-pipeline nodes doing
real I/O (Gemini calls with retry). Covers the "one bad call must not discard
the other call's good output" fix, and that failures are logged, not swallowed.
"""
import logging

import pytest

from report_agent.nodes import risk_narrator as rn


def _base_state(**overrides) -> dict:
    state = {
        "df_curr": None, "df_prev": None, "curr_month": "2026-06", "prev_month": None,
        "enabled_sections": [], "filters_applied": {}, "skip_ai": False,
        "section_data": {"portfolio_health": {"kpis": {}}},
        "executive_narrative": "", "action_plan": "", "ai_skipped": False,
        "html_report": "", "email_to": "", "email_sent": False, "email_error": "",
        "run_id": "test-run", "error": "",
    }
    state.update(overrides)
    return state


class TestRiskNarratorNode:
    def test_no_api_key_returns_empty_and_skipped(self, monkeypatch):
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        out = rn.risk_narrator_node(_base_state())
        assert out["executive_narrative"] == ""
        assert out["action_plan"] == ""
        assert out["ai_skipped"] is True

    def test_no_section_data_returns_empty_and_skipped(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")
        out = rn.risk_narrator_node(_base_state(section_data={}))
        assert out["ai_skipped"] is True

    def test_both_calls_succeed(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")

        class _Resp:
            def __init__(self, text):
                self.text = text
                self.usage_metadata = None

        responses = iter([_Resp("- narrative bullet"), _Resp("1. action item")])
        monkeypatch.setattr(rn, "_call_gemini_with_retry", lambda *a, **k: next(responses))
        monkeypatch.setattr(rn.genai, "Client", lambda api_key: object())

        out = rn.risk_narrator_node(_base_state())
        assert out["executive_narrative"] == "- narrative bullet"
        assert out["action_plan"] == "1. action item"
        assert out["ai_skipped"] is False

    def test_action_plan_failure_does_not_discard_successful_narrative(self, monkeypatch, caplog):
        # Regression: both Gemini calls used to share ONE try/except, so a
        # failure on the second call threw away the first call's good output.
        monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")

        class _Resp:
            def __init__(self, text):
                self.text = text
                self.usage_metadata = None

        calls = {"n": 0}

        def _fake_call(client, model, contents, config, max_retries=2):
            calls["n"] += 1
            if calls["n"] == 1:
                return _Resp("- narrative bullet")
            raise RuntimeError("Gemini quota exceeded")

        monkeypatch.setattr(rn, "_call_gemini_with_retry", _fake_call)
        monkeypatch.setattr(rn.genai, "Client", lambda api_key: object())

        with caplog.at_level(logging.WARNING):
            out = rn.risk_narrator_node(_base_state())

        assert out["executive_narrative"] == "- narrative bullet"
        assert out["action_plan"] == ""
        assert out["ai_skipped"] is False  # partial success is still "has AI content"
        assert any("action plan generation failed" in r.message for r in caplog.records)

    def test_narrative_failure_does_not_discard_successful_action_plan(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")

        class _Resp:
            def __init__(self, text):
                self.text = text
                self.usage_metadata = None

        calls = {"n": 0}

        def _fake_call(client, model, contents, config, max_retries=2):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("transient failure")
            return _Resp("1. action item")

        monkeypatch.setattr(rn, "_call_gemini_with_retry", _fake_call)
        monkeypatch.setattr(rn.genai, "Client", lambda api_key: object())

        out = rn.risk_narrator_node(_base_state())
        assert out["executive_narrative"] == ""
        assert out["action_plan"] == "1. action item"
        assert out["ai_skipped"] is False

    def test_both_calls_fail_returns_empty_and_skipped(self, monkeypatch, caplog):
        monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")

        def _fake_call(client, model, contents, config, max_retries=2):
            raise RuntimeError("Gemini down")

        monkeypatch.setattr(rn, "_call_gemini_with_retry", _fake_call)
        monkeypatch.setattr(rn.genai, "Client", lambda api_key: object())

        with caplog.at_level(logging.WARNING):
            out = rn.risk_narrator_node(_base_state())

        assert out["executive_narrative"] == ""
        assert out["action_plan"] == ""
        assert out["ai_skipped"] is True
        assert len(caplog.records) >= 2  # both failures logged, neither silently swallowed


class TestCallGeminiWithRetry:
    def test_retries_then_succeeds(self, monkeypatch):
        monkeypatch.setattr(rn.time, "sleep", lambda *_: None)
        attempts = {"n": 0}

        class _Client:
            class models:
                @staticmethod
                def generate_content(model, contents, config):
                    attempts["n"] += 1
                    if attempts["n"] < 2:
                        raise RuntimeError("transient")
                    return "ok"

        result = rn._call_gemini_with_retry(_Client(), "model", "prompt", {})
        assert result == "ok"
        assert attempts["n"] == 2

    def test_exhausts_retries_and_raises(self, monkeypatch):
        monkeypatch.setattr(rn.time, "sleep", lambda *_: None)

        class _Client:
            class models:
                @staticmethod
                def generate_content(model, contents, config):
                    raise RuntimeError("permanent failure")

        with pytest.raises(RuntimeError, match="permanent failure"):
            rn._call_gemini_with_retry(_Client(), "model", "prompt", {}, max_retries=1)


class TestAddTokenUsage:
    def test_never_raises_even_if_langsmith_import_fails(self):
        # Best-effort only -- must not break report generation over a metadata hiccup.
        class _Resp:
            usage_metadata = None
        rn._add_token_usage(_Resp())  # should not raise
