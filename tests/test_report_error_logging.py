"""Regression: a report section that throws must not vanish with zero trace.

Bug: portfolio_analyzer_node stashed the exception under f"{name}_error" in
section_data, but nothing ever read that key, and report_builder_node's own
render loop wrapped each section in a bare `except Exception: pass`. A broken
section disappeared from the (often unattended, SMTP auto-sent) report with
no log line anywhere. Both swallow points now log a warning naming the
section and the error before continuing.
"""
import logging

import pandas as pd

from report_agent.nodes.portfolio_analyzer import portfolio_analyzer_node, _SECTION_FN
from report_agent.nodes.report_builder import report_builder_node, _RENDERERS


def _base_state(**overrides):
    state = {
        "df_curr": pd.DataFrame(), "df_prev": pd.DataFrame(),
        "curr_month": "Jan-2026", "prev_month": "", "enabled_sections": [],
        "filters_applied": {}, "skip_ai": True, "section_data": {},
        "executive_narrative": "", "action_plan": "", "ai_skipped": True,
        "html_report": "", "email_to": "", "email_sent": False,
        "email_error": "", "run_id": "test", "error": "",
    }
    state.update(overrides)
    return state


class TestPortfolioAnalyzerLogsSwallowedErrors:
    def test_failing_section_is_logged_and_recorded(self, monkeypatch, caplog):
        monkeypatch.setitem(_SECTION_FN, "verdict", lambda c, p: (_ for _ in ()).throw(ValueError("boom")))
        state = _base_state(enabled_sections=["verdict"])

        with caplog.at_level(logging.WARNING, logger="report_agent.nodes.portfolio_analyzer"):
            result = portfolio_analyzer_node(state)

        assert result["section_data"]["verdict_error"] == "boom"
        assert any("verdict" in r.message and "boom" in r.message for r in caplog.records)


class TestReportBuilderLogsSwallowedRenderErrors:
    def test_failing_renderer_is_logged(self, monkeypatch, caplog):
        monkeypatch.setitem(_RENDERERS, "verdict", lambda data: (_ for _ in ()).throw(ValueError("render boom")))
        state = _base_state(section_data={"verdict": {"any": "data"}})

        with caplog.at_level(logging.WARNING, logger="report_agent.nodes.report_builder"):
            result = report_builder_node(state)

        assert "html_report" in result
        assert any("verdict" in r.message and "render boom" in r.message for r in caplog.records)
