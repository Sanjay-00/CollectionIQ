"""Broad `except` audit (code review item 7, 2026-10-28). 37 broad handlers in
the app: most re-raise, show the error, or are telemetry that must never break
a query. These are the ones that hid something: an untrue "saved" message, and
failures that reached the user but left no traceback for whoever debugs it."""
import logging

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest


class TestFeedback:
    def test_a_failed_send_says_so(self, monkeypatch):
        import langsmith
        from ui.components import _send_feedback

        class Down:
            def create_feedback(self, **kw):
                raise ConnectionError("no network")
        monkeypatch.setattr(langsmith, "Client", lambda: Down())
        assert _send_feedback("run-1", 1.0) is False

    def test_a_sent_one_says_so(self, monkeypatch):
        import langsmith
        from ui.components import _send_feedback
        sent = {}
        monkeypatch.setattr(langsmith, "Client", lambda: type("C", (), {"create_feedback": lambda self, **kw: sent.update(kw)})())
        assert _send_feedback("run-1", 0.0) is True and sent == {"run_id": "run-1", "key": "result_quality", "score": 0.0}

    @pytest.mark.parametrize("ok, expected", [(True, "Feedback saved to LangSmith"),
                                              (False, "Couldn't save your feedback")])
    def test_the_toast_tells_the_truth(self, monkeypatch, ok, expected):
        """It said "Feedback saved to LangSmith" whether or not the save worked."""
        monkeypatch.setenv("LANGSMITH_API_KEY", "test")
        monkeypatch.setattr("ui.query_answer._send_feedback", lambda run_id, score: ok)

        def page():
            from ui.query_answer import _render_feedback
            _render_feedback({"run_id": "run-1"}, "k")
        at = AppTest.from_function(page).run()
        at.button(key="k_fb_up").click().run()
        assert [t.value for t in at.toast][-1].startswith(expected)


class TestFailuresAreLogged:
    def test_an_ai_query_step_that_crashes_logs_its_traceback(self, monkeypatch, caplog):
        import graph

        def boom(*a, **k):
            raise RuntimeError("planner exploded")
        monkeypatch.setattr(graph, "plan_logical", boom)
        monkeypatch.setattr(graph, "log_query_outcome", lambda r: None)
        with caplog.at_level(logging.ERROR, logger="graph"):
            result = graph.run_query("anything", pd.DataFrame({"Loan No": ["L1"]}), skip_insights=True)
        assert result["error"].startswith("Query planning failed")          # the user still gets the message
        assert any(r.exc_info and "planner exploded" in str(r.exc_info[1]) for r in caplog.records)

    def test_a_query_log_that_cannot_be_written_is_reported(self, monkeypatch, tmp_path, caplog):
        import query_log
        monkeypatch.setattr(query_log, "QUERY_LOG_ENABLED", True)
        monkeypatch.setattr(query_log, "QUERY_LOG_PATH", str(tmp_path))      # a folder: can't be opened for append
        with caplog.at_level(logging.WARNING, logger="query_log"):
            query_log.log_query_outcome({"query": "q"})                       # still never raises
        assert any("query log" in r.getMessage().lower() for r in caplog.records)

    def test_a_tab_that_crashes_logs_its_traceback(self, caplog):
        def page():
            from ui.components import render_tab_error
            try:
                raise KeyError("Unit")
            except KeyError as e:
                render_tab_error("Dashboard", e)
        with caplog.at_level(logging.ERROR, logger="ui.components"):
            at = AppTest.from_function(page).run()
        assert "Dashboard tab failed to render" in at.error[0].value
        assert any(r.exc_info and r.exc_info[0] is KeyError for r in caplog.records)
