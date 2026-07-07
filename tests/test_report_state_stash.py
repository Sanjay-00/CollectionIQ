"""Regression coverage for report_agent/state.py's _stash_large/_fetch_large --
the fix for a ~120MB-per-node-transition LangSmith trace payload (6x the 20MB
limit) caused by df_curr/df_prev sitting directly in ReportState, which
LangGraph serializes on every node transition regardless of any individual
node's own @traceable wrapper. Same bug, same fix shape as graph.py's own
_stash_large/_fetch_large for the AI Query pipeline.
"""
import pandas as pd
import pytest

from report_agent import state as report_state
from report_agent.graph import run_report
from report_agent.nodes.portfolio_analyzer import portfolio_analyzer_node
from helpers import make_df


@pytest.fixture(autouse=True)
def _clear_stash():
    """Thread-local stash persists across tests in the same process/thread --
    clear it before and after every test in this file so a stashed df from
    one test can never leak into another test's direct node-function call
    (which relies on _fetch_large's state.get(...) fallback, only correct
    when the stash is empty)."""
    if hasattr(report_state._tls, "large_data"):
        report_state._tls.large_data.clear()
    yield
    if hasattr(report_state._tls, "large_data"):
        report_state._tls.large_data.clear()


class TestStashFetchRoundTrip:
    def test_fetch_returns_stashed_value_over_state(self):
        report_state._stash_large("df_curr", "STASHED")
        assert report_state._fetch_large({"df_curr": "IN_STATE"}, "df_curr") == "STASHED"

    def test_fetch_falls_back_to_state_when_nothing_stashed(self):
        assert report_state._fetch_large({"df_curr": "IN_STATE"}, "df_curr") == "IN_STATE"

    def test_fetch_falls_back_to_default_when_neither_present(self):
        assert report_state._fetch_large({}, "df_curr", default="fallback") == "fallback"


class TestRunReportNeverPutsRealFrameInState:
    def test_initial_state_df_curr_is_placeholder_not_real_frame(self, monkeypatch):
        captured = {}

        def _fake_invoke(initial, config=None):
            captured.update(initial)
            return {**initial, "section_data": {}, "html_report": "<html></html>"}

        monkeypatch.setattr("report_agent.graph._compiled.invoke", _fake_invoke)

        real_df = pd.DataFrame({"Loan No": ["L1", "L2"], "RegionName": ["WEST", "EAST"]})
        run_report(df_curr=real_df, df_prev=pd.DataFrame(), curr_month="2026-06", skip_ai=True)

        # The real DataFrame must never appear as the value LangGraph would
        # trace -- only a cheap placeholder does.
        assert captured["df_curr"] is None
        assert captured["df_prev"] is None

    def test_stashed_value_is_the_real_dataframe(self, monkeypatch):
        monkeypatch.setattr(
            "report_agent.graph._compiled.invoke",
            lambda initial, config=None: {**initial, "section_data": {}, "html_report": ""},
        )
        real_df = pd.DataFrame({"Loan No": ["L1"]})
        run_report(df_curr=real_df, df_prev=pd.DataFrame(), curr_month="2026-06", skip_ai=True)
        assert report_state._tls.large_data["df_curr"] is real_df


class TestPortfolioAnalyzerReadsStashedFrame:
    def test_uses_stashed_frame_when_state_has_placeholder(self):
        real_df = make_df([{"RegionName": "WEST", "Unit": "MAHAD", "curr_bucket": "STD"}])
        report_state._stash_large("df_curr", real_df)
        report_state._stash_large("df_prev", pd.DataFrame())

        state = {
            "df_curr": None, "df_prev": None, "curr_month": "2026-06",
            "enabled_sections": ["portfolio_health"], "filters_applied": {},
        }
        out = portfolio_analyzer_node(state)
        assert "portfolio_health" in out["section_data"]

    def test_falls_back_to_state_when_stash_empty(self):
        # No _stash_large call here -- this is the direct-call test pattern
        # tests/test_report_error_logging.py already relies on; must keep
        # working unchanged.
        real_df = make_df([{"RegionName": "WEST", "Unit": "MAHAD", "curr_bucket": "STD"}])
        state = {
            "df_curr": real_df, "df_prev": pd.DataFrame(), "curr_month": "2026-06",
            "enabled_sections": ["portfolio_health"], "filters_applied": {},
        }
        out = portfolio_analyzer_node(state)
        assert "portfolio_health" in out["section_data"]
