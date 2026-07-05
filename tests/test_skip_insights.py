"""Regression coverage for the skip_insights opt-out (graph.py, ui/tabs/ai_query.py).

Two independent contracts, each previously untested:
1. graph.py::analyze_node must not call the Insight Generator's Gemini call
   (generate_insights) at all when skip_insights=True - the checkbox is
   framed to the user as "off by default: skips the 2nd Gemini call", so a
   silent call would both cost money/latency and contradict the UI copy.
2. ui/tabs/ai_query.py's AI-result cache key includes skip_insights, so a
   query cached WITHOUT the AI summary must not be served back once the user
   turns the checkbox on and re-asks the identical question (or vice versa).
"""
import graph
from ui.tabs.ai_query import _ai_cache_get, _ai_cache_put, _AI_CACHE_MAX_ENTRIES


class TestAnalyzeNodeSkipInsights:
    def test_skip_insights_true_never_calls_generate_insights(self, monkeypatch):
        calls = []
        monkeypatch.setattr(graph, "generate_insights", lambda **kw: calls.append(kw) or "should not happen")

        state = {"skip_insights": True, "query": "q", "ir1": {}, "result_kpis": {}, "result_rankings": {}}
        result = graph.analyze_node(state)

        assert calls == []
        assert result["insights"] == ""

    def test_skip_insights_false_calls_generate_insights(self, monkeypatch):
        calls = []
        monkeypatch.setattr(graph, "generate_insights", lambda **kw: calls.append(kw) or "• bullet")

        state = {"skip_insights": False, "query": "q", "ir1": {}, "result_kpis": {}, "result_rankings": {}}
        result = graph.analyze_node(state)

        assert len(calls) == 1
        assert result["insights"] == "• bullet"

    def test_skip_insights_absent_defaults_to_calling_generate_insights(self, monkeypatch):
        calls = []
        monkeypatch.setattr(graph, "generate_insights", lambda **kw: calls.append(kw) or "• bullet")

        state = {"query": "q", "ir1": {}, "result_kpis": {}, "result_rankings": {}}
        result = graph.analyze_node(state)

        assert len(calls) == 1
        assert result["insights"] == "• bullet"


class TestAiQueryCacheKeyIncludesSkipInsights:
    def test_no_insights_cache_entry_not_served_when_checkbox_turned_on(self):
        cache = {}
        no_insights_key = ("show npa loans", 1, "region=ALL", True)   # skip_insights=True
        with_insights_key = ("show npa loans", 1, "region=ALL", False)  # skip_insights=False

        _ai_cache_put(cache, no_insights_key, {"insights": "", "result_df": None})

        assert _ai_cache_get(cache, no_insights_key) is not None
        assert _ai_cache_get(cache, with_insights_key) is None

    def test_identical_key_including_skip_insights_is_a_hit(self):
        cache = {}
        key = ("show npa loans", 1, "region=ALL", False)
        value = {"insights": "• bullet", "result_df": None}
        _ai_cache_put(cache, key, value)
        assert _ai_cache_get(cache, key) is value

    def test_cache_evicts_oldest_entry_at_capacity(self):
        cache = {}
        for i in range(_AI_CACHE_MAX_ENTRIES):
            _ai_cache_put(cache, (f"q{i}", 1, "region=ALL", False), {"n": i})
        assert len(cache) == _AI_CACHE_MAX_ENTRIES

        first_key = (f"q0", 1, "region=ALL", False)
        _ai_cache_put(cache, ("q_new", 1, "region=ALL", False), {"n": "new"})

        assert len(cache) == _AI_CACHE_MAX_ENTRIES
        assert _ai_cache_get(cache, first_key) is None  # oldest evicted
        assert _ai_cache_get(cache, ("q_new", 1, "region=ALL", False)) == {"n": "new"}
