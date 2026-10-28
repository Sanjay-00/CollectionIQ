"""LangSmith tracing beyond AI Query (2026-10-27). Only the AI Query pipeline
was traced; the Investigator, the review analyst and the report summary
called Gemini with no trace at all, so 4 days of Investigator use showed
nothing in LangSmith. Every Gemini call is now an "llm" run (gemini_client),
under a named run per feature, and a hedged call records its time and
whether a backup was sent (code review item 2 follow-up)."""
import json
import threading
import time

import pandas as pd
import pytest

import gemini_client as gc
from investigator.state import EntityMemory


class _Reply:
    def __init__(self, text="{}"):
        self.text = text
        self.usage_metadata = type("U", (), {"prompt_token_count": 100, "candidates_token_count": 20,
                                             "total_token_count": 120})()


class _Client:
    def __init__(self, *delays):
        self.delays, self.calls, self.lock = list(delays), 0, threading.Lock()
        self.models = self

    def generate_content(self, model, contents, config):
        with self.lock:
            delay = self.delays[min(self.calls, len(self.delays) - 1)]
            self.calls += 1
        time.sleep(delay)
        return _Reply()


class TestWhatATraceHolds:
    def test_the_long_system_prompt_is_counted_not_copied(self):
        out = gc._trace_call_inputs({"client": object(), "model": "m", "contents": "question",
                                     "config": {"temperature": 0.1, "system_instruction": "x" * 41_000}})
        assert out["config"] == {"temperature": 0.1, "system_instruction_chars": 41_000}
        assert "client" not in out and out["prompt"] == "question"

    def test_a_long_prompt_is_cut(self):
        out = gc._trace_call_inputs({"model": "m", "contents": "y" * 50_000, "config": {}})
        assert len(out["prompt"]) <= gc.TRACE_TEXT_CHARS and out["prompt_chars"] == 50_000

    def test_token_counts_are_reported_the_way_langsmith_reads_them(self):
        out = gc._trace_call_output(_Reply("hello"))
        assert out["usage_metadata"] == {"input_tokens": 100, "output_tokens": 20, "total_tokens": 120}
        assert out["text"] == "hello"

    def test_router_trace_carries_no_tables(self):
        """The 2026 incident in graph.py: whole DataFrames in a trace made 20-85MB
        payloads that failed to upload and retried for 60s+ per query."""
        from investigator import llm
        memory = EntityMemory()
        memory.touch("branch", "A", result_df=pd.DataFrame({"x": range(50_000)}), step_type="entity_summary")
        out = llm._trace_turn_inputs({"question": "why", "entity_memory": memory, "allow_multi": True})
        assert len(json.dumps(out, default=str)) < 2_000
        assert out["question"] == "why" and out["remembered"] == 1

    def test_narration_trace_carries_the_shape_not_the_table(self):
        from investigator import llm
        out = llm._trace_narration_inputs({"step_type": "entity_summary",
                                            "result_df": pd.DataFrame({"x": range(50_000)})})
        assert out == {"step_type": "entity_summary", "rows": 50_000, "columns": ["x"]}


class TestHedgeRecord:
    def test_a_quick_call_records_its_time_and_no_backup(self):
        gc.call_gemini_hedged(_Client(0.01), "m", "q", {}, hedge_after_s=0.2)
        info = gc.last_call_info()
        assert info["hedged"] is False and info["backup_won"] is False and 0 < info["seconds"] < 0.2

    def test_a_slow_call_records_the_backup(self):
        gc.call_gemini_hedged(_Client(0.6, 0.01), "m", "q", {}, hedge_after_s=0.1)
        info = gc.last_call_info()
        assert info["hedged"] is True and info["backup_won"] is True and info["seconds"] < 0.5

    def test_reading_it_clears_it(self):
        gc.call_gemini_hedged(_Client(0.01), "m", "q", {}, hedge_after_s=0.2)
        gc.last_call_info()
        assert gc.last_call_info() == {}

    def test_it_goes_on_the_current_trace(self, monkeypatch):
        seen = {}
        tree = type("T", (), {"add_metadata": lambda self, d: seen.update(d)})()
        monkeypatch.setattr(gc, "_current_run", lambda: tree)
        gc.call_gemini_hedged(_Client(0.01), "m", "q", {}, hedge_after_s=0.2)
        assert {"gemini_seconds", "hedged", "backup_won"} <= set(seen)

    def test_the_planner_call_reaches_the_query_log(self, monkeypatch, tmp_path):
        import graph
        logged = []

        def plan(query, snapshot_dates=None, repair_feedback="", allow_clarification=True):
            gc.call_gemini_hedged(_Client(0.01), "m", "q", {}, hedge_after_s=0.2)
            return {"intent": "loan_table", "query_title": "t", "filters": [], "dimensions": [], "measures": [],
                    "view": None}
        monkeypatch.setattr(graph, "plan_logical", plan)
        monkeypatch.setattr(graph, "log_query_outcome", logged.append)
        df = pd.DataFrame({"Loan No": ["L1"], "Arrears / EMI": [0.0]})
        result = graph.run_query("all loans", df, snapshot_dates={"curr": "2026-09-30"}, skip_insights=True)
        assert result["planner_call"]["hedged"] is False and result["planner_call"]["seconds"] > 0

        import query_log
        monkeypatch.setattr(query_log, "QUERY_LOG_ENABLED", True)
        monkeypatch.setattr(query_log, "QUERY_LOG_PATH", str(tmp_path / "log.jsonl"))
        query_log.log_query_outcome(result)
        entry = json.loads((tmp_path / "log.jsonl").read_text(encoding="utf-8").splitlines()[-1])
        assert entry["planner_seconds"] == result["planner_call"]["seconds"] and entry["planner_hedged"] is False

    def test_a_reused_plan_logs_no_call(self, monkeypatch):
        import graph
        from agents.plan_cache import PLAN_CACHE, planner_cache_key
        ir = {"intent": "loan_table", "query_title": "t", "filters": [], "dimensions": [], "measures": [], "view": None}
        PLAN_CACHE.put(planner_cache_key("all loans", {"curr": "2026-09-30"}, True), ir)
        monkeypatch.setattr(graph, "log_query_outcome", lambda r: None)
        gc.call_gemini_hedged(_Client(0.01), "m", "q", {}, hedge_after_s=0.2)    # a stale record from elsewhere
        df = pd.DataFrame({"Loan No": ["L1"], "Arrears / EMI": [0.0]})
        result = graph.run_query("all loans", df, snapshot_dates={"curr": "2026-09-30"}, skip_insights=True)
        assert result["plan_cache"] == "hit" and result["planner_call"] == {}


class TestEveryFeatureIsTraced:
    @pytest.mark.parametrize("module, fn, name", [
        ("investigator.llm", "parse_turn", "InvestigatorRouter"),
        ("investigator.llm", "narrate_step", "InvestigatorNarration"),
        ("investigator.llm", "narrate_step_stream", "InvestigatorNarration"),
        ("investigation.review.analyst", "gemini_generate", "ReviewAnalyst"),
        ("report_agent.ai_summary", "polish", "ReportSummary"),
    ])
    def test_named_run(self, module, fn, name):
        import importlib
        f = getattr(importlib.import_module(module), fn)
        assert getattr(f, "__langsmith_traceable__", False) or hasattr(f, "__wrapped__"), f"{module}.{fn} is not traced"
        assert name in _traced_names(module)


def _traced_names(module):
    import importlib
    import inspect
    return inspect.getsource(importlib.import_module(module))
