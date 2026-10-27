"""AI Query plan cache (code review item 3, 2026-10-27): the planner's one
Gemini call is about 90% of a question's time, and its plan depends only on
the question and the planner's prompt (vocabulary and reporting months), not
on the data. A plan that answered a question is reused for the same
question, in any session, with no AI call."""
import copy
import threading

import pytest

import graph
from agents.plan_cache import PLAN_CACHE, PlanCache
from helpers import make_df

DATES = {"curr": "2026-09-30", "prev": "2026-08-31"}
DELINQUENT = {"intent": "loan_table", "query_title": "Delinquent loans", "description": "loans with arrears",
              "filters": [{"concept": "delinquent"}], "dimensions": [], "measures": [], "view": None}


class _Planner:
    """A scripted planner; len() is its call count (each one is a Gemini call in the app)."""
    def __init__(self):
        self.ir, self.calls, self.logged = DELINQUENT, [], []

    def plan(self, query, snapshot_dates=None, repair_feedback="", allow_clarification=True):
        self.calls.append((query, repair_feedback))
        return copy.deepcopy(self.ir)

    def __len__(self):
        return len(self.calls)


@pytest.fixture
def planner(monkeypatch):
    p = _Planner()
    monkeypatch.setattr(graph, "plan_logical", p.plan)
    monkeypatch.setattr(graph, "log_query_outcome", p.logged.append)
    return p


def _loans():
    return make_df([{"Arrears / EMI": 0.0}, {"Arrears / EMI": 1.5}, {"Arrears / EMI": 3.0}])


def _ask(question="show delinquent loans", dates=DATES, steps=None):
    return graph.run_query(question, _loans(), on_step=(steps.append if steps is not None else None),
                           snapshot_dates=dates, skip_insights=True)


class TestReuse:
    def test_the_same_question_is_planned_once(self, planner):
        first, second = _ask(), _ask()
        assert len(planner) == 1
        assert list(second["result_df"]["Loan No"]) == list(first["result_df"]["Loan No"]) == ["L002", "L003"]
        assert (first["plan_cache"], second["plan_cache"]) == ("miss", "hit")

    def test_spacing_does_not_matter(self, planner):
        _ask("show delinquent loans")
        _ask("  show   delinquent\nloans ")
        assert len(planner) == 1

    def test_case_does(self, planner):
        """Case can carry meaning (a branch or executive name in a filter), and
        a cache must never change an answer, so different case is a new plan."""
        _ask("loans in branch PUNE")
        _ask("loans in branch Pune")
        assert len(planner) == 2

    def test_the_steps_say_the_plan_was_reused(self, planner):
        steps = []
        _ask()
        _ask(steps=steps)
        assert any("reused" in s.lower() and "no ai call" in s.lower() for s in steps), steps

    def test_the_hit_or_miss_reaches_the_query_log(self, planner):
        _ask(), _ask()
        assert [r["plan_cache"] for r in planner.logged] == ["miss", "hit"]


class TestWhenNotToReuse:
    def test_other_reporting_months_are_a_new_plan(self, planner):
        _ask()
        _ask(dates={"curr": "2026-10-31", "prev": "2026-09-30"})
        _ask(dates={"curr": "2026-09-30"})                     # no last-month file loaded
        assert len(planner) == 3

    def test_a_question_that_failed_is_not_kept(self, planner):
        planner.ir = {**DELINQUENT, "filters": [{"column": "No Such Column", "op": ">", "value": 1}]}
        assert _ask()["error"]
        planner.ir = DELINQUENT
        assert not _ask()["error"]
        assert len(planner) == 3                                 # the failed run: plan + repair; then the good plan
        assert _ask()["plan_cache"] == "hit"

    def test_a_clarification_is_not_kept(self, planner):
        planner.ir = {**DELINQUENT, "needs_clarification": True, "clarification_question": "Which one?",
                      "clarification_options": ["A", "B"]}
        _ask(), _ask()
        assert len(planner) == 2

    def test_a_reused_plan_that_now_fails_is_dropped(self, planner, monkeypatch):
        _ask()
        broken = {**DELINQUENT, "filters": [{"column": "No Such Column", "op": ">", "value": 1}]}
        key = next(iter(PLAN_CACHE._entries))
        PLAN_CACHE._entries[key] = broken                      # e.g. a file without a column the plan uses
        planner.ir = broken
        assert _ask()["error"]
        assert len(PLAN_CACHE) == 0                             # dropped, so it can't keep failing


class TestPlanCache:
    def test_oldest_out_when_full(self):
        cache = PlanCache(max_entries=2)
        for k in ("a", "b", "c"):
            cache.put(k, {"k": k})
        assert cache.get("a") is None and cache.get("c") == {"k": "c"} and len(cache) == 2

    def test_a_reused_plan_cannot_be_changed_by_its_caller(self):
        cache = PlanCache()
        cache.put("k", {"filters": [1]})
        cache.get("k")["filters"].append(2)
        assert cache.get("k") == {"filters": [1]}

    def test_counts_hits_and_misses(self):
        cache = PlanCache()
        cache.get("x")
        cache.put("x", {})
        cache.get("x")
        assert cache.stats() == {"hits": 1, "misses": 1, "entries": 1}

    def test_safe_across_threads(self):
        cache = PlanCache(max_entries=50)
        threads = [threading.Thread(target=lambda i=i: [cache.put(f"k{i % 60}", {"i": i}), cache.get(f"k{i}")])
                   for i in range(300)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(cache) == 50
