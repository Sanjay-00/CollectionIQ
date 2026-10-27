"""Hedged Gemini calls (code review item 4, 2026-10-27). Measured live: the
same question with an identical 221-token reply took anywhere from 1.9s to
7.5s, so the slow ones are random server delay, not longer answers. A call
the user waits on with nothing on screen (the AI Query planner, the
Investigator router) sends one identical backup request if the first hasn't
answered by GEMINI_HEDGE_AFTER_S, and takes whichever answers first."""
import threading
import time

import pytest
from google.genai import errors as genai_errors

import gemini_client as gc
from investigator.state import EntityMemory

FAST, SLOW, HEDGE = 0.02, 0.8, 0.15


def _refused():
    """A permanent error (bad request): never retried."""
    return genai_errors.ClientError(400, {"error": {"message": "bad request"}})


class _Client:
    """Answers each request per a script of (delay, reply or error), in call order."""
    def __init__(self, *script):
        self.script, self.calls, self._lock = list(script), 0, threading.Lock()
        self.models = self

    def generate_content(self, model, contents, config):
        with self._lock:
            delay, out = self.script[self.calls]
            self.calls += 1
        time.sleep(delay)
        if isinstance(out, Exception):
            raise out
        return out


def _call(client, hedge=HEDGE):
    start = time.perf_counter()
    reply = gc.call_gemini_hedged(client, "m", "q", {}, hedge_after_s=hedge)
    return reply, time.perf_counter() - start


class TestHedging:
    def test_a_quick_reply_sends_no_backup(self):
        client = _Client((FAST, "first"))
        before = gc.hedge_stats()
        assert _call(client)[0] == "first"
        assert client.calls == 1 and gc.hedge_stats()["hedged"] == before["hedged"]

    def test_a_slow_reply_is_beaten_by_the_backup(self):
        client = _Client((SLOW, "first"), (FAST, "backup"))
        before = gc.hedge_stats()
        reply, took = _call(client)
        assert reply == "backup" and took < SLOW / 2
        after = gc.hedge_stats()
        assert after["hedged"] == before["hedged"] + 1 and after["backup_won"] == before["backup_won"] + 1

    def test_the_first_reply_still_wins_if_it_arrives_first(self):
        client = _Client((HEDGE + 0.1, "first"), (SLOW, "backup"))
        reply, took = _call(client)
        assert reply == "first" and took < SLOW

    def test_a_quick_permanent_error_is_raised_with_no_backup(self):
        client = _Client((FAST, _refused()))
        with pytest.raises(genai_errors.ClientError):
            _call(client)
        time.sleep(HEDGE * 2)
        assert client.calls == 1

    def test_a_slow_failure_falls_back_to_the_backup(self):
        client = _Client((HEDGE + 0.1, _refused()), (0.3, "backup"))
        assert _call(client)[0] == "backup"

    def test_if_both_fail_the_error_is_raised(self):
        client = _Client((HEDGE + 0.05, _refused()), (FAST, _refused()))
        with pytest.raises(genai_errors.ClientError):
            _call(client)

    def test_the_losing_request_stops_retrying(self, monkeypatch):
        """Seen live: the slow first request had hung, timed out after 60s and
        started retrying for a reply nobody wanted any more."""
        monkeypatch.setattr(gc, "time", type("T", (), {"sleep": staticmethod(lambda s: None)}))
        client = _Client((HEDGE + 0.1, TimeoutError("read timed out")), (FAST, "backup"),
                         (FAST, "retry 1"), (FAST, "retry 2"))
        assert _call(client)[0] == "backup"
        time.sleep(HEDGE + 0.2)
        assert client.calls == 2

    def test_zero_turns_hedging_off(self):
        client = _Client((HEDGE + 0.1, "first"))
        assert _call(client, hedge=0)[0] == "first" and client.calls == 1


class TestWhereItIsUsed:
    """Only where the user waits with nothing on screen; narration streams, and
    the summary, reports and the review analyst aren't on that path."""
    def test_the_ai_query_planner_is_hedged(self, monkeypatch):
        import agents.logical_planner as lp
        used = []
        monkeypatch.setenv("GOOGLE_API_KEY", "dummy")
        monkeypatch.setattr(lp, "_add_token_usage", lambda *a, **k: None)
        monkeypatch.setattr(lp, "_call_gemini_hedged",
                            lambda *a, **k: used.append(1) or type("R", (), {"text": '{"intent": "loan_table"}'})())
        lp.plan_logical("show delinquent loans", allow_clarification=False)
        assert used == [1]

    def test_the_investigator_router_is_hedged(self, monkeypatch):
        from investigator import llm
        used = []
        monkeypatch.setenv("GOOGLE_API_KEY", "dummy")
        monkeypatch.setattr(llm, "make_client", lambda key: object())
        monkeypatch.setattr(llm, "_add_token_usage", lambda *a, **k: None)
        monkeypatch.setattr(llm, "_call_gemini_hedged",
                            lambda *a, **k: used.append(1) or type("R", (), {"text": "{}"})())
        llm.parse_turn("why did collection drop", EntityMemory())
        assert used == [1]
