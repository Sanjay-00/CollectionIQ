"""query_log.py: local outcome logging for AI Query runs.

The point of this file is finding out what real users ask that the registry
vocabulary doesn't cover yet - so classify_outcome's categories and
log_query_outcome's "never raise" guarantee are both worth locking in.
"""
import datetime
import json

import pandas as pd
import pytest

import query_log
from query_log import classify_outcome, log_query_outcome, _prune_old_entries


class TestClassifyOutcome:
    def test_clarification_takes_priority(self):
        state = {"needs_clarification": True, "error": "some error"}
        assert classify_outcome(state) == "clarification"

    def test_error_without_clarification(self):
        state = {"needs_clarification": False, "error": "compile failed"}
        assert classify_outcome(state) == "error"

    def test_priority_mode(self):
        state = {"priority_mode": True}
        assert classify_outcome(state) == "priority_mode"

    def test_view_hit(self):
        state = {"ir1": {"view": {"name": "executive_scorecard", "params": {}}}}
        assert classify_outcome(state) == "view_hit"

    def test_view_hit_when_view_is_a_bare_string(self):
        state = {"ir1": {"view": "executive_scorecard"}}
        assert classify_outcome(state) == "view_hit"

    def test_compiled_ok_default(self):
        state = {"ir1": {"view": None}}
        assert classify_outcome(state) == "compiled_ok"

    def test_empty_state_is_compiled_ok(self):
        assert classify_outcome({}) == "compiled_ok"


class TestLogQueryOutcome:
    def _read_lines(self, path):
        with open(path, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    def test_writes_one_json_line_with_expected_fields(self, tmp_path, monkeypatch):
        log_path = tmp_path / "sub" / "query_log.jsonl"
        monkeypatch.setattr(query_log, "QUERY_LOG_PATH", str(log_path))
        monkeypatch.setattr(query_log, "QUERY_LOG_ENABLED", True)

        state = {
            "run_id": "abc-123",
            "query": "who hasn't paid this month in Pune",
            "needs_clarification": False,
            "error": "",
            "ir1": {"intent": "loan_table", "view": None},
            "result_df": pd.DataFrame({"Loan No": ["L1", "L2"]}),
        }
        log_query_outcome(state)

        entries = self._read_lines(log_path)
        assert len(entries) == 1
        e = entries[0]
        assert e["run_id"] == "abc-123"
        assert e["query"] == "who hasn't paid this month in Pune"
        assert e["outcome"] == "compiled_ok"
        assert e["intent"] == "loan_table"
        assert e["result_rows"] == 2

    def test_creates_parent_directory(self, tmp_path, monkeypatch):
        log_path = tmp_path / "does" / "not" / "exist" / "query_log.jsonl"
        monkeypatch.setattr(query_log, "QUERY_LOG_PATH", str(log_path))
        monkeypatch.setattr(query_log, "QUERY_LOG_ENABLED", True)

        log_query_outcome({"query": "test"})
        assert log_path.exists()

    def test_appends_rather_than_overwrites(self, tmp_path, monkeypatch):
        log_path = tmp_path / "query_log.jsonl"
        monkeypatch.setattr(query_log, "QUERY_LOG_PATH", str(log_path))
        monkeypatch.setattr(query_log, "QUERY_LOG_ENABLED", True)

        log_query_outcome({"query": "first"})
        log_query_outcome({"query": "second"})
        entries = self._read_lines(log_path)
        assert [e["query"] for e in entries] == ["first", "second"]

    def test_disabled_writes_nothing(self, tmp_path, monkeypatch):
        log_path = tmp_path / "query_log.jsonl"
        monkeypatch.setattr(query_log, "QUERY_LOG_PATH", str(log_path))
        monkeypatch.setattr(query_log, "QUERY_LOG_ENABLED", False)

        log_query_outcome({"query": "should not be written"})
        assert not log_path.exists()

    def test_never_raises_on_bad_state(self, monkeypatch):
        # A state missing every expected key, or with wrong types, must not
        # propagate an exception into the real query pipeline.
        monkeypatch.setattr(query_log, "QUERY_LOG_ENABLED", True)
        monkeypatch.setattr(query_log, "QUERY_LOG_PATH", "\0invalid\0path")
        log_query_outcome({"ir1": "not-a-dict", "result_df": object()})
        # No exception means the test passes.

    def test_captures_error_and_clarification_text(self, tmp_path, monkeypatch):
        log_path = tmp_path / "query_log.jsonl"
        monkeypatch.setattr(query_log, "QUERY_LOG_PATH", str(log_path))
        monkeypatch.setattr(query_log, "QUERY_LOG_ENABLED", True)

        log_query_outcome({
            "query": "ambiguous question",
            "needs_clarification": True,
            "clarification_question": "Which region did you mean?",
        })
        entries = self._read_lines(log_path)
        assert entries[0]["outcome"] == "clarification"
        assert entries[0]["clarification_question"] == "Which region did you mean?"


class TestRetentionPruning:
    def _write_raw(self, path, entries):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for e in entries:
                f.write(json.dumps(e) + "\n")

    def test_entries_older_than_retention_are_dropped(self, tmp_path, monkeypatch):
        log_path = tmp_path / "query_log.jsonl"
        monkeypatch.setattr(query_log, "QUERY_LOG_RETENTION_DAYS", 30)
        now = datetime.datetime.now()
        old_ts = (now - datetime.timedelta(days=45)).isoformat(timespec="seconds")
        recent_ts = (now - datetime.timedelta(days=5)).isoformat(timespec="seconds")
        self._write_raw(log_path, [
            {"ts": old_ts, "query": "ancient"},
            {"ts": recent_ts, "query": "recent"},
        ])

        _prune_old_entries(log_path)

        with open(log_path, encoding="utf-8") as f:
            entries = [json.loads(l) for l in f if l.strip()]
        assert [e["query"] for e in entries] == ["recent"]

    def test_unparseable_line_is_dropped_not_fatal(self, tmp_path, monkeypatch):
        log_path = tmp_path / "query_log.jsonl"
        monkeypatch.setattr(query_log, "QUERY_LOG_RETENTION_DAYS", 30)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("not valid json at all\n")
            f.write(json.dumps({"ts": datetime.datetime.now().isoformat(), "query": "ok"}) + "\n")

        _prune_old_entries(log_path)  # must not raise

        with open(log_path, encoding="utf-8") as f:
            entries = [json.loads(l) for l in f if l.strip()]
        assert [e["query"] for e in entries] == ["ok"]

    def test_second_call_within_interval_is_a_no_op(self, tmp_path, monkeypatch):
        log_path = tmp_path / "query_log.jsonl"
        monkeypatch.setattr(query_log, "QUERY_LOG_RETENTION_DAYS", 30)
        old_ts = (datetime.datetime.now() - datetime.timedelta(days=45)).isoformat(timespec="seconds")
        self._write_raw(log_path, [{"ts": old_ts, "query": "ancient"}])

        _prune_old_entries(log_path)  # first call: prunes, writes the marker
        # Re-add an old entry directly (simulating it slipping in between checks)
        # and confirm a second call within the interval does NOT re-scan/prune it.
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": old_ts, "query": "ancient-again"}) + "\n")
        _prune_old_entries(log_path)

        with open(log_path, encoding="utf-8") as f:
            entries = [json.loads(l) for l in f if l.strip()]
        assert [e["query"] for e in entries] == ["ancient-again"]

    def test_log_query_outcome_prunes_before_appending(self, tmp_path, monkeypatch):
        log_path = tmp_path / "query_log.jsonl"
        monkeypatch.setattr(query_log, "QUERY_LOG_PATH", str(log_path))
        monkeypatch.setattr(query_log, "QUERY_LOG_ENABLED", True)
        monkeypatch.setattr(query_log, "QUERY_LOG_RETENTION_DAYS", 30)
        old_ts = (datetime.datetime.now() - datetime.timedelta(days=45)).isoformat(timespec="seconds")
        self._write_raw(log_path, [{"ts": old_ts, "query": "ancient"}])

        log_query_outcome({"query": "new query"})

        with open(log_path, encoding="utf-8") as f:
            entries = [json.loads(l) for l in f if l.strip()]
        assert [e["query"] for e in entries] == ["new query"]
