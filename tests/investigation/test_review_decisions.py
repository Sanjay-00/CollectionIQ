"""The decision log and the next steps after a choice (spec 7.3, 7.4)."""
import json
import threading

import pytest

from investigation.review import runner as R
from investigation.review.decisions import DecisionLog
from investigation.review.next_steps import largest_branch, next_steps
from investigation.review.synthetic import REVIEW_CURR_MONTH as CURR, REVIEW_PREV_MONTH as PREV, make_review_book


def _entry(**over):
    base = dict(data_version="v1", run="r1", segment="review", option_id="investigate:collection_shortfall:WEST",
                region="WEST", list_name="collection_shortfall", ai_recommended="investigate:collection_shortfall:WEST",
                followed_ai=True, evidence_ids=["E4", "E31"], period={"prev": PREV, "curr": CURR})
    return {**base, **over}


class TestDecisionLog:
    def test_a_repeated_choice_is_one_entry(self):
        log = DecisionLog()
        first, new = log.append(**_entry())
        again, new_again = log.append(**_entry(note="clicked twice"))
        assert new and not new_again and again is first and len(log.for_data("v1")) == 1

    def test_a_step_with_no_evidence_or_period_is_recorded(self):
        # End-to-end run: next steps, actions and "handled" pass no evidence
        # ids, and every one of them failed to record (a field named "list"
        # shadowed the built-in in default_factory=list).
        d, new = DecisionLog().append(data_version="v1", run="r1", segment="close", option_id="close:x:WEST")
        assert new and d.evidence_ids == [] and d.period == {}

    def test_it_survives_a_restart(self, tmp_path):
        path = tmp_path / "log.jsonl"
        DecisionLog(path).append(**_entry())
        DecisionLog(path).append(**_entry(segment="close", option_id="close:collection_shortfall:WEST"))
        reloaded = DecisionLog(path).for_data("v1")
        assert [d.segment for d in reloaded] == ["review", "close"]
        assert reloaded[0].evidence_ids == ["E4", "E31"] and reloaded[0].followed_ai is True

    def test_only_decisions_on_the_same_files_are_shown(self):
        log = DecisionLog()
        log.append(**_entry())
        log.append(**_entry(data_version="v2"))
        assert [d.data_version for d in log.for_data("v1")] == ["v1"]

    def test_download_and_upload_round_trip(self, tmp_path):
        a = DecisionLog()
        a.append(**_entry())
        a.append(**_entry(segment="action", option_id="email_draft:NPA Accounts:WES-BR1", branch="WES-BR1"))
        b = DecisionLog(tmp_path / "other.jsonl")
        assert b.import_jsonl(a.export_jsonl()) == (2, 0)
        assert b.import_jsonl(a.export_jsonl()) == (0, 2)              # uploading twice adds nothing
        assert len(DecisionLog(tmp_path / "other.jsonl").for_data("v1")) == 2   # and it was written to disk

    def test_a_download_holds_only_these_files_decisions(self):
        # The server's file holds every session's decisions; one session's
        # download must not carry another's files' choices.
        log = DecisionLog()
        log.append(**_entry())
        log.append(**_entry(data_version="someone_elses_files", region="NORTH"))
        lines = log.export_jsonl("v1").decode().splitlines()
        assert len(lines) == 1 and json.loads(lines[0])["data_version"] == "v1"

    def test_an_uploaded_log_cannot_smuggle_anything_in(self):
        good = json.dumps(_entry() | {"id": "abc", "at": "2026-10-25 10:00:00"})
        bad = [
            "not json at all",
            json.dumps(_entry(segment="delete_everything")),
            json.dumps(_entry(note="x" * 5000)),
            json.dumps(_entry(region=["a list"])),
            json.dumps(_entry(option_id="stop") | {"customer_rows": [{"Cust Name": "SECRET"}]}),   # unknown key: dropped
        ]
        log = DecisionLog()
        added, skipped = log.import_jsonl("\n".join([good, *bad]).encode())
        assert added == 2 and skipped == 4                    # the last one loads, without its extra key
        assert "SECRET" not in log.export_jsonl().decode()

    def test_an_unknown_step_is_refused(self):
        with pytest.raises(ValueError):
            DecisionLog().append(**_entry(segment="send_email"))

    def test_concurrent_clicks_never_duplicate(self, tmp_path):
        log = DecisionLog(tmp_path / "log.jsonl")
        threads = [threading.Thread(target=log.append, kwargs=_entry(option_id=f"opt{i % 5}")) for i in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(log.for_data("v1")) == 5
        assert len((tmp_path / "log.jsonl").read_text(encoding="utf-8").splitlines()) == 5


class TestTheAppsLogging:
    def test_the_panel_records_the_reviews_book(self, monkeypatch):
        # Two-month end-to-end run: the review context carried the book but
        # log_decision dropped it, so next month found no outcomes at all.
        import ui.review_panel as rp
        log = DecisionLog()
        monkeypatch.setattr(rp, "decision_log", lambda: log)
        ctx = {"data_version": "v1", "run": "r1", "period": {"curr": CURR, "prev": PREV}, "book": "book-1",
               "ai_recommended": None}
        rp.log_decision(ctx, "close", "close:collection_shortfall:WEST", region="WEST")
        [d] = log.entries_for_book("book-1")
        assert d.book == "book-1" and d.period == {"curr": CURR, "prev": PREV}


class TestNextSteps:
    def test_a_named_branch_comes_first_then_region_then_close_and_back(self):
        steps = next_steps("WEST", "collection_shortfall", "WES-BR1")
        assert [s.id for s in steps] == ["priority_branch:WES-BR1", "priority_region:WEST",
                                         "close:collection_shortfall:WEST", "back"]
        assert steps[0].step_type == "priority_menu" and steps[0].params == {"scope_col": "branch", "scope_value": "WES-BR1"}
        assert steps[1].params == {"scope_col": "region", "scope_value": "WEST"}

    def test_without_a_branch_the_region_list_is_first(self):
        assert [s.id for s in next_steps("EAST", "worsening")][0] == "priority_region:EAST"

    def test_the_largest_branch_comes_from_the_deep_dive(self):
        prev, curr = make_review_book(shortfall={"WEST": 0.85})
        brief = R.run_review(curr, prev, CURR, PREV)
        dive = next(d for d in brief.deep_dives if d.region == "WEST")
        assert largest_branch(brief, dive.candidate_id) == dive.branches[0][0]
        assert largest_branch(brief, "worsening:NOWHERE") is None

    def test_every_next_step_runs_on_a_real_book(self):
        # The step types and params must be ones the Investigator can run.
        from investigator.steps import STEP_REGISTRY
        prev, curr = make_review_book(shortfall={"WEST": 0.85})
        for s in next_steps("WEST", "collection_shortfall", "WES-BR1"):
            if s.step_type:
                out = STEP_REGISTRY[s.step_type](curr, **s.params, as_of="2026-09-30")
                assert {"Category", "Count"} <= set(out.columns), s.id
