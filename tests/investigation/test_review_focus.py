"""Focus cards (the Investigator's briefing-and-workspace layout): one card per
region that stands out, ordered and labelled for a leader."""
import pytest

from investigation.review import runner as R
from investigation.review.analyst import Analysis
from investigation.review.decisions import DecisionLog
from investigation.review.focus import HANDLED, LOOKED_INTO, OPEN, focuses
from investigation.review.synthetic import REVIEW_CURR_MONTH as CURR, REVIEW_PREV_MONTH as PREV, make_review_book

PLANTED = dict(shortfall={"WEST": 0.85}, risk={"NORTH": 1.6}, worsen={"EAST": 4.0})


@pytest.fixture(scope="module")
def brief():
    prev, curr = make_review_book(**PLANTED)
    b = R.run_review(curr, prev, CURR, PREV)
    west = next(f for f in b.findings if f.region == "WEST")
    b.analysis = Analysis("ok", "summary", f"investigate:{west.candidate_id}", "why WEST", "", "")
    return b


def _decide(log, brief, segment, option_id, region):
    log.append(data_version=brief.data_version, run="r", segment=segment, option_id=option_id, region=region)


class TestFocuses:
    def test_one_card_per_region_with_all_its_findings(self, brief):
        cards = {f.region: f for f in focuses(brief, [])}
        assert set(cards) == {f.region for f in brief.findings}
        assert len(cards["EAST"].findings) == len([f for f in brief.findings if f.region == "EAST"])
        assert cards["EAST"].lists == [f.list for f in brief.findings if f.region == "EAST"]

    def test_the_ai_pick_comes_first_then_more_lists_then_larger_amounts(self, brief):
        cards = focuses(brief, [])
        assert cards[0].region == "WEST" and cards[0].ai_pick
        rest = cards[1:]
        assert [len(c.lists) for c in rest] == sorted((len(c.lists) for c in rest), reverse=True)

    def test_the_lens_is_the_ai_pick_or_the_looked_into_list(self, brief):
        cards = {f.region: f for f in focuses(brief, [])}
        assert cards["WEST"].lens == brief.analysis.recommended_option.split(":", 1)[1]
        dived = {d.candidate_id for d in brief.deep_dives}
        assert cards["EAST"].lens in dived or cards["EAST"].lens == cards["EAST"].findings[0].candidate_id

    def test_each_list_shows_its_amount(self, brief):
        west = next(f for f in focuses(brief, []) if f.region == "WEST")
        [(name, amount)] = west.chips
        assert name == "collection shortfall" and amount.startswith("₹")

    def test_status_comes_from_the_decision_log(self, brief):
        log = DecisionLog()
        _decide(log, brief, "review", "investigate:collection_shortfall:WEST", "WEST")
        _decide(log, brief, "review", "investigate:excess_30plus:NORTH", "NORTH")
        _decide(log, brief, "close", "close:excess_30plus:NORTH", "NORTH")
        status = {f.region: f.status for f in focuses(brief, log.for_data(brief.data_version))}
        assert status["WEST"] == LOOKED_INTO and status["NORTH"] == HANDLED and status["EAST"] == OPEN

    def test_another_files_decisions_do_not_count(self, brief):
        log = DecisionLog()
        log.append(data_version="other files", run="r", segment="close", option_id="close:x:WEST", region="WEST")
        assert next(f for f in focuses(brief, log.for_data("other files")) if f.region == "WEST").status == OPEN

    def test_nothing_stands_out_means_no_cards(self):
        prev, curr = make_review_book(seed=3)
        assert focuses(R.run_review(curr, prev, CURR, PREV), []) == []
