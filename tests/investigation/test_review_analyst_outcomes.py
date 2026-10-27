"""Last month's outcomes in the analyst summary (2026-10-26): the summary
says how last month's chosen regions moved, quoting the outcome's own
figures, and is refused when it gets an outcome's direction wrong."""
import json

import pytest

from investigation.review import runner as R
from investigation.review.analyst import analyst_facts, build_prompt, outcomes_signature, verify, write_analysis
from investigation.review.decisions import DecisionLog
from investigation.review.outcomes import IMPROVED, track_outcomes
from investigation.review.synthetic import make_review_book

AUG, SEP = ("2026-08", "2026-07"), ("2026-09", "2026-08")


def _review(months, **plants):
    prev, curr = make_review_book(**plants)
    return R.run_review(curr, prev, *months)


@pytest.fixture(scope="module")
def september():
    """WEST chosen in August's review (short on collection), collecting
    normally in September; NORTH and EAST stand out in September."""
    aug = _review(AUG, shortfall={"WEST": 0.80}, risk={"NORTH": 1.6}, seed=1)
    log = DecisionLog()
    c = next(c for c in aug.candidates["collection_shortfall"] if c.region == "WEST")
    log.append(data_version=aug.data_version, run="r", segment="review", period=dict(aug.period),
               option_id=f"investigate:{c.id}", region="WEST", list_name="collection_shortfall", book=aug.book,
               baseline=c.baseline())
    sep = _review(SEP, risk={"NORTH": 1.6}, worsen={"EAST": 4.0}, seed=2)
    outcomes = track_outcomes(sep, log.entries_for_book(sep.book))
    assert [o.status for o in outcomes] == [IMPROVED]
    return sep, outcomes


def _answer(brief, **over):
    opt = next(o["id"] for o in brief.options if o["id"].startswith("investigate:"))
    region = opt.rsplit(":", 1)[1]
    base = {"summary": f"{region} stands out this month. Last month WEST was chosen and has improved.",
            "recommended_option": opt, "why": f"{region} first.", "tradeoff": "", "watch": "Watch WEST next month."}
    return json.dumps({**base, **over})


class TestFacts:
    def test_last_month_is_in_the_facts(self, september):
        sep, outcomes = september
        facts = analyst_facts(sep, outcomes)
        [lm] = facts["last_month"]
        assert lm["region"] == "WEST" and lm["status"] == IMPROVED and lm["text"] == outcomes[0].text

    def test_no_outcomes_no_section(self, september):
        sep, _ = september
        assert "last_month" not in analyst_facts(sep)
        assert "LAST MONTH" not in build_prompt(analyst_facts(sep))
        assert "LAST MONTH" in build_prompt(analyst_facts(sep, september[1]))

    def test_the_outcomes_own_figures_may_be_quoted(self, september):
        sep, outcomes = september
        gap = f"{outcomes[0].then['gap_pts']:.2f}"
        answer = json.loads(_answer(sep, summary=f"Last month WEST had a gap of {gap} points and has improved."))
        assert not any("numbers not written" in p for p in verify(answer, analyst_facts(sep, outcomes)))


class TestDirection:
    def test_the_right_direction_passes(self, september):
        sep, outcomes = september
        assert verify(json.loads(_answer(sep)), analyst_facts(sep, outcomes)) == []

    @pytest.mark.parametrize("claim", ["Last month WEST was chosen and got worse.",
                                       "Last month WEST was chosen and shows no clear change."])
    def test_a_wrong_direction_is_refused(self, september, claim):
        sep, outcomes = september
        problems = verify(json.loads(_answer(sep, summary=claim)), analyst_facts(sep, outcomes))
        assert any("WEST" in p and "improved" in p for p in problems), problems

    def test_a_region_on_this_months_worse_list_is_not_an_outcome_claim(self, september):
        # "30+ getting worse" is a list name, not last month's outcome.
        sep, outcomes = september
        answer = json.loads(_answer(sep, summary="EAST is on the 30+ getting worse list. Last month WEST improved."))
        assert not any("last month" in p.lower() for p in verify(answer, analyst_facts(sep, outcomes)))


class TestListCounts:
    """Live run 1 with outcomes: "NORTH is recommended first as it is on two
    lists ... EAST is on only one list" where each was on one. A count in
    words is a number the number check can't see."""

    @pytest.mark.parametrize("claim", ["NORTH is first as it is on two lists.",
                                       "NORTH is on 2 lists while EAST is on only one list."])
    def test_a_wrong_list_count_is_refused(self, september, claim):
        sep, outcomes = september
        problems = verify(json.loads(_answer(sep, why=claim)), analyst_facts(sep, outcomes))
        assert any("NORTH" in p and "on 1 list" in p for p in problems), problems

    def test_a_right_list_count_passes(self, september):
        sep, outcomes = september
        answer = json.loads(_answer(sep, why="NORTH is on one list, and so is EAST."))
        assert not any("list" in p for p in verify(answer, analyst_facts(sep, outcomes)))


class TestRun:
    def test_the_analysis_records_which_outcomes_it_saw(self, september):
        sep, outcomes = september
        a = write_analysis(sep, lambda prompt: _answer(sep), outcomes=outcomes)
        assert a.status == "ok" and a.outcomes_seen == outcomes_signature(outcomes) != ""
        assert write_analysis(sep, lambda prompt: _answer(sep)).outcomes_seen == ""
