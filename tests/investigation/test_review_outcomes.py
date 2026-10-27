"""Outcome tracking (spec 7.4, build step 5): next month's review says
whether the regions chosen last month improved beyond normal variation.
Two consecutive monthly reviews of the same synthetic book."""
import pytest

from investigation.review import runner as R
from investigation.review.decisions import DecisionLog
from investigation.review.outcomes import IMPROVED, NOT_COMPARABLE, NO_CLEAR_CHANGE, WORSE, track_outcomes
from investigation.review.synthetic import make_review_book

AUG = ("2026-08", "2026-07")         # (this month, last month) of the review the choice was made on
SEP = ("2026-09", "2026-08")


def _review(months, **plants):
    prev, curr = make_review_book(**plants)
    return R.run_review(curr, prev, *months)


def _choose(log, brief, list_name, region, run="r", **extra):
    """What the panel logs when a leader picks a region from a review."""
    c = next(c for c in brief.candidates[list_name] if c.region == region)
    log.append(data_version=brief.data_version, run=run, segment="review", period=dict(brief.period),
               option_id=f"investigate:{c.id}", region=region, list_name=list_name, book=brief.book,
               baseline=c.baseline(), **extra)


@pytest.fixture(scope="module")
def august():
    return _review(AUG, shortfall={"WEST": 0.80}, risk={"NORTH": 1.6}, seed=1)


class TestOutcomes:
    def test_a_region_that_recovered_is_improved(self, august):
        log = DecisionLog()
        _choose(log, august, "collection_shortfall", "WEST")
        september = _review(SEP, risk={"NORTH": 1.6}, seed=2)          # WEST collects normally again
        [o] = track_outcomes(september, log.entries_for_book(september.book))
        assert (o.region, o.list_name, o.status) == ("WEST", "collection_shortfall", IMPROVED)
        assert o.then["gap_pts"] > 5 and abs(o.now.gap_pts) < o.then["gap_pts"] and o.change_pts < -o.band_pts
        assert "Aug 2026" in o.text and "Sep 2026" in o.text and "improved beyond normal" in o.text

    def test_a_region_still_short_is_mostly_no_clear_change(self, august):
        # Same shortfall both months: only chance moves the gap, so a 2-error
        # band reads "no clear change" about 19 times in 20. One fixed seed
        # (2) is a 2.2-error draw; the error itself is calibrated (actual
        # spread over 30 seeds within 10% of the reported error).
        log = DecisionLog()
        _choose(log, august, "collection_shortfall", "WEST")
        statuses = [track_outcomes(s, log.entries_for_book(s.book))[0].status
                    for s in (_review(SEP, shortfall={"WEST": 0.80}, risk={"NORTH": 1.6}, seed=10 + k)
                              for k in range(10))]
        assert statuses.count(NO_CLEAR_CHANGE) >= 8, statuses

    def test_a_region_that_got_worse_is_worse(self, august):
        log = DecisionLog()
        _choose(log, august, "collection_shortfall", "WEST")
        september = _review(SEP, shortfall={"WEST": 0.55}, risk={"NORTH": 1.6}, seed=2)
        [o] = track_outcomes(september, log.entries_for_book(september.book))
        assert o.status == WORSE

    def test_getting_worse_is_judged_on_this_months_change(self):
        # "30+ getting worse" is a change, so next month asks whether the
        # region now moves better than the rest, not whether a level fell.
        log = DecisionLog()
        aug = _review(AUG, worsen={"EAST": 5.0}, seed=1)
        _choose(log, aug, "worsening", "EAST")
        sep = _review(SEP, seed=2)                                       # no rise this month
        [o] = track_outcomes(sep, log.entries_for_book(sep.book))
        assert o.status in (NO_CLEAR_CHANGE, IMPROVED) and "moved" in o.text

    def test_only_last_months_choices_on_the_same_book_count(self, august):
        log = DecisionLog()
        _choose(log, august, "collection_shortfall", "WEST")
        other_book = _review(AUG, regions=("A1", "B1", "C1"), seed=1)
        assert other_book.book != august.book
        older = _review(("2026-07", "2026-06"), shortfall={"WEST": 0.80}, seed=3)
        _choose(log, older, "collection_shortfall", "WEST")              # decided two months back
        september = _review(SEP, seed=2)
        outcomes = track_outcomes(september, log.entries_for_book(september.book))
        assert [(o.region, o.then["month"]) for o in outcomes] == [("WEST", "2026-08")]

    def test_a_region_too_small_now_is_not_comparable(self, august):
        log = DecisionLog()
        _choose(log, august, "collection_shortfall", "WEST")
        september = _review(SEP, sizes={"WEST": 60}, seed=2)
        [o] = track_outcomes(september, log.entries_for_book(september.book))
        assert o.status == NOT_COMPARABLE and o.now is not None and "too small" in o.text

    def test_handled_is_shown_and_choosing_twice_is_one_outcome(self, august):
        log = DecisionLog()
        _choose(log, august, "collection_shortfall", "WEST")
        _choose(log, august, "collection_shortfall", "WEST", run="another run")    # chose it again later
        log.append(data_version=august.data_version, run="r", segment="close", period=dict(august.period),
                   option_id="close:collection_shortfall:WEST", region="WEST", list_name="collection_shortfall",
                   book=august.book)
        september = _review(SEP, seed=2)
        [o] = track_outcomes(september, log.entries_for_book(september.book))
        assert o.handled and "marked handled" in o.text

    def test_wording_never_claims_the_choice_caused_it(self, august):
        log = DecisionLog()
        _choose(log, august, "collection_shortfall", "WEST")
        september = _review(SEP, seed=2)
        [o] = track_outcomes(september, log.entries_for_book(september.book))
        assert not any(w in o.text.lower() for w in ("because", "due to", "thanks to", "as a result", "caused"))


class TestBaselineInTheLog:
    def test_the_figures_at_decision_time_survive_a_restart(self, august, tmp_path):
        path = tmp_path / "log.jsonl"
        _choose(DecisionLog(path), august, "collection_shortfall", "WEST")
        [d] = DecisionLog(path).entries_for_book(august.book)
        assert d.book == august.book and d.baseline["month"] == "2026-08"
        assert set(d.baseline) >= {"gap_pts", "se_pts", "value", "benchmark", "impact", "unit", "month"}

    def test_an_uploaded_baseline_must_be_plain_numbers(self):
        import json
        bad = {"data_version": "v", "run": "r", "segment": "review", "option_id": "o", "book": "b",
               "baseline": {"gap_pts": "DROP TABLE", "rows": [1, 2]}}
        assert DecisionLog().import_jsonl(json.dumps(bad).encode()) == (0, 1)

    def test_every_candidate_carries_its_error_in_points(self, august):
        for cands in august.candidates.values():
            for c in cands:
                assert c.se_pts is not None and c.se_pts >= 0, c.id
