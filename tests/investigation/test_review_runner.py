"""The background review (spec 7.1): deep-dives, the template brief, and the
runner. Synthetic books only."""
import re
import time

import pytest

from investigation.review import runner as R
from investigation.review.brief import LIST_NAMES
from investigation.review.prioritise import EXCESS, SHORTFALL, WORSE
from investigation.review.synthetic import REVIEW_CURR_MONTH as CURR, REVIEW_PREV_MONTH as PREV, make_review_book
from utils import fmt_value

PLANTED = dict(shortfall={"WEST": 0.85}, risk={"NORTH": 1.6}, worsen={"EAST": 4.0})


@pytest.fixture(scope="module")
def planted():
    prev, curr = make_review_book(**PLANTED)
    return R.run_review(curr, prev, CURR, PREV)


def _wait(job, timeout=120):
    end = time.time() + timeout
    while job.status == "running" and time.time() < end:
        time.sleep(0.05)
    return job


class TestBrief:
    def test_headline_names_every_region_that_stands_out(self, planted):
        assert planted.status == "ok"
        assert planted.headline.startswith("Sep 2026 vs Aug 2026, 6 regions reviewed:")
        for region in ("WEST", "NORTH", "EAST"):
            assert region in planted.headline

    def test_one_finding_per_listed_candidate_with_its_impact(self, planted):
        listed = {name: {f.region for f in planted.findings if f.list == name} for name in (SHORTFALL, EXCESS, WORSE)}
        # EAST's planted rise lifts its 30+ level too, so it is (rightly) on the excess list as well.
        assert "WEST" in listed[SHORTFALL] and "NORTH" in listed[EXCESS] and "EAST" in listed[WORSE]
        for f in planted.findings:
            cand = next(c for c in planted.candidates[f.list] if c.id == f.candidate_id)
            shown = fmt_value(cand.impact["amount"], "money") if cand.impact["unit"] == "Rs" \
                else f"{cand.impact['amount']:,.0f}"
            assert shown in f.text, (f.list, shown, f.text)
            assert all(planted.store.get(e) for e in f.evidence_ids)

    def test_deep_dives_say_where_and_their_parts_reconcile(self, planted):
        dives = {d.list: d for d in planted.deep_dives}
        assert set(dives) == {SHORTFALL, EXCESS, WORSE}
        for name in (SHORTFALL, EXCESS):
            d = dives[name]
            assert planted.store.get(d.evidence_id).checks_passed       # parts add up to the region's figure
            assert d.parts and d.branches and all(b.startswith(d.region[:3]) for b, _ in d.branches)
        assert dives[WORSE].investigation is not None
        assert dives[WORSE].investigation.status in ("concluded", "inconclusive", "no_material_change")
        # Every worsening finding says where or how inside the region, never nothing (live: NASHIK, sample data).
        worse = next(f for f in planted.findings if f.list == WORSE)
        assert f"Inside {worse.region}" in worse.text

    def test_options_offer_each_finding_then_scorecard_and_stop(self, planted):
        ids = [o["id"] for o in planted.options]
        assert ids[-2:] == ["scorecard", "stop"]
        # Only the regions looked into get a button (a pro brief offers a few next steps, not every row).
        assert {i for i in ids if i.startswith("investigate:")} == {f"investigate:{d.candidate_id}"
                                                                    for d in planted.deep_dives}

    def test_wording_is_descriptive_plain_and_dash_free(self, planted):
        text = " ".join([planted.headline, *(f.text for f in planted.findings), *planted.overlaps,
                         *planted.data_notes, *planted.caveats, *(o["label"] for o in planted.options)])
        assert not re.search(r"\b(caus\w*|due to|because|driven by|leads? to|results? in)\b", text, re.I)
        assert "—" not in text and "–" not in text

    def test_pure_noise_says_nothing_stands_out(self):
        prev, curr = make_review_book(seed=3)
        brief = R.run_review(curr, prev, CURR, PREV)
        assert "none stands out" in brief.headline and not brief.findings
        assert [o["id"] for o in brief.options] == ["scorecard", "stop"]

    def test_a_data_halt_is_the_answer(self):
        prev, curr = make_review_book()
        brief = R.run_review(curr, prev.sample(frac=0.5, random_state=1), CURR, PREV)
        assert brief.status == "halted_data"
        assert brief.headline.startswith("The two files can't be compared yet") and brief.data_notes

    def test_the_apps_carried_columns_are_not_a_data_problem(self):
        # The app hands over this month with last month's figures carried on
        # each loan (prev_bucket, prev_SOH, ...). They aren't columns of the
        # file, so the data check must not report them as "in one month's
        # file only" (found in an end-to-end trace on the sample data).
        prev, curr = make_review_book()
        joined = curr.merge(prev[["Loan No", "curr_bucket", "SOH"]].rename(
            columns={"curr_bucket": "prev_bucket", "SOH": "prev_SOH"}), on="Loan No", how="left")
        brief = R.run_review(joined, prev, CURR, PREV)
        assert not any("prev_" in n for n in brief.data_notes), brief.data_notes

    def test_one_region_in_view_has_nothing_to_compare_with(self):
        # A sidebar region filter leaves one region: every list compares a
        # region with the rest of the portfolio, so there is no review to run.
        prev, curr = make_review_book()
        brief = R.run_review(curr[curr["RegionName"] == "WEST"], prev[prev["RegionName"] == "WEST"], CURR, PREV)
        assert brief.status == "too_narrow" and "WEST" in brief.headline and not brief.findings

    def test_no_customer_field_reaches_the_brief(self):
        prev, curr = make_review_book(**PLANTED)
        curr = curr.assign(**{"Cust Name": "SECRET PERSON", "Cust Mob No": "9999912345"})
        brief = R.run_review(curr, prev, CURR, PREV)
        stores = [brief.store] + [d.investigation_store for d in brief.deep_dives if d.investigation_store]
        text = repr(brief.findings) + repr(brief.deep_dives) + "".join(
            repr(e.summary) + "".join(t.to_csv() for t in e.tables.values()) for s in stores for e in s.all())
        assert "SECRET PERSON" not in text and "9999912345" not in text


class TestSharedWork:
    def test_the_months_are_paired_once_per_review(self, monkeypatch):
        # Every step reuses the runner's pair: pairing again (make_pair in the
        # review or a deep-dive) would redo the whole book's work per region.
        def refuse(*a, **k):
            raise AssertionError("paired again")
        monkeypatch.setattr("investigation.review.prioritise.make_pair", refuse)
        monkeypatch.setattr("investigation.baseline.make_pair", refuse)
        prev, curr = make_review_book(**PLANTED)
        assert R.run_review(curr, prev, CURR, PREV).findings

    def test_progress_reports_each_step_in_order(self):
        prev, curr = make_review_book(**PLANTED)
        steps = []
        R.run_review(curr, prev, CURR, PREV, progress=steps.append)
        assert steps[0].startswith("Paired the two months") and steps[1].startswith("Checked the data")
        assert steps[2] == "Scored 6 regions on 3 lists" and steps[-1] == "Wrote the brief"
        assert [s for s in steps if s.startswith("Looked inside")] == [
            f"Looked inside WEST ({LIST_NAMES[SHORTFALL]})", f"Looked inside NORTH ({LIST_NAMES[EXCESS]})",
            f"Looked inside EAST ({LIST_NAMES[WORSE]})"]


class TestRunner:
    def test_same_key_is_one_run(self):
        prev, curr = make_review_book(**PLANTED)
        runner = R.ReviewRunner()
        a = runner.submit("v1", curr, prev, CURR, PREV)
        b = runner.submit("v1", curr, prev, CURR, PREV)
        assert a is b
        _wait(a)
        assert a.status == "done" and a.brief.findings and a.finished is not None
        assert [t for t, _ in a.steps][-1] == "Wrote the brief"

    def test_a_failure_is_reported_then_retried(self):
        prev, curr = make_review_book()
        runner = R.ReviewRunner()
        bad = _wait(runner.submit("v1", curr.drop(columns=["curr_bucket"]), prev, CURR, PREV))
        assert bad.status == "failed" and "curr_bucket" in bad.error
        again = _wait(runner.submit("v1", curr, prev, CURR, PREV))
        assert again is not bad and again.status == "done"

    def test_only_the_last_jobs_are_kept(self):
        prev, curr = make_review_book(n_per_region=400, regions=("A1", "B1"))
        runner = R.ReviewRunner(max_jobs=2)
        for key in ("k1", "k2", "k3"):
            _wait(runner.submit(key, curr, prev, CURR, PREV))
        assert runner.get("k1") is None and runner.get("k2") and runner.get("k3")
