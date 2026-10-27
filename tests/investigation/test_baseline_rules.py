"""Phase 3 review fixes: the baseline's rules one by one, and the report's wording."""
import dataclasses

import numpy as np
import pandas as pd
import pytest

from investigation import baseline as B
from investigation.eval_cases import CASES, CURR_MONTH, PREV_MONTH
from investigation.evaluation import InvestigationResult, score
from investigation.evidence import EvidenceStore, ToolError
from investigation.panel import make_pair
from investigation.report import build_report
from investigation.synthetic import _loans, make_book
from investigation.tools import controlled_decompose, sample_loans


class TestChoosingTheLead:
    LEAD, RIVAL = {"dimension": "vintage"}, {"dimension": "product"}

    def test_a_lead_that_persists_inside_its_rival_stays(self):
        assert B.choose_lead(self.LEAD, self.RIVAL, lambda who: who is self.LEAD) is self.LEAD

    def test_a_lead_that_vanishes_inside_a_rival_that_persists_is_swapped(self):
        # Review: this branch never ran in any case.
        assert B.choose_lead(self.LEAD, self.RIVAL, lambda who: who is self.RIVAL) is self.RIVAL

    def test_neither_persisting_or_untestable_keeps_the_lead(self):
        assert B.choose_lead(self.LEAD, self.RIVAL, lambda who: False) is self.LEAD
        assert B.choose_lead(self.LEAD, self.RIVAL, lambda who: None) is self.LEAD

    def test_nested_geographies_are_not_rivals(self):
        assert not B.are_rivals("branch", "region") and not B.are_rivals("executive", "zone")
        assert B.are_rivals("branch", "product") and B.are_rivals("vintage", "product")


def test_mechanism_of_a_new_segment_is_mix():
    assert B.mechanism_of({"mix_pp": 0.0, "rate_pp": 0.0, "new_pp": 0.4, "gone_pp": 0.0}) == "mix"
    assert B.mechanism_of({"mix_pp": 0.1, "rate_pp": 0.5, "new_pp": 0.0, "gone_pp": 0.0}) == "rate"


def test_deep_drivers_only_when_the_inside_moved_the_same_way():
    assert B.deep_allowed(inner_total=0.9, total=0.7, first={"rate_pp": 0.5})
    assert not B.deep_allowed(inner_total=-0.9, total=0.7, first={"rate_pp": 0.5})
    assert not B.deep_allowed(inner_total=0.9, total=0.7, first={"rate_pp": -0.1})


def test_a_new_segment_as_first_driver_does_not_crash():
    # Review C1: last month's disbursals are NA last month, so the newest
    # vintage month is always new in eligibility terms; drilling it crashed.
    prev, curr = make_book(20000, seed=7, churn=0.06)
    rng = np.random.default_rng(1)
    extra = _loans(rng, 2500, 900000).assign(Ag_Date=pd.Timestamp("2026-08-20"))
    extra["curr_bucket"] = np.where(rng.random(2500) < 0.6, "SMA-1", "STD")
    result, store = B.run_baseline(pd.concat([curr, extra], ignore_index=True), prev, CURR_MONTH, PREV_MONTH)
    assert result.status == "concluded" and result.deep == []
    assert "2026-08" in [d.segment for d in result.drivers] and result.mechanism == "mix"
    build_report(result, store)                                   # no crash on a segment new this month


def _case(case_id, seed=None):
    c = next(c for c in CASES if c.id == case_id)
    if seed is None:
        return c
    plants = {"exec_rate_up": ("BR3", "EXEC7", 40, 400), "churn": 0.06}
    return dataclasses.replace(c, build=lambda: make_book(30000, seed=seed, **plants))


@pytest.mark.parametrize("seed", [2027, 555])
def test_case_13_holds_on_other_seeds(seed):
    # Review I9: region R1 and branch BR3 nearly tie; the finer geography leads.
    case = _case(13, seed)
    prev, curr = case.build()
    result, store = B.run_baseline(curr, prev, CURR_MONTH, PREV_MONTH)
    assert score("13", case.expected, result, store).passed, (result.drivers, result.deep)


def test_rival_test_runs_inside_the_scope_and_is_reported_when_untestable():
    prev, curr = _case(10).build()
    scope = {"zone": "Z1"}
    _, store = B.run_baseline(curr, prev, CURR_MONTH, PREV_MONTH, scope=scope)
    rivals = [e for e in store.all() if e.tool in ("controlled_decompose", "rival_untestable")]
    assert rivals and all(e.args["scope"] == scope for e in rivals)


def test_controlled_decompose_needs_two_levels_and_the_leads_direction():
    pair = make_pair(*reversed(_case(10).build()), CURR_MONTH, PREV_MONTH)
    with pytest.raises(ToolError, match="two"):            # one testable level is no test at all
        controlled_decompose(EvidenceStore(), pair, "branch", "BR3", control="region")
    e = controlled_decompose(EvidenceStore(), pair, "product", "TRACTOR", control="vintage")
    assert e.summary["persists"] and all("rate_change_pp" in lv for lv in e.summary["levels"])


class TestReportWording:
    def test_offsetting_moves_show_no_invented_share(self):
        prev, curr = _case(9).build()
        rep = build_report(*B.run_baseline(curr, prev, CURR_MONTH, PREV_MONTH))
        assert "BR1" in rep["where"] and "% of the change" not in rep["markdown"]

    def test_a_fall_is_described_as_a_fall_without_verdict_words(self):
        prev, curr = make_book(20000, seed=70, churn=0.06, shift={"BR2": -500})
        rep = build_report(*B.run_baseline(curr, prev, CURR_MONTH, PREV_MONTH))
        assert " fell " in rep["headline"] and "worse" not in rep["markdown"].lower() and "BR2" in rep["headline"]

    def test_most_only_when_the_drivers_cover_most_of_the_change(self):
        prev, curr = _case(10).build()
        rep = build_report(*B.run_baseline(curr, prev, CURR_MONTH, PREV_MONTH))
        assert ("Most of it sits in" in rep["headline"]) == (rep["coverage_pct"] >= 60)

    def test_under_the_line_but_beyond_noise_says_so(self):
        store = EvidenceStore()
        store.add("metric_change", {}, {"change_pp": 0.15, "noise_pp": 0.08, "material": False,
                                         "prev": {"rate": 20.0, "numerator": 10, "eligible": 50},
                                         "curr": {"rate": 20.15, "numerator": 10, "eligible": 50},
                                         "numerator_effect_pp": 0.1, "base_effect_pp": 0.05})
        store.add("flow_bridge", {}, {"lines": {k: 0 for k in ("opening", "in_from_std", "in_from_1_30", "in_from_no_emi",
                                                               "new_to_book", "cured_to_std", "cured_to_1_30",
                                                               "became_no_emi", "left_book", "closing")}})
        rep = build_report(InvestigationResult(status="no_material_change"), store)
        assert "under the 0.20 % point line" in rep["headline"] and "within the normal" not in rep["headline"]


def test_left_the_running_book_is_never_cited():
    # Real Aug -> Sept: "Executive: Left the running book" came out as a deep
    # driver; it is the placeholder for closed loans, not an executive.
    from investigation.panel import LEFT_BOOK
    from investigation.tools import mix_rate
    prev, curr = make_book(20000, seed=72, churn=0.06)
    curr = curr.drop(curr.index[(curr["curr_bucket"] == "STD").to_numpy()][:3000])      # good loans close
    e = mix_rate(EvidenceStore(), make_pair(curr, prev, CURR_MONTH, PREV_MONTH), "executive", min_size=1)
    assert LEFT_BOOK in set(e.tables["segments"]["segment"])
    assert all(t["segment"] != LEFT_BOOK for t in e.summary["top"])


def test_sample_size_must_be_positive():
    prev, curr = make_book(2000, seed=71)
    with pytest.raises(ToolError, match="at least 1"):
        sample_loans(EvidenceStore(), make_pair(curr, prev, CURR_MONTH, PREV_MONTH), {}, "thirty_plus_now", n=0)
