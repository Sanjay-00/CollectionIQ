"""controlled_decompose (the rival test), drill and sample_loans (spec 7.6 to 7.8)."""
import pandas as pd
import pytest

from investigation.eval_cases import CASES, CURR_MONTH, PREV_MONTH
from investigation.evidence import EvidenceStore, ToolError
from investigation.panel import make_pair
from investigation.tools import controlled_decompose, drill, sample_loans


def _pair(case_id):
    prev, curr = next(c for c in CASES if c.id == case_id).build()
    return make_pair(curr, prev, CURR_MONTH, PREV_MONTH)


class TestRivalTest:
    def test_the_real_cause_persists_inside_the_rival(self):
        # Case 10: TRACTOR loans are all recent, so vintage looks guilty too.
        e = controlled_decompose(EvidenceStore(), _pair(10), "product", "TRACTOR", control="vintage")
        assert e.tool == "controlled_decompose" and e.summary["persists"] is True
        assert e.summary["persisting_loans"] >= e.summary["tested_loans"] / 2

    def test_the_confounded_pattern_does_not_persist_inside_the_cause(self):
        e = controlled_decompose(EvidenceStore(), _pair(10), "vintage", "2025-Q3", control="product")
        assert e.summary["persists"] is False
        assert all(lv["evidence_id"] for lv in e.summary["levels"])

    def test_an_unusable_control_is_refused(self):
        prev, curr = CASES[0].build()
        pair = make_pair(curr.drop(columns="FUEL_TYPE"), prev.drop(columns="FUEL_TYPE"), CURR_MONTH, PREV_MONTH)
        with pytest.raises(ToolError, match="Fuel"):
            controlled_decompose(EvidenceStore(), pair, "vintage", "2025-Q1", control="fuel")


class TestDrill:
    def test_drill_finds_the_executive_inside_the_branch(self):
        e = drill(EvidenceStore(), _pair(13), "branch", "BR3")
        first = e.summary["ranked"][0]
        assert e.tool == "drill" and e.summary["parent"] == ["branch", "BR3"]
        assert first["dimension"] == "executive" and first["top"][0]["segment"] == "EXEC7 (BR3)"
        assert first["top"][0]["beyond_share"]
        assert "branch" not in [r["dimension"] for r in e.summary["ranked"]]

    def test_a_segment_too_small_to_split_gives_an_empty_ranking(self):
        e = drill(EvidenceStore(), _pair(13), "branch", "TINYBR")
        assert e.summary["ranked"] == [] or all(r["concentration_pct"] in (None, 0.0) for r in e.summary["ranked"])


def _hand_pair():
    rows = [("L1", "STD", "SMA-1"), ("L2", "1-30 DPD", "NPA"), ("L3", "SMA-1", "STD"), ("L4", "NPA", "NPA"),
            ("L5", "STD", "STD")]
    def month(i):
        return pd.DataFrame({"Loan No": [r[0] for r in rows], "curr_bucket": [r[i] for r in rows],
                             "Loan Status": "RUN", "Unit": "B1", "MNT NAME": "RAJ", "SOH": [5, 4, 3, 2, 1],
                             "Cust Name": "SECRET NAME", "Cust Mob No": "9999999999"})
    return make_pair(month(2), month(1), CURR_MONTH, PREV_MONTH)


class TestSampleLoans:
    def test_kinds_pick_the_right_loans_largest_soh_first(self):
        store = EvidenceStore()
        assert list(sample_loans(store, _hand_pair(), {}, "rolled_in").tables["loans"]["Loan No"]) == ["L1", "L2"]
        assert list(sample_loans(store, _hand_pair(), {}, "cured").tables["loans"]["Loan No"]) == ["L3"]
        assert list(sample_loans(store, _hand_pair(), {}, "thirty_plus_now").tables["loans"]["Loan No"]) == ["L1", "L2", "L4"]

    def test_summary_has_counts_only_and_n_is_capped(self):
        e = sample_loans(EvidenceStore(), _hand_pair(), {}, "thirty_plus_now", n=500)
        assert e.summary == {"kind": "thirty_plus_now", "scope": {}, "count_all": 3, "shown": 3}
        assert "SECRET NAME" not in str(e.summary)
        with pytest.raises(ToolError):
            sample_loans(EvidenceStore(), _hand_pair(), {}, "everything")
