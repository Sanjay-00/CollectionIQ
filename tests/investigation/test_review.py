"""Portfolio Review, Phase 1 (spec 2026-10-24 sections 5-6, 8): the
scorecard, the three lists, and the evaluation cases. Synthetic books only."""
import math

import pandas as pd
import pytest

from analysis.portfolio_intelligence import compute_region_scorecard
from investigation.review import review_regions
from investigation.review.prioritise import excess_table, shortfall_table
from investigation.review.synthetic import REVIEW_CURR_MONTH as CURR, REVIEW_PREV_MONTH as PREV, make_review_book

SHORTFALL, EXCESS, WORSE = "collection_shortfall", "excess_30plus", "worsening"


def _review(**plants):
    prev, curr = make_review_book(**plants)
    return review_regions(curr, prev, CURR, PREV)


def _listed(res, name):
    return [c.region for c in res.lists[name]]


def _raw_top(res, name):
    """The region a ranking without standardising would put first."""
    return max(res.assessed[name], key=lambda c: c.raw_impact).region


# ── Formulas, checked by hand ────────────────────────────────────────────────

class TestShortfallFormula:
    def test_expected_collection_uses_the_rest_of_the_book_per_stratum(self):
        # A: 100 demand in STD (collected 50), 100 in NPA (collected 0).
        # Rest (B): STD rate 0.9, NPA rate 0.5.
        df = pd.DataFrame({"g": ["A", "A", "B", "B"], "s": ["STD", "NPA", "STD", "NPA"],
                           "d": [100.0, 100.0, 1000.0, 1000.0], "c": [50.0, 0.0, 900.0, 500.0]})
        t = shortfall_table(df, "g", "s", "d", "c").set_index("region")
        a = t.loc["A"]
        assert a["expected"] == pytest.approx(100 * 0.9 + 100 * 0.5)       # 140
        assert a["shortfall"] == pytest.approx(140 - 50)
        assert a["raw_shortfall"] == pytest.approx(200 * (1400 / 2000) - 50)  # 90: same mix as the rest
        assert a["mix_part"] == pytest.approx(a["raw_shortfall"] - a["shortfall"])
        assert a["gap_pts"] == pytest.approx(100 * 90 / 200)

    def test_book_mix_alone_is_not_a_shortfall(self):
        # A collects at exactly the rest's rate in each stratum, but holds
        # mostly NPA: its raw rate is low, its standardised gap is zero.
        df = pd.DataFrame({"g": ["A", "A", "B", "B"], "s": ["STD", "NPA", "STD", "NPA"],
                           "d": [100.0, 900.0, 900.0, 100.0], "c": [90.0, 450.0, 810.0, 50.0]})
        t = shortfall_table(df, "g", "s", "d", "c").set_index("region")
        assert t.loc["A", "shortfall"] == pytest.approx(0.0)
        assert t.loc["A", "raw_shortfall"] > 0           # what an unstandardised ranking would report
        assert t.loc["A", "mix_part"] == pytest.approx(t.loc["A", "raw_shortfall"])

    def test_a_stratum_the_rest_lacks_falls_back_and_says_so(self):
        df = pd.DataFrame({"g": ["A", "A", "B"], "s": ["STD", "NEW", "STD"],
                           "d": [100.0, 100.0, 100.0], "c": [100.0, 100.0, 80.0]})
        t = shortfall_table(df, "g", "s", "d", "c").set_index("region")
        assert t.loc["A", "fallback_strata"] == ["NEW"]
        assert t.loc["A", "expected"] == pytest.approx(100 * 0.8 + 100 * 0.8)


class TestExcessFormula:
    def test_expected_30_plus_per_stratum_in_accounts_and_soh(self):
        # A: vintage V1 2 loans (1 is 30+), V2 2 loans (2 are 30+).
        # Rest: V1 rate 0.25 (1 of 4), V2 rate 0.5 (2 of 4); SOH 100 per loan.
        df = pd.DataFrame({
            "g": ["A"] * 4 + ["B"] * 8, "s": ["V1", "V1", "V2", "V2"] + ["V1"] * 4 + ["V2"] * 4,
            "y": [1, 0, 1, 1] + [1, 0, 0, 0] + [1, 1, 0, 0], "h": [100.0] * 12})
        t = excess_table(df, "g", "s", "y", "h").set_index("region")
        a = t.loc["A"]
        assert a["expected_loans"] == pytest.approx(2 * 0.25 + 2 * 0.5)     # 1.5
        assert a["excess_loans"] == pytest.approx(3 - 1.5)
        assert a["excess_soh"] == pytest.approx(300 - (200 * 0.25 + 200 * 0.5))
        assert a["raw_excess_loans"] == pytest.approx(3 - 4 * (3 / 8))


# ── Evaluation cases (spec section 6) ────────────────────────────────────────

class TestEvalCases:
    def test_1_bucket_mix_trap_is_not_a_shortfall(self):
        res = _review(mix={"EAST": {"NPA": 0.45, "SMA-2": 0.2, "SMA-1": 0.1, "STD": 0.25}})
        assert _raw_top(res, SHORTFALL) == "EAST"        # raw ranking falls for it
        assert "EAST" not in _listed(res, SHORTFALL)

    def test_2_real_shortfall_is_listed_first(self):
        res = _review(shortfall={"WEST": 0.85})
        listed = res.lists[SHORTFALL]
        assert listed and listed[0].region == "WEST"
        top = listed[0]
        assert top.impact["unit"] == "Rs" and top.impact["amount"] > 0 and top.material

    def test_3_seasoning_trap_is_not_excess_risk(self):
        res = _review(old_book={"SOUTH": 3.0})
        assert _raw_top(res, EXCESS) == "SOUTH"
        assert "SOUTH" not in _listed(res, EXCESS)
        # Most of SOUTH's loans are older than any other region's: it says it
        # can't benchmark them, instead of comparing them with a younger book.
        south = next(c for c in res.assessed[EXCESS] if c.region == "SOUTH")
        assert any("like-for-like" in r for r in south.reasons_not_listed)

    def test_3b_real_excess_risk_is_listed(self):
        res = _review(risk={"NORTH": 1.6})
        assert _listed(res, EXCESS)[:1] == ["NORTH"]

    def test_4_small_region_is_shown_never_ranked(self):
        res = _review(sizes={"COAST": 80}, risk={"COAST": 3.0}, shortfall={"COAST": 0.3})
        assert "COAST" in res.too_small
        assert all("COAST" not in _listed(res, name) for name in res.lists)
        assert "COAST" in set(res.scorecard["Region"])

    def test_5_pure_noise_rarely_lists_anything(self):
        # Every region behaves the same: a listing is a false positive. The
        # floors plus Benjamini-Hochberg keep that rare (spec 5.5).
        noisy = sum(any(review.lists.values())
                    for review in (_review(seed=s) for s in range(20)))
        assert noisy <= 2

    def test_6_a_region_on_two_lists_is_flagged(self):
        res = _review(shortfall={"WEST": 0.85}, worsen={"WEST": 4.0})
        assert {SHORTFALL, WORSE} <= set(res.on_lists.get("WEST", []))

    def test_7_same_input_same_answer_any_row_order(self):
        prev, curr = make_review_book(shortfall={"WEST": 0.85}, risk={"NORTH": 1.6})
        a = review_regions(curr, prev, CURR, PREV)
        b = review_regions(curr, prev, CURR, PREV)
        c = review_regions(curr.sample(frac=1, random_state=1), prev.sample(frac=1, random_state=2), CURR, PREV)
        key = lambda r: {n: [(x.region, round(x.impact["amount"], 2)) for x in v] for n, v in r.lists.items()}
        assert key(a) == key(b) == key(c)
        assert a.data_version == c.data_version


# ── Scorecard and evidence ───────────────────────────────────────────────────

class TestScorecard:
    def test_dashboard_columns_equal_the_portfolio_intelligence_scorecard(self):
        prev, curr = make_review_book(shortfall={"WEST": 0.85})
        res = review_regions(curr, prev, CURR, PREV)
        pi = compute_region_scorecard(res.curr_joined, prev).set_index("Region")
        sc = res.scorecard.set_index("Region")
        for col in ("Accounts", "Collection%", "NPA%", "Delinquency%", "Strike%", "SOH (Cr)", "Δ NPA%"):
            pd.testing.assert_series_equal(sc[col].sort_index(), pi[col].sort_index(), check_names=False,
                                           check_dtype=False)

    def test_30_plus_column_matches_the_investigation_metric(self):
        from investigation.metrics import get_metric
        from investigation.tools.change import side_counts
        prev, curr = make_review_book()
        res = review_regions(curr, prev, CURR, PREV)
        m = get_metric("thirty_plus_rate")
        north = side_counts(curr[curr["RegionName"] == "NORTH"], m)
        assert res.scorecard.set_index("Region").loc["NORTH", "30+%"] == pytest.approx(round(north["rate"], 2))


class TestEvidence:
    def test_every_candidate_is_backed_and_its_checks_pass(self):
        res = _review(shortfall={"WEST": 0.85}, risk={"NORTH": 1.6})
        for name, cands in res.assessed.items():
            for c in cands:
                ev = res.store.get(c.evidence_id)
                assert ev.checks_passed, (name, c.region, ev.checks)
                assert c.data_version == res.data_version
                assert c.formula and c.metric and c.period == {"prev": PREV, "curr": CURR}

    def test_no_customer_field_reaches_a_candidate_or_evidence(self):
        prev, curr = make_review_book(shortfall={"WEST": 0.85})
        curr = curr.assign(**{"Cust Name": "SECRET PERSON", "Cust Mob No": "9999912345"})
        res = review_regions(curr, prev, CURR, PREV)
        text = repr(res.lists) + repr(res.assessed) + "".join(
            repr(e.summary) + "".join(t.to_csv() for t in e.tables.values()) for e in res.store.all())
        assert "SECRET PERSON" not in text and "9999912345" not in text

    def test_collection_from_loans_with_no_emi_due_is_reconciled_not_dropped(self):
        # Dummy file: matured loans with no demand still collect, and the
        # Dashboard counts it (a region at 139.99%). The shortfall leaves it
        # out, says so, and the evidence still reconciles to the scorecard.
        prev, curr = make_review_book(shortfall={"WEST": 0.85})
        idle = curr.index[curr["RegionName"] == "NORTH"][:300]
        curr.loc[idle, "Net Collection Demand Inst+Exp+BC"] = 0.0
        curr.loc[idle, "Month Collection (Excluding Reserve Collection)"] = 50_000.0
        res = review_regions(curr, prev, CURR, PREV)
        north = next(c for c in res.assessed[SHORTFALL] if c.region == "NORTH")
        assert res.store.get(north.evidence_id).checks_passed
        assert any("no EMI due this month" in x for x in north.caveats)
        assert res.lists[SHORTFALL][0].region == "WEST"

    def test_bh_q_values_are_monotone_with_z(self):
        res = _review(shortfall={"WEST": 0.85, "EAST": 0.93})
        cands = sorted(res.assessed[SHORTFALL], key=lambda c: -c.z)
        qs = [c.q_value for c in cands]
        assert qs == sorted(qs) and all(0 <= q <= 1 for q in qs if not math.isnan(q))
