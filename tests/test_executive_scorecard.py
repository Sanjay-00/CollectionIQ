import pandas as pd

from analysis.executive_scorecard import compute_executive_scorecard, rank_by_metric
from helpers import make_df


def _exec_df():
    rows = []
    # RAJ: high collection, low strike (5 accounts, min_accounts default is 5)
    for i in range(5):
        rows.append({
            "MNT NAME": "RAJ", "Unit": "MAHAD", "Strike": "N",
            "Net Collection Demand Inst+Exp+BC": 10_000.0,
            "Month Collection (Excluding Reserve Collection)": 10_000.0,
        })
    # SUNIL: low collection, high strike (5 accounts)
    for i in range(5):
        rows.append({
            "MNT NAME": "SUNIL", "Unit": "MAHAD", "Strike": "Y",
            "Net Collection Demand Inst+Exp+BC": 10_000.0,
            "Month Collection (Excluding Reserve Collection)": 2_000.0,
        })
    return make_df(rows)


class TestComputeExecutiveScorecard:
    def test_ranked_by_collection_by_default(self):
        sc = compute_executive_scorecard(_exec_df(), min_accounts=5)
        assert list(sc["Executive (Branch)"].str.split(" (", regex=False).str[0]) == ["RAJ", "SUNIL"]
        assert sc.iloc[0]["Collection %"] == 100.0
        assert sc.iloc[1]["Strike Rate %"] == 100.0


class TestStrikeRateNormalization:
    def test_lowercase_and_padded_strike_values_are_still_counted(self):
        # Regression: compute_executive_scorecard used to filter the Strike
        # denominator with a raw .isin(["Y", "N"]) - no case/whitespace
        # normalization - while the numerator (is_yes) did normalize. Lowercase
        # or padded "y"/"n" values were silently dropped from the denominator,
        # undercounting valid accounts. Now routed through utils.compute_strike_pct,
        # which normalizes both sides consistently.
        rows = (
            [{"MNT NAME": "RAJ", "Unit": "MAHAD", "Strike": "y"}] * 3
            + [{"MNT NAME": "RAJ", "Unit": "MAHAD", "Strike": " n "}] * 2
        )
        sc = compute_executive_scorecard(make_df(rows), min_accounts=5)
        assert len(sc) == 1
        # 3 of 5 valid (normalized) rows are Y -> 60.0%, not excluded to 0/0.
        assert sc.iloc[0]["Strike Rate %"] == 60.0


class TestRollRateNoPriorMatchIsNoneNotZero:
    """Regression: a newly appointed executive (or one whose whole book is freshly
    originated this month) has zero Loan No overlap with the previous file -- there
    is no basis to compute a roll rate at all. This used to fabricate 0.0% for both
    Roll Fwd % and Roll Bwd %, which reads as "verified: nothing got worse" -- a
    different, false claim from the true state "not enough history to say".
    analysis/portfolio_intelligence.py::_roll_rates() already returns None for the
    identical situation at the region/branch grain; this brings the executive
    scorecard's own inline roll-rate calc back in line with it."""

    def _new_exec_df(self):
        rows = [{
            "MNT NAME": "NEW EXEC", "Unit": "MAHAD", "Strike": "N",
            "curr_bucket": "NPA", "prev_bucket": None,
        } for _ in range(5)]
        return make_df(rows)

    def test_no_prior_match_gives_none_not_zero(self):
        sc = compute_executive_scorecard(self._new_exec_df(), min_accounts=5)
        assert sc.iloc[0]["Roll Fwd %"] is None
        assert sc.iloc[0]["Roll Bwd %"] is None

    def test_partial_prior_match_still_computes_a_real_ratio(self):
        # 3 loans have a real prev_bucket match (2 worsened, 1 stable), 2 are brand
        # new (prev_bucket None) -- total_valid must be 3, not 5, and the 2 new
        # loans must not be silently treated as "no change" (which would understate
        # Roll Fwd%) or excluded from the executive entirely.
        rows = (
            [{"MNT NAME": "EXEC", "Unit": "MAHAD", "curr_bucket": "NPA", "prev_bucket": "SMA-2"}] * 2
            + [{"MNT NAME": "EXEC", "Unit": "MAHAD", "curr_bucket": "STD", "prev_bucket": "STD"}]
            + [{"MNT NAME": "EXEC", "Unit": "MAHAD", "curr_bucket": "NPA", "prev_bucket": None}] * 2
        )
        sc = compute_executive_scorecard(make_df(rows), min_accounts=5)
        assert sc.iloc[0]["Roll Fwd %"] == round(2 / 3 * 100, 1)

    def test_na_prev_bucket_is_not_comparable_either(self):
        # Regression: this is DIFFERENT from "no prior match" (prev_bucket is
        # None/NaN, a true merge miss). Here prev_bucket is the literal string
        # "NA" -- a real match against a loan whose prior Arrears/EMI was itself
        # missing/unparseable. That used to score as -1 (lower than every real
        # bucket), so "NA" -> STD (the healthiest real bucket) was miscounted as
        # a 100% roll-forward -- confirmed on real production data, not
        # hypothetical. Must be excluded from the ratio exactly like a true NaN.
        rows = [{
            "MNT NAME": "EXEC4", "Unit": "MAHAD",
            "curr_bucket": "STD", "prev_bucket": "NA",
        } for _ in range(5)]
        sc = compute_executive_scorecard(make_df(rows), min_accounts=5)
        assert sc.iloc[0]["Roll Fwd %"] is None
        assert sc.iloc[0]["Roll Bwd %"] is None


class TestRankByMetric:
    def test_reranks_by_strike_rate_independently_of_collection_rank(self):
        sc = compute_executive_scorecard(_exec_df(), min_accounts=5)
        # Sanity: default order is collection-ranked (RAJ first).
        assert sc.iloc[0]["Executive (Branch)"].startswith("RAJ")

        by_strike = rank_by_metric(sc, "Strike Rate %")
        # Re-ranked order flips: SUNIL (100% strike) now leads.
        assert by_strike.iloc[0]["Executive (Branch)"].startswith("SUNIL")
        assert by_strike.iloc[0]["Tier"] == "top"
        # Original collection-ranked df is untouched.
        assert sc.iloc[0]["Executive (Branch)"].startswith("RAJ")

    def test_empty_df_returns_as_is(self):
        empty = pd.DataFrame()
        assert rank_by_metric(empty, "Strike Rate %").empty

    def test_missing_column_returns_as_is(self):
        sc = compute_executive_scorecard(_exec_df(), min_accounts=5)
        result = rank_by_metric(sc, "No Such Column")
        assert result is sc
