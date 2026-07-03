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
