"""Guards the Strike %/Hard Bucket % duplication called out in CLAUDE.md: the
dashboard/Portfolio Intelligence path (utils.compute_strike_pct / compute_hard_bucket_pct,
a single shared helper as of this file) and the AI Query path (registry/ontology.py's
strike_pct/hard_bucket_pct count_ratio METRICs, compiled and executed independently via
compiler/core.py + agents/plan_executor.py) are two separate implementations of the same
business definition with no shared code path between them. If either drifts, this test
fails instead of the two silently disagreeing in production.
"""
import pandas as pd

from compiler.core import compile_logical
from agents.plan_executor import execute_plan
from utils import compute_strike_pct, compute_hard_bucket_pct, compute_delinquency_pct, compute_metrics


def _df():
    return pd.DataFrame({
        "Loan No": [f"L{i}" for i in range(10)],
        "Unit": ["A"] * 10,
        "Arrears / EMI": [7, 8, 0, 1, 2, 6, 6, 0, 0, 0],
        "Strike": ["Y", "Y", "Y", "N", "N", "Y", "N", "N", "N", "N"],
    })


def _via_compiler(metric: str, df: pd.DataFrame) -> float:
    # "dimensions": ["branch"] with a single Unit value gives a one-row portfolio-level
    # result -- compile_logical requires at least one filter or dimension.
    cols = list(df.columns)
    ir1 = {
        "intent": "aggregation", "view": None, "filters": [], "dimensions": ["branch"],
        "measures": [{"metric": metric, "alias": "m"}],
        "metrics": [], "having": [], "order_by": [], "limit": None,
        "display_columns": [], "time": None,
    }
    plan, errs = compile_logical(ir1, cols)
    assert errs == [], errs
    result, err = execute_plan(df, plan)
    assert err == "", err
    return float(result["m"].iloc[0])


class TestHardBucketPctConsistency:
    def test_ai_query_path_matches_shared_helper(self):
        df = _df()
        assert _via_compiler("hard_bucket_pct", df) == compute_hard_bucket_pct(df)

    def test_still_matches_on_a_different_distribution(self):
        df = _df()
        df["Arrears / EMI"] = [0, 0, 0, 6, 6, 6, 6, 9, 9, 12]
        assert _via_compiler("hard_bucket_pct", df) == compute_hard_bucket_pct(df)


class TestDelinquencyPctConsistency:
    def test_ai_query_path_matches_shared_helper(self):
        df = _df()
        assert _via_compiler("delinquency_pct", df) == compute_delinquency_pct(df)

    def test_still_matches_on_a_different_distribution(self):
        df = _df()
        df["Arrears / EMI"] = [0, 0, 0, 0.5, 1, 2, 3, 0, 0, 0]
        assert _via_compiler("delinquency_pct", df) == compute_delinquency_pct(df)


class TestStrikePctConsistency:
    def test_ai_query_path_matches_shared_helper(self):
        df = _df()
        assert _via_compiler("strike_pct", df) == compute_strike_pct(df)

    def test_still_matches_with_invalid_strike_values_present(self):
        # compute_strike_pct excludes rows where Strike isn't Y/N from the denominator;
        # the compiler's denominator_where (Strike in [Y, N, YES, NO]) must do the same.
        df = _df()
        df.loc[0, "Strike"] = ""
        df.loc[1, "Strike"] = "MAYBE"
        assert _via_compiler("strike_pct", df) == compute_strike_pct(df)

    def test_still_matches_when_strike_is_spelled_out_yes_no(self):
        # Regression: a real production file had every Strike value spelled out
        # as "YES"/"NO" (zero exact "Y"/"N"). Both the shared helper's denominator
        # and the compiler's denominator_where/numerator_where used to gate
        # strictly on {"Y", "N"} only, so both independently zeroed out on such a
        # file -- a same-wrong-answer bug that this test wouldn't have caught
        # before, since it only guarded that they *agree*, not that they're
        # *correct*. Now both accept the spelled-out form and must still agree.
        df = _df()
        df["Strike"] = ["YES", "YES", "YES", "NO", "NO", "YES", "NO", "NO", "NO", "NO"]
        result = _via_compiler("strike_pct", df)
        assert result == compute_strike_pct(df) == 40.0


class TestLccPctConsistency:
    """Guards the LCC% drift found in production: the dashboard (utils.compute_metrics)
    used to sum "Total Cum Collection" (a broader figure) instead of the documented
    "Cum Coll (Inst+Exp)" numerator, and never capped at 100 -- both silently
    diverging from the AI Query path's lcc_pct METRIC (registry/ontology.py), with
    no test catching it since LCC% wasn't covered here originally.
    """

    def _df(self):
        return pd.DataFrame({
            "Loan No": [f"L{i}" for i in range(4)],
            "Unit": ["A"] * 4,
            # Deliberately different from Cum Coll (Inst+Exp) so a numerator mix-up
            # would be caught by a value mismatch, not masked by coincidence.
            "Total Cum Collection":  [500.0, 500.0, 500.0, 500.0],
            "Cum Coll (Inst+Exp)":   [80.0, 90.0, 200.0, 50.0],
            "Cum Due-Inst":          [50.0, 50.0, 100.0, 60.0],
            "Cum Due-Exp":           [30.0, 30.0, 50.0, 20.0],
            "Net Collection Demand Inst+Exp+BC": [100.0] * 4,
            "Month Collection (Excluding Reserve Collection)": [100.0] * 4,
            "SOH": [1000.0] * 4,
            "curr_bucket": ["STD"] * 4,
            "Strike": ["Y"] * 4,
        })

    def _via_compiler_lcc(self, df: pd.DataFrame) -> float:
        return _via_compiler("lcc_pct", df)

    def test_dashboard_lcc_matches_ai_query_path(self):
        df = self._df()
        dashboard_lcc = compute_metrics(df, df.iloc[0:0])["LCC%"][0]
        assert dashboard_lcc == self._via_compiler_lcc(df)

    def test_still_matches_when_ratio_would_exceed_100(self):
        df = self._df()
        df["Cum Coll (Inst+Exp)"] = [200.0, 200.0, 200.0, 200.0]  # > cum due -- both paths must cap at 100
        dashboard_lcc = compute_metrics(df, df.iloc[0:0])["LCC%"][0]
        assert dashboard_lcc == self._via_compiler_lcc(df) == 100.0


class TestFlagConceptsMatchDashboard:
    """The AI Query / Investigator concepts (registry/ontology.py) and the
    dashboard's own masks must pick the SAME loans. They once disagreed: on
    the demo file Alerts showed 33 non-starters and AI Query 9, and the
    co-lending / not-paying-3M concepts matched "Y" but not "Yes"."""

    def _flags(self):
        from config import NOT_PAYING_3M_FLAG_COL
        from helpers import make_df
        return make_df([
            {"Non Starter": "Y",   "Arrears / EMI": 1.0, "VehEMI Accrued": 1},
            {"Non Starter": "Yes", "Arrears / EMI": 2.0, "VehEMI Accrued": 3},
            {"Non Starter": "Y",   "Arrears / EMI": 0.0, "VehEMI Accrued": 2},   # cleared: not one
            {"Non Starter": "N",   "Arrears / EMI": 4.0, "VehEMI Accrued": 1},
            {"CoLending_Loans": "Yes", "Arrears / EMI": 1.0},
            {"CoLending_Loans": "Y",   "Arrears / EMI": 0.5},
            {"CoLending_Loans": "Y",   "Arrears / EMI": 0.0},
        ]).assign(**{NOT_PAYING_3M_FLAG_COL: ["Yes", "Y", "N", "NO", "N", "N", "N"]})

    def _concept(self, df, name):
        from investigator.steps import concept_filter
        return len(concept_filter(df, name))

    def test_non_starter(self):
        from smart_alerts import alert_non_starters
        from utils import non_starter_mask
        df = self._flags()
        assert self._concept(df, "non_starter") == int(non_starter_mask(df).sum()) \
            == alert_non_starters(df)["count"] == 2

    def test_colending_at_risk_accepts_yes(self):
        from smart_alerts import alert_colending_at_risk
        df = self._flags()
        assert self._concept(df, "colending_at_risk") == alert_colending_at_risk(df)["count"] == 2

    def test_no_collection_3m_accepts_yes(self):
        from utils import loan_flags
        df = self._flags()
        assert self._concept(df, "no_collection_3m") == int(loan_flags(df)["not_paying_3m"].sum()) == 2

    def test_priority_tier_uses_the_same_non_starter_rule(self):
        from registry.ontology import CONCEPTS, PRIORITY_RULES
        tier = next(r for r in PRIORITY_RULES if r["label"] == "Non Starters")
        assert tier["conditions"] == CONCEPTS["non_starter"]["conditions"]

    def test_demo_file_counts_agree_across_tabs(self):
        from pathlib import Path
        from smart_alerts import alert_non_starters
        from utils import load_and_validate, non_starter_mask
        path = Path(__file__).resolve().parent.parent / "sample_data" / "Current_Month_Demo.xlsx"
        with open(path, "rb") as f:
            df, errs = load_and_validate.__wrapped__(f)
        assert errs == []
        n = int(non_starter_mask(df).sum())
        assert n > 0
        assert self._concept(df, "non_starter") == alert_non_starters(df)["count"] == n


class TestCollectionPctsConsistency:
    """Collection %, Overdue Collection % and Month Demand Collection %: the AI
    Query metrics must equal the Dashboard's own figures (utils.unit_metrics
    and utils.compute_overdue_demand_pct)."""

    def _df(self):
        from helpers import make_df
        from utils import assign_buckets
        return assign_buckets(make_df([
            {"Arrear Opening": 5_000.0, "Month Collection (Excluding Reserve Collection)": 12_000.0,
             "Net Collection Demand Inst+Exp+BC": 10_000.0},
            {"Arrear Opening": 0.0, "Month Collection (Excluding Reserve Collection)": 3_000.0,
             "Net Collection Demand Inst+Exp+BC": 10_000.0},
            {"Arrear Opening": 20_000.0, "Month Collection (Excluding Reserve Collection)": 0.0,
             "Net Collection Demand Inst+Exp+BC": 8_000.0},
        ]))

    def test_collection_pct(self):
        from utils import portfolio_metrics
        df = self._df()
        assert _via_compiler("collection_pct", df) == float(portfolio_metrics(df)["Collection%"])

    def test_overdue_and_month_demand_collection_pct(self):
        from utils import compute_overdue_demand_pct
        df = self._df()
        shared = compute_overdue_demand_pct(df)
        assert _via_compiler("overdue_collection_pct", df) == shared["overdue_pct"]
        assert _via_compiler("month_demand_collection_pct", df) == shared["demand_pct"]
