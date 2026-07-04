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
from utils import compute_strike_pct, compute_hard_bucket_pct, compute_metrics


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


class TestStrikePctConsistency:
    def test_ai_query_path_matches_shared_helper(self):
        df = _df()
        assert _via_compiler("strike_pct", df) == compute_strike_pct(df)

    def test_still_matches_with_invalid_strike_values_present(self):
        # compute_strike_pct excludes rows where Strike isn't Y/N from the denominator;
        # the compiler's denominator_where (Strike in [Y, N]) must do the same.
        df = _df()
        df.loc[0, "Strike"] = ""
        df.loc[1, "Strike"] = "MAYBE"
        assert _via_compiler("strike_pct", df) == compute_strike_pct(df)


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
