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
from utils import compute_strike_pct, compute_hard_bucket_pct


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
