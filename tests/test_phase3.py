"""Phase 3 tests  -  compiler reaches parity with aggregation_mode.

Each test builds an aggregation_spec (legacy path) AND the equivalent IR-1 (new
path), runs both engines, and asserts the results match via the shadow comparator.
This is the evidence that aggregation_mode can become a compile target  -  proven
BEFORE any three-mode routing is deleted (correctness-first).
"""
import pandas as pd

from agents.data_executor import execute_aggregation
from agents.plan_executor import execute_plan
from compiler.core import compile_logical
from compiler.shadow import compare_results


# Insurance-case conditions (multi-condition; the dropped-condition-prone kind).
_INS_PREV = [
    {"column": "prev_Arrears_Inst", "op": "<=", "value": 0},
    {"column": "prev_Arrears_Exp", "op": ">", "value": 5000},
    {"column": "prev_Arrears_EMI", "op": ">", "value": 0},
]
_INS_CURR = [
    {"column": "ARREARS AGAINST INST", "op": "<=", "value": 0},
    {"column": "ARREARS AGAINST EXP", "op": ">", "value": 5000},
    {"column": "Arrears / EMI", "op": ">", "value": 0},
]


def _ins_df():
    return pd.DataFrame({
        "Loan No":    [1, 2, 3, 4, 5, 6],
        "RegionName": ["PUNE", "PUNE", "PUNE", "MUMBAI", "MUMBAI", "MUMBAI"],
        "ARREARS AGAINST INST": [0, 0, 100, 0, 0, 0],
        "ARREARS AGAINST EXP":  [6000, 0, 6000, 7000, 0, 8000],
        "Arrears / EMI":        [1, 0, 2, 1, 0, 1],
        "prev_Arrears_Inst":    [0, 0, 0, 0, 0, 0],
        "prev_Arrears_Exp":     [6000, 6000, 0, 0, 0, 0],
        "prev_Arrears_EMI":     [1, 1, 0, 0, 0, 0],
    })


class TestInsuranceReductionParity:
    def test_compiler_matches_aggregation_engine(self):
        spec = {
            "group_by": "RegionName",
            "counts": [
                {"alias": "ins_prev", "func": "count", "where": _INS_PREV},
                {"alias": "ins_curr", "func": "count", "where": _INS_CURR},
            ],
            "sums": [],
            "metrics": [{"expr": "ins_prev - ins_curr", "label": "Insurance Reduction"}],
            "sort_asc": False,
        }
        ir = {
            "dimensions": ["region"],
            "measures": [
                {"agg": "count", "alias": "ins_prev", "where": _INS_PREV},
                {"agg": "count", "alias": "ins_curr", "where": _INS_CURR},
            ],
            "metrics": [{"expr": "ins_prev - ins_curr", "label": "Insurance Reduction"}],
        }
        df = _ins_df()
        legacy, lerr = execute_aggregation(df, spec)
        plan, errs = compile_logical(ir, df.columns)
        assert errs == [], errs
        new, nerr = execute_plan(df, plan)
        assert lerr == "" and nerr == ""
        assert compare_results(legacy, new)["status"] == "match"


class TestConceptCountWithHavingAndPercentParity:
    def _df(self):
        return pd.DataFrame({
            "Loan No":         [1, 2, 3, 4, 5],
            "Unit":            ["A", "A", "A", "B", "B"],
            "CoLending_Loans": ["Y", "Y", "N", "N", "Y"],
            "Arrears / EMI":   [2, 0, 5, 0, 0],
        })

    def test_compiler_matches_aggregation_engine(self):
        # Branch A: 1 co-lending-at-risk of 3 total. Branch B: 0 -> removed by HAVING.
        colending = [
            {"column": "CoLending_Loans", "op": "==", "value": "Y"},
            {"column": "Arrears / EMI", "op": ">", "value": 0},
        ]
        spec = {
            "group_by": "Unit",
            "counts": [
                {"alias": "total", "column": "__total__"},
                {"alias": "at_risk", "func": "count", "where": colending},
            ],
            "sums": [],
            "metrics": [{"expr": "at_risk / total * 100", "label": "At Risk %"}],
            "having": [{"alias": "at_risk", "op": ">=", "value": 1}],
            "sort_asc": False,
        }
        ir = {
            "dimensions": ["branch"],
            "measures": [
                {"agg": "count", "alias": "total"},
                {"agg": "count", "concept": "colending_at_risk", "alias": "at_risk"},
            ],
            "metrics": [{"expr": "at_risk / total * 100", "label": "At Risk %"}],
            "having": [{"alias": "at_risk", "op": ">=", "value": 1}],
            "order_by": [{"by": "At Risk %", "dir": "desc"}],
        }
        df = self._df()
        legacy, lerr = execute_aggregation(df, spec)
        plan, errs = compile_logical(ir, df.columns)
        assert errs == [], errs
        new, nerr = execute_plan(df, plan)
        assert lerr == "" and nerr == ""
        # Both keep only branch A.
        assert len(new) == 1 and new.iloc[0]["Unit"] == "A"
        assert new.iloc[0]["at_risk"] == 1 and new.iloc[0]["total"] == 3
        assert compare_results(legacy, new)["status"] == "match"

    def test_concept_count_uses_full_conditions(self):
        # The compiled conditional count must carry BOTH co-lending conditions.
        ir = {
            "dimensions": ["branch"],
            "measures": [{"agg": "count", "concept": "colending_at_risk", "alias": "at_risk"}],
        }
        plan, errs = compile_logical(ir, self._df().columns)
        assert errs == []
        ga = next(s for s in plan if s["op"] == "group_aggregate")
        where_cols = {c["column"] for c in ga["aggregations"][0]["where"]}
        assert where_cols == {"CoLending_Loans", "Arrears / EMI"}
