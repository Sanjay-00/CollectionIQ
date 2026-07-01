"""Measure-framework tests (Step 1 of IR-2).

Proves the generic measure framework: ratio measures recompute from
sum(num)/sum(den) at the target grain (NOT mean-of-ratios), semi-additive measures
refuse aggregation over a banned axis, existing metrics stay backward compatible,
and a BRAND-NEW measure kind can be added by registering a handler WITHOUT editing
the compiler.
"""
import pandas as pd

from compiler.core import compile_logical
from compiler.measures import MEASURE_HANDLERS, measure_handler, Lowered
from agents.plan_executor import execute_plan


def _lcc_df():
    """Two branches. Branch A: collections 80,90 vs dues (50+50)=100, (40+40)=80.
        sum(num)=170, sum(den)=180 -> 170/180*100 = 94.44 (grain-correct).
        mean of per-loan LCC% = mean(80, 112.5->cap100) ... = 90 (WRONG way).
       Branch B: 200 vs 100 -> 200% -> capped to 100."""
    return pd.DataFrame({
        "Loan No":            [1, 2, 3],
        "Unit":               ["A", "A", "B"],
        "Cum Coll (Inst+Exp)": [80.0, 90.0, 200.0],
        "Cum Due-Inst":        [50.0, 40.0, 80.0],
        "Cum Due-Exp":         [50.0, 40.0, 20.0],
    })


class TestRatioMeasure:
    def test_lcc_recomputed_at_grain_not_averaged(self):
        ir = {"dimensions": ["branch"], "measures": [{"metric": "lcc_pct"}]}
        df = _lcc_df()
        plan, errs = compile_logical(ir, df.columns)
        assert errs == [], errs
        out, err = execute_plan(df, plan)
        assert err == ""
        a = out[out["Unit"] == "A"].iloc[0]["lcc_pct"]
        b = out[out["Unit"] == "B"].iloc[0]["lcc_pct"]
        assert round(a, 2) == 94.44      # sum(170)/sum(180)*100, the correct way
        assert round(a, 2) != 90.00      # NOT the mean of per-loan ratios
        assert b == 100.0                # 200% capped at 100

    def test_ratio_lowers_to_sums_plus_divide_not_a_mean(self):
        ir = {"dimensions": ["branch"], "measures": [{"metric": "lcc_pct"}]}
        plan, errs = compile_logical(ir, _lcc_df().columns)
        ga = next(s for s in plan if s["op"] == "group_aggregate")
        funcs = {a["func"] for a in ga["aggregations"]}
        assert funcs == {"sum"}          # only sums of num/den, never a mean of the ratio
        assert any(s["op"] == "derive" and s["column"] == "lcc_pct" for s in plan)


class TestCountDistinct:
    def test_count_distinct_lowers_to_nunique(self):
        ir = {"dimensions": ["branch"],
              "measures": [{"agg": "count", "distinct": "Cust Mob No", "alias": "customers"}]}
        cols = ["Unit", "Cust Mob No"]
        plan, errs = compile_logical(ir, cols)
        assert errs == []
        ga = next(s for s in plan if s["op"] == "group_aggregate")
        assert ga["aggregations"][0] == {"alias": "customers", "func": "nunique", "column": "Cust Mob No"}


class TestSemiAdditiveGuard:
    def test_refuses_aggregation_over_banned_axis(self):
        # Directly exercise the handler with a context that aggregates over time.
        mdef = {"alias": "soh", "kind": "semi_additive", "column": "SOH",
                "non_additive_over": ["time"]}
        low = MEASURE_HANDLERS["semi_additive"](mdef, {"aggregating_over": {"time"}})
        assert low.errors and "semi-additive" in low.errors[0]

    def test_allows_aggregation_over_other_dimensions(self):
        mdef = {"alias": "soh", "kind": "semi_additive", "column": "SOH",
                "non_additive_over": ["time"]}
        low = MEASURE_HANDLERS["semi_additive"](mdef, {"aggregating_over": set()})
        assert low.errors == []
        assert low.aggregations[0]["column"] == "SOH"


class TestBackwardCompatibility:
    def test_existing_additive_metric_unchanged(self):
        ir = {"dimensions": ["branch"], "measures": [{"metric": "exposure"}]}
        cols = ["Unit", "SOH"]
        plan, errs = compile_logical(ir, cols)
        assert errs == []
        ga = next(s for s in plan if s["op"] == "group_aggregate")
        assert ga["aggregations"][0] == {"alias": "exposure", "func": "sum", "column": "SOH"}

    def test_inline_count_still_works(self):
        ir = {"dimensions": ["branch"], "measures": [{"agg": "count", "alias": "total"}]}
        plan, errs = compile_logical(ir, ["Unit"])
        assert errs == []
        ga = next(s for s in plan if s["op"] == "group_aggregate")
        assert ga["aggregations"][0] == {"alias": "total", "func": "count"}


class TestUnknownKind:
    def test_unknown_kind_is_loud_error(self):
        ir = {"dimensions": ["branch"], "measures": [{"kind": "teleport", "column": "SOH"}]}
        plan, errs = compile_logical(ir, ["Unit", "SOH"])
        assert any("unknown measure kind 'teleport'" in e for e in errs)


class TestExtensibility:
    def test_new_kind_added_without_touching_compiler(self):
        # Register a brand-new measure kind purely via the handler registry.
        @measure_handler("sum_of_squares")
        def _sos(mdef, ctx):
            col = mdef["column"]
            pre = [{"column": f"{mdef['alias']}__sq", "expr": f"{col} * {col}"}]
            agg = [{"alias": mdef["alias"], "func": "sum", "column": f"{mdef['alias']}__sq"}]
            return Lowered(pre_derives=pre, aggregations=agg)
        try:
            df = pd.DataFrame({"Unit": ["A", "A"], "POS": [2.0, 3.0]})
            ir = {"dimensions": ["branch"],
                  "measures": [{"kind": "sum_of_squares", "column": "POS", "alias": "sos"}]}
            plan, errs = compile_logical(ir, df.columns)
            assert errs == [], errs
            # The pre-aggregation derive the handler asked for is in the plan, before grouping.
            assert plan[0]["op"] == "derive" and plan[0]["column"] == "sos__sq"
            out, err = execute_plan(df, plan)
            assert err == ""
            assert out.iloc[0]["sos"] == 13.0   # 2^2 + 3^2
        finally:
            MEASURE_HANDLERS.pop("sum_of_squares", None)
