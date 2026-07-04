"""Nested aggregation tests (Step 2 of IR-2).

The compiler derives a multi-pass plan (intermediate grain -> filter -> terminal
grain) from the grain lattice. The IR-2 the LLM emits contains NO step list, no
group keys, no pass ordering - only dimension names, an entity predicate, and
measure names. These tests assert correct numbers for the five design walkthroughs,
including the ambiguity case the compiler must refuse to guess.
"""
import pandas as pd

from compiler.core import compile_logical
from agents.plan_executor import execute_plan


def _df():
    """cust 111: 4 loans in branch A / region R1 (2 NPA).
       cust 222: 2 loans in A / R1.
       cust 333: 3 loans in B / R1 (3 NPA).
       cust 444: 4 loans in C / R2 (1 NPA).
       Every loan: SOH 100; Cum Coll 90; Cum Due-Inst 50; Cum Due-Exp 50."""
    rows = []
    def add(loan, cust, unit, region, bucket):
        rows.append({
            "Loan No": loan, "Cust Mob No": cust, "Unit": unit, "RegionName": region,
            "curr_bucket": bucket, "SOH": 100.0,
            "Cum Coll (Inst+Exp)": 90.0, "Cum Due-Inst": 50.0, "Cum Due-Exp": 50.0,
        })
    for i in range(1, 5):  add(i,   111, "A", "R1", "NPA" if i <= 2 else "STD")
    for i in range(5, 7):  add(i,   222, "A", "R1", "STD")
    for i in range(7, 10): add(i,   333, "B", "R1", "NPA")
    for i in range(10, 14):add(i,   444, "C", "R2", "NPA" if i == 10 else "STD")
    return pd.DataFrame(rows)


def _run(ir):
    df = _df()
    plan, errs = compile_logical(ir, df.columns)
    assert errs == [], errs
    out, err = execute_plan(df, plan)
    assert err == ""
    return out


class TestQuery1_CustomersPerBranchWithManyLoans:
    def test(self):
        out = _run({
            "dimensions": ["branch"],
            "entity_filters": [{"entity": "customer",
                                "having": [{"agg": "count", "distinct": "Loan No", "op": ">", "value": 3}]}],
            "measures": [{"agg": "count", "distinct": "Cust Mob No", "alias": "customer_count"}],
        })
        got = dict(zip(out["Unit"], out["customer_count"]))
        assert got.get("A") == 1            # cust 111 (4 loans)
        assert got.get("C") == 1            # cust 444 (4 loans)
        assert "B" not in got               # cust 333 has 3 (not > 3) -> branch drops out


class TestQuery2_FleetOperatorsWithExposure:
    def test(self):
        out = _run({
            "dimensions": ["region"],
            "entity_filters": [{"concept": "fleet_operator"}],   # >= 3 loans, from registry
            "measures": [
                {"agg": "count", "distinct": "Cust Mob No", "alias": "fleet_operators"},
                {"metric": "exposure", "alias": "total_exposure"},
            ],
        })
        r1 = out[out["RegionName"] == "R1"].iloc[0]
        r2 = out[out["RegionName"] == "R2"].iloc[0]
        assert r1["fleet_operators"] == 2 and r1["total_exposure"] == 700.0  # 111(400)+333(300)
        assert r2["fleet_operators"] == 1 and r2["total_exposure"] == 400.0  # 444


class TestQuery3_BorrowersWithMultipleNpaLoans:
    def test(self):
        out = _run({
            "dimensions": ["branch"],
            "entity_filters": [{"entity": "customer",
                                "having": [{"agg": "count", "concept": "npa", "op": ">=", "value": 2}]}],
            "measures": [{"agg": "count", "distinct": "Cust Mob No", "alias": "borrowers"}],
        })
        got = dict(zip(out["Unit"], out["borrowers"]))
        assert got.get("A") == 1            # cust 111 (2 NPA)
        assert got.get("B") == 1            # cust 333 (3 NPA)
        assert "C" not in got               # cust 444 has only 1 NPA


class TestQuery4_FleetLccRatioRollup:
    def test(self):
        out = _run({
            "dimensions": ["region"],
            "entity_filters": [{"concept": "fleet_operator"}],
            "measures": [{"metric": "lcc_pct", "alias": "fleet_lcc"}],
        })
        # All fleet loans: 90/100 -> 90% (sum(num)/sum(den)*100), capped at 100.
        assert out[out["RegionName"] == "R1"].iloc[0]["fleet_lcc"] == 90.0
        assert out[out["RegionName"] == "R2"].iloc[0]["fleet_lcc"] == 90.0


class TestQuery5_AmbiguityIsRefused:
    def test_multi_branch_per_region_is_grain_ambiguous(self):
        # Predicate aggregates over Unit (branch) but output is region: a customer
        # can span branches across regions -> compiler must refuse, not guess.
        plan, errs = compile_logical({
            "dimensions": ["region"],
            "entity_filters": [{"entity": "customer",
                                "having": [{"agg": "nunique", "column": "Unit", "op": ">=", "value": 3}]}],
            "measures": [{"agg": "count", "distinct": "Cust Mob No", "alias": "multi_branch"}],
        }, _df().columns)
        assert any("grain ambiguity" in e for e in errs)


class TestStructureIsCompilerDerived:
    def test_ir_has_no_step_list(self):
        # The IR-2 the LLM emits declares intent only; the multi-pass STRUCTURE is
        # produced by the compiler. Two group_aggregate passes appear in the PLAN,
        # never in the IR.
        ir = {
            "dimensions": ["branch"],
            "entity_filters": [{"entity": "customer",
                                "having": [{"agg": "count", "distinct": "Loan No", "op": ">", "value": 3}]}],
            "measures": [{"agg": "count", "distinct": "Cust Mob No", "alias": "customer_count"}],
        }
        assert "op" not in ir and "plan" not in ir and "stages" not in ir
        plan, errs = compile_logical(ir, _df().columns)
        assert errs == []
        passes = [s for s in plan if s["op"] == "group_aggregate"]
        assert len(passes) == 2                       # intermediate + terminal, derived
        assert passes[0]["group_by"] == ["Unit", "Cust Mob No"]
        assert passes[1]["group_by"] == ["Unit"]


class TestNonNestableMeasureRejected:
    def test_mean_cannot_roll_up(self):
        plan, errs = compile_logical({
            "dimensions": ["branch"],
            "entity_filters": [{"concept": "fleet_operator"}],
            "measures": [{"column": "SOH", "agg": "mean", "alias": "avg_soh"}],
        }, _df().columns)
        assert any("cannot be rolled up" in e for e in errs)
