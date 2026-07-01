import pandas as pd
import pytest

from agents.plan_executor import execute_plan, validate_plan


def _fleet_df():
    # 3 branches; customers keyed by mobile. Customer with >3 loans = "fleet".
    rows = []
    # Branch A: cust 1 has 4 loans (fleet), cust 2 has 2 loans
    rows += [("A", "111", f"A1-{i}") for i in range(4)]
    rows += [("A", "222", f"A2-{i}") for i in range(2)]
    # Branch B: cust 3 has 5 loans (fleet), cust 4 has 4 loans (fleet)
    rows += [("B", "333", f"B3-{i}") for i in range(5)]
    rows += [("B", "444", f"B4-{i}") for i in range(4)]
    # Branch C: cust 5 has 1 loan
    rows += [("C", "555", "C5-0")]
    return pd.DataFrame(rows, columns=["Unit", "Cust Mob No", "Loan No"])


FLEET_PLAN = [
    {"op": "group_aggregate", "group_by": ["Unit", "Cust Mob No"],
     "aggregations": [{"alias": "loan_count", "func": "nunique", "column": "Loan No"}]},
    {"op": "filter", "conditions": [{"column": "loan_count", "op": ">", "value": 3}]},
    {"op": "group_aggregate", "group_by": ["Unit"],
     "aggregations": [{"alias": "customer_count", "func": "nunique", "column": "Cust Mob No"}]},
    {"op": "sort", "by": "customer_count", "ascending": False},
]


class TestExecutePlan:
    def test_nested_count_distinct(self):
        out, err = execute_plan(_fleet_df(), FLEET_PLAN)
        assert err == ""
        counts = dict(zip(out["Unit"], out["customer_count"]))
        assert counts == {"B": 2, "A": 1}   # C has no fleet customer -> excluded
        # sorted descending
        assert list(out["Unit"]) == ["B", "A"]

    def test_limit(self):
        plan = FLEET_PLAN + [{"op": "limit", "n": 1}]
        out, err = execute_plan(_fleet_df(), plan)
        assert err == ""
        assert len(out) == 1
        assert out.iloc[0]["Unit"] == "B"

    def test_derive(self):
        df = pd.DataFrame({"Unit": ["A", "B"], "Loan No": ["1", "2"],
                           "prev": ["NPA", "NPA"], "curr": ["STD", "NPA"]})
        plan = [
            {"op": "group_aggregate", "group_by": ["Unit"],
             "aggregations": [{"alias": "n", "func": "count"}]},
            {"op": "derive", "column": "doubled", "expr": "n * 2"},
        ]
        out, err = execute_plan(df, plan)
        assert err == ""
        assert set(out["doubled"]) == {2}

    def test_sum_func(self):
        df = pd.DataFrame({"Unit": ["A", "A", "B"], "SOH": [10, 20, 5]})
        plan = [{"op": "group_aggregate", "group_by": ["Unit"],
                 "aggregations": [{"alias": "exposure", "func": "sum", "column": "SOH"}]}]
        out, err = execute_plan(df, plan)
        assert err == ""
        assert dict(zip(out["Unit"], out["exposure"])) == {"A": 30, "B": 5}

    def test_empty_after_filter_returns_empty_df_no_error(self):
        # A valid plan that matches zero rows returns an empty DataFrame, not an error.
        # Empty results are semantically correct (no matches), not a pipeline failure.
        plan = [
            {"op": "group_aggregate", "group_by": ["Unit", "Cust Mob No"],
             "aggregations": [{"alias": "loan_count", "func": "nunique", "column": "Loan No"}]},
            {"op": "filter", "conditions": [{"column": "loan_count", "op": ">", "value": 999}]},
        ]
        out, err = execute_plan(_fleet_df(), plan)
        assert out.empty and err == ""

    def test_unknown_op(self):
        out, err = execute_plan(_fleet_df(), [{"op": "frobnicate"}])
        assert out.empty and "unknown operation" in err

    def test_conditional_aggregation_where(self):
        # Per branch: total loans and count of NPA loans (conditional count).
        df = pd.DataFrame({
            "Unit":   ["A", "A", "A", "B", "B"],
            "Loan No": ["1", "2", "3", "4", "5"],
            "curr_bucket": ["NPA", "STD", "NPA", "STD", "STD"],
        })
        plan = [{"op": "group_aggregate", "group_by": ["Unit"], "aggregations": [
            {"alias": "total", "func": "count"},
            {"alias": "npa", "func": "count",
             "where": [{"column": "curr_bucket", "op": "==", "value": "NPA"}]},
        ]}]
        out, err = execute_plan(df, plan)
        assert err == ""
        got = {r["Unit"]: (r["total"], r["npa"]) for _, r in out.iterrows()}
        assert got == {"A": (3, 2), "B": (2, 0)}   # B has 0 NPA but still appears

    def test_where_then_derive_unpaid(self):
        # paid_loans = count where flag == Y; unpaid = total - paid
        df = pd.DataFrame({
            "Unit": ["A", "A", "A"],
            "Cust Mob No": ["1", "1", "1"],
            "Loan No": ["1", "2", "3"],
            "paid_flag": ["Y", "N", "N"],
        })
        plan = [
            {"op": "group_aggregate", "group_by": ["Unit", "Cust Mob No"], "aggregations": [
                {"alias": "loan_count", "func": "nunique", "column": "Loan No"},
                {"alias": "paid", "func": "count",
                 "where": [{"column": "paid_flag", "op": "==", "value": "Y"}]}]},
            {"op": "derive", "column": "unpaid", "expr": "loan_count - paid"},
        ]
        out, err = execute_plan(df, plan)
        assert err == ""
        assert int(out.iloc[0]["unpaid"]) == 2

    def test_where_column_validated(self):
        plan = [{"op": "group_aggregate", "group_by": ["Unit"], "aggregations": [
            {"alias": "x", "func": "count",
             "where": [{"column": "GHOST", "op": "==", "value": "Y"}]}]}]
        errs = validate_plan(plan, ["Unit", "Loan No"])
        assert any("GHOST" in e for e in errs)


def _rich_df():
    """Region / Branch / Customer / Loan, with known per-customer loan counts."""
    rows = []
    rows += [("W", "A", "1", f"a1{i}") for i in range(3)]   # cust 1: 3 loans
    rows += [("W", "A", "2", f"a2{i}") for i in range(2)]   # cust 2: 2 loans
    rows += [("W", "B", "3", f"b3{i}") for i in range(4)]   # cust 3: 4 loans
    rows += [("E", "C", "4", "c40")]                        # cust 4: 1 loan
    rows += [("E", "C", "5", f"c5{i}") for i in range(2)]   # cust 5: 2 loans
    return pd.DataFrame(rows, columns=["RegionName", "Unit", "Cust Mob No", "Loan No"])


class TestComplexPlans:
    def test_three_level_rollup(self):
        # customer -> branch -> region, with a per-customer threshold in the middle
        plan = [
            {"op": "group_aggregate", "group_by": ["RegionName", "Unit", "Cust Mob No"],
             "aggregations": [{"alias": "loan_count", "func": "nunique", "column": "Loan No"}]},
            {"op": "filter", "conditions": [{"column": "loan_count", "op": ">=", "value": 2}]},
            {"op": "group_aggregate", "group_by": ["RegionName", "Unit"],
             "aggregations": [{"alias": "fleet", "func": "nunique", "column": "Cust Mob No"}]},
            {"op": "group_aggregate", "group_by": ["RegionName"],
             "aggregations": [{"alias": "total_fleet", "func": "sum", "column": "fleet"},
                              {"alias": "branch_count", "func": "nunique", "column": "Unit"}]},
            {"op": "sort", "by": "total_fleet", "ascending": False},
        ]
        out, err = execute_plan(_rich_df(), plan)
        assert err == ""
        got = {r["RegionName"]: (int(r["total_fleet"]), int(r["branch_count"])) for _, r in out.iterrows()}
        assert got == {"W": (3, 2), "E": (1, 1)}
        assert list(out["RegionName"]) == ["W", "E"]   # sorted by total_fleet desc

    def test_conditional_count_with_date_where_and_derive(self):
        df = pd.DataFrame({
            "Unit": ["A", "A", "A"], "Cust Mob No": ["1", "1", "1"],
            "Loan No": ["L1", "L2", "L3"],
            "Last Receipt Date": pd.to_datetime(["2026-03-05", "2026-02-10", None]),
        })
        plan = [
            {"op": "group_aggregate", "group_by": ["Unit", "Cust Mob No"], "aggregations": [
                {"alias": "loan_count", "func": "nunique", "column": "Loan No"},
                {"alias": "paid", "func": "count",
                 "where": [{"column": "Last Receipt Date", "op": ">=", "value": "2026-03-01"}]}]},
            {"op": "derive", "column": "unpaid", "expr": "loan_count - paid"},
        ]
        out, err = execute_plan(df, plan)
        assert err == ""
        assert int(out.iloc[0]["paid"]) == 1      # only the Mar receipt; NaT counts as unpaid
        assert int(out.iloc[0]["unpaid"]) == 2

    def test_mean_with_where(self):
        df = pd.DataFrame({"Unit": ["A", "A", "A"], "val": [10, 30, 20], "flag": ["Y", "N", "Y"]})
        plan = [{"op": "group_aggregate", "group_by": ["Unit"], "aggregations": [
            {"alias": "avg_y", "func": "mean", "column": "val",
             "where": [{"column": "flag", "op": "==", "value": "Y"}]}]}]
        out, err = execute_plan(df, plan)
        assert err == ""
        assert float(out.iloc[0]["avg_y"]) == 15.0   # mean of 10 and 20

    def test_where_matching_nothing_keeps_group_as_zero(self):
        df = pd.DataFrame({"Unit": ["A", "A", "B"], "bucket": ["NPA", "STD", "STD"]})
        plan = [{"op": "group_aggregate", "group_by": ["Unit"], "aggregations": [
            {"alias": "npa", "func": "count",
             "where": [{"column": "bucket", "op": "==", "value": "NPA"}]}]}]
        out, err = execute_plan(df, plan)
        assert err == ""
        assert {r["Unit"]: int(r["npa"]) for _, r in out.iterrows()} == {"A": 1, "B": 0}

    def test_derive_divide_by_zero_is_zero(self):
        df = pd.DataFrame({"Unit": ["A"], "bucket": ["STD"]})
        plan = [
            {"op": "group_aggregate", "group_by": ["Unit"], "aggregations": [
                {"alias": "total", "func": "count"},
                {"alias": "zero", "func": "count",
                 "where": [{"column": "bucket", "op": "==", "value": "ZZZ"}]}]},
            {"op": "derive", "column": "ratio", "expr": "total / zero"},
        ]
        out, err = execute_plan(df, plan)
        assert err == ""
        assert float(out.iloc[0]["ratio"]) == 0.0   # inf -> 0

    def test_validate_three_level_ok(self):
        plan = [
            {"op": "group_aggregate", "group_by": ["RegionName", "Unit", "Cust Mob No"],
             "aggregations": [{"alias": "loan_count", "func": "nunique", "column": "Loan No"}]},
            {"op": "filter", "conditions": [{"column": "loan_count", "op": ">=", "value": 2}]},
            {"op": "group_aggregate", "group_by": ["RegionName"],
             "aggregations": [{"alias": "fleet", "func": "nunique", "column": "Cust Mob No"}]},
        ]
        assert validate_plan(plan, _rich_df().columns) == []

    def test_validate_catches_column_dropped_two_steps_back(self):
        # after the first group_aggregate, "Loan No" is gone; a later sort on it is invalid
        plan = [
            {"op": "group_aggregate", "group_by": ["RegionName", "Unit"],
             "aggregations": [{"alias": "n", "func": "count"}]},
            {"op": "filter", "conditions": [{"column": "n", "op": ">", "value": 1}]},
            {"op": "sort", "by": "Loan No", "ascending": False},
        ]
        errs = validate_plan(plan, _rich_df().columns)
        assert any("Loan No" in e for e in errs)


class TestValidatePlan:
    def test_valid_plan(self):
        assert validate_plan(FLEET_PLAN, _fleet_df().columns) == []

    def test_bad_group_by_column(self):
        plan = [{"op": "group_aggregate", "group_by": ["BRANCHX"],
                 "aggregations": [{"alias": "n", "func": "count"}]}]
        errs = validate_plan(plan, ["Unit", "Loan No"])
        assert any("BRANCHX" in e for e in errs)

    def test_column_tracking_after_aggregate(self):
        # step 2 references an alias from step 1 -> valid; and a fake -> invalid
        plan = [
            {"op": "group_aggregate", "group_by": ["Unit"],
             "aggregations": [{"alias": "n", "func": "count"}]},
            {"op": "filter", "conditions": [{"column": "n", "op": ">", "value": 1}]},
            {"op": "filter", "conditions": [{"column": "ghost", "op": ">", "value": 1}]},
        ]
        errs = validate_plan(plan, ["Unit", "Loan No"])
        assert any("ghost" in e for e in errs)
        assert not any("'n'" in e for e in errs)   # alias n is valid

    def test_original_column_dropped_after_aggregate(self):
        # Loan No is gone after a group_aggregate that didn't keep it
        plan = [
            {"op": "group_aggregate", "group_by": ["Unit"],
             "aggregations": [{"alias": "n", "func": "count"}]},
            {"op": "sort", "by": "Loan No", "ascending": False},
        ]
        errs = validate_plan(plan, ["Unit", "Loan No"])
        assert any("Loan No" in e for e in errs)
