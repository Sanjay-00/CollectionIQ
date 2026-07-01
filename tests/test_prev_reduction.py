"""
Month-over-month reduction / change capability.

When a previous-month file is uploaded, a curated set of numeric columns is
carried into the current-month DataFrame as prev_* columns (see
utils.PREV_CARRYOVER_COLS and the merge in app.py::_cached_filter). These tests
lock in the two execution paths that answer "reduction vs last month" questions:

1. amount reductions (SOH, arrears, ...) via a single GROUP BY (execute_aggregation)
2. multi-condition case reductions (e.g. insurance cases) via the step-plan
   engine's conditional 'where' counts on prev_* and current columns.

They also guard the data-prep contract: PREV_CARRYOVER_COLS names must be
eval-safe (no spaces/slashes) so derive expressions and the validators work.
"""
import re

import pandas as pd

from utils import PREV_CARRYOVER_COLS
from agents.data_executor import execute_aggregation, validate_aggregation_spec
from agents.plan_executor import execute_plan, validate_plan


def _df():
    """df_curr with prev_* columns already merged in. Loan 6 is new this month
    (no previous row), so its prev_* values are NaN."""
    return pd.DataFrame({
        "Loan No":    [1, 2, 3, 4, 5, 6],
        "RegionName": ["PUNE", "PUNE", "PUNE", "MUMBAI", "MUMBAI", "MUMBAI"],
        "SOH":        [100.0, 200.0, 50.0, 300.0, 100.0, 0.0],
        "prev_SOH":   [150.0, 250.0, 50.0, 200.0, 400.0, float("nan")],
        "ARREARS AGAINST INST": [0, 0, 100, 0, 0, 0],
        "ARREARS AGAINST EXP":  [6000, 0, 6000, 7000, 0, 8000],
        "Arrears / EMI":        [1, 0, 2, 1, 0, 1],
        "prev_Arrears_Inst":    [0, 0, 0, 0, 0, 0],
        "prev_Arrears_Exp":     [6000, 6000, 0, 0, 0, 0],
        "prev_Arrears_EMI":     [1, 1, 0, 0, 0, 0],
    })


class TestCarryoverContract:
    def test_prev_names_are_eval_safe(self):
        # derive (df.eval) and the validator regex require plain identifiers.
        ident = re.compile(r"^[A-Za-z_]\w*$")
        for target in PREV_CARRYOVER_COLS.values():
            assert ident.match(target), f"{target} is not an eval-safe identifier"

    def test_targets_are_prefixed_and_unique(self):
        targets = list(PREV_CARRYOVER_COLS.values())
        assert all(t.startswith("prev_") for t in targets)
        assert len(targets) == len(set(targets))


class TestAmountReduction:
    def test_soh_reduction_per_region_ranks_biggest_drop_first(self):
        spec = {
            "group_by": "RegionName",
            "counts": [],
            "sums": [
                {"alias": "prev_soh", "column": "prev_SOH"},
                {"alias": "curr_soh", "column": "SOH"},
            ],
            "metrics": [{"expr": "prev_soh - curr_soh", "label": "SOH Reduction"}],
            "sort_asc": False,
        }
        assert validate_aggregation_spec(spec, _df().columns) == []
        out, err = execute_aggregation(_df(), spec)
        assert err == ""
        # PUNE: prev 450 - curr 350 = 100. MUMBAI: prev 600 (NaN->skipped) - curr 400 = 200.
        assert list(out["RegionName"]) == ["MUMBAI", "PUNE"]
        assert out.iloc[0]["SOH Reduction"] == 200.0
        assert out.iloc[1]["SOH Reduction"] == 100.0

    def test_new_loan_with_nan_prev_does_not_break_sum(self):
        # Loan 6 (MUMBAI) has NaN prev_SOH; sum must skip it, not yield NaN.
        spec = {
            "group_by": "RegionName",
            "counts": [],
            "sums": [{"alias": "prev_soh", "column": "prev_SOH"}],
            "metrics": [],
            "sort_asc": False,
        }
        out, err = execute_aggregation(_df(), spec)
        assert err == ""
        mumbai = out[out["RegionName"] == "MUMBAI"].iloc[0]
        assert mumbai["prev_soh"] == 600.0  # 200 + 400, NaN skipped


class TestCaseReduction:
    def test_insurance_case_reduction_uses_multi_condition_where(self):
        plan = [
            {"op": "group_aggregate", "group_by": ["RegionName"], "aggregations": [
                {"alias": "ins_prev", "func": "count", "where": [
                    {"column": "prev_Arrears_Inst", "op": "<=", "value": 0},
                    {"column": "prev_Arrears_Exp", "op": ">", "value": 5000},
                    {"column": "prev_Arrears_EMI", "op": ">", "value": 0}]},
                {"alias": "ins_curr", "func": "count", "where": [
                    {"column": "ARREARS AGAINST INST", "op": "<=", "value": 0},
                    {"column": "ARREARS AGAINST EXP", "op": ">", "value": 5000},
                    {"column": "Arrears / EMI", "op": ">", "value": 0}]}]},
            {"op": "derive", "column": "insurance_reduction", "expr": "ins_prev - ins_curr"},
            {"op": "sort", "by": "insurance_reduction", "ascending": False},
        ]
        assert validate_plan(plan, list(_df().columns)) == []
        out, err = execute_plan(_df(), plan)
        assert err == ""
        pune = out[out["RegionName"] == "PUNE"].iloc[0]
        mumbai = out[out["RegionName"] == "MUMBAI"].iloc[0]
        # PUNE prev 2 (L1,L2) - curr 1 (L1) = 1 (improved).
        # MUMBAI prev 0 - curr 2 (L4,L6) = -2 (insurance cases went up).
        assert pune["insurance_reduction"] == 1
        assert mumbai["insurance_reduction"] == -2

    def test_agg_mode_where_count_matches_plan_mode(self):
        # Robustness guarantee: a multi-condition case count must give the SAME
        # answer whether the LLM routes it through aggregation_mode (count + where)
        # or plan_mode (group_aggregate + where). This removes the routing-variance
        # risk of a silently-wrong answer for multi-condition "case" reductions.
        spec = {
            "group_by": "RegionName",
            "counts": [
                {"alias": "ins_prev", "func": "count", "where": [
                    {"column": "prev_Arrears_Inst", "op": "<=", "value": 0},
                    {"column": "prev_Arrears_Exp", "op": ">", "value": 5000},
                    {"column": "prev_Arrears_EMI", "op": ">", "value": 0}]},
                {"alias": "ins_curr", "func": "count", "where": [
                    {"column": "ARREARS AGAINST INST", "op": "<=", "value": 0},
                    {"column": "ARREARS AGAINST EXP", "op": ">", "value": 5000},
                    {"column": "Arrears / EMI", "op": ">", "value": 0}]},
            ],
            "sums": [],
            "metrics": [{"expr": "ins_prev - ins_curr", "label": "Insurance Reduction"}],
            "sort_asc": False,
        }
        assert validate_aggregation_spec(spec, _df().columns) == []
        out, err = execute_aggregation(_df(), spec)
        assert err == ""
        assert out[out["RegionName"] == "PUNE"].iloc[0]["Insurance Reduction"] == 1
        assert out[out["RegionName"] == "MUMBAI"].iloc[0]["Insurance Reduction"] == -2

    def test_agg_where_validator_flags_bad_where_column(self):
        spec = {
            "group_by": "RegionName",
            "counts": [{"alias": "x", "func": "count", "where": [
                {"column": "NoSuchCol", "op": ">", "value": 0}]}],
            "sums": [], "metrics": [],
        }
        errs = validate_aggregation_spec(spec, _df().columns)
        assert any("NoSuchCol" in e for e in errs)

    def test_conditional_count_syntax_variants_all_agree(self):
        # The planner expresses a multi-condition count in several shapes. ALL of
        # them must produce the same correct answer - none may silently fall back
        # to counting every row (__total__). Regression guard for the silent-wrong
        # co-lending-at-risk reduction bug.
        cdf = pd.DataFrame({
            "Loan No": [1, 2, 3, 4, 5],
            "Unit": ["A", "A", "A", "A", "B"],
            "CoLending_Loans": ["Y", "Y", "Y", "N", "Y"],
            "Arrears / EMI": [0, 0, 2, 5, 0],       # curr at-risk co-lending in A: loan3 -> 1
            "prev_Arrears_EMI": [1, 2, 3, 0, 0],    # prev at-risk co-lending in A: loan1,2,3 -> 3
        })
        prev = [{"column": "CoLending_Loans", "op": "==", "value": "Y"},
                {"column": "prev_Arrears_EMI", "op": ">", "value": 0}]
        curr = [{"column": "CoLending_Loans", "op": "==", "value": "Y"},
                {"column": "Arrears / EMI", "op": ">", "value": 0}]

        def spec(pk, ck):
            cp = {"alias": "cl_prev", "column": "__total__"}; cp.update(pk)
            cc = {"alias": "cl_curr", "column": "__total__"}; cc.update(ck)
            return {"group_by": "Unit", "counts": [cp, cc], "sums": [],
                    "metrics": [{"expr": "cl_prev - cl_curr", "label": "Reduction"}],
                    "sort_asc": False}

        variants = [
            (spec({"op": "filter", "value": prev}, {"op": "filter", "value": curr})),
            (spec({"op": "conditional_count", "conditions": prev}, {"op": "conditional_count", "conditions": curr})),
            (spec({"filter": {"and": prev}}, {"filter": {"and": curr}})),
            (spec({"where": prev}, {"where": curr})),
        ]
        for s in variants:
            assert validate_aggregation_spec(s, cdf.columns) == []
            out, err = execute_aggregation(cdf, s)
            assert err == ""
            assert out[out["Unit"] == "A"].iloc[0]["Reduction"] == 2  # 3 - 1

    def test_uninterpretable_conditional_count_is_rejected(self):
        # A count that signals it is conditional but carries no usable conditions
        # must be flagged (-> repair), never silently treated as count-all.
        spec = {"group_by": "Unit",
                "counts": [{"alias": "x", "column": "__total__", "op": "conditional_count"}],
                "sums": [], "metrics": []}
        errs = validate_aggregation_spec(spec, ["Unit"])
        assert any("malformed" in e for e in errs)

    def test_improvised_op_and_with_filters_is_rejected(self):
        # Regression: the model sometimes emits op:"and" + a 'filters' list. That is
        # not a real count op, so it must be rejected rather than silently counting
        # all rows of the top-level column.
        spec = {"group_by": "Unit", "counts": [{
            "alias": "x", "column": "CoLending_Loans", "op": "and", "value": "Y",
            "filters": [{"column": "Arrears / EMI", "op": ">", "value": 0}]}],
            "sums": [], "metrics": []}
        errs = validate_aggregation_spec(spec, ["Unit", "CoLending_Loans", "Arrears / EMI"])
        assert any("malformed" in e for e in errs)

    def test_in_operator_value_list_not_mistaken_for_conditions(self):
        # The legitimate 'in' op carries value=[scalars]; it must NOT be read as a
        # conditional 'where' (which expects dicts).
        cdf = pd.DataFrame({"Unit": ["A", "A", "B"], "curr_bucket": ["NPA", "SMA-2", "STD"]})
        spec = {"group_by": "Unit",
                "counts": [{"alias": "bad_buckets", "column": "curr_bucket",
                            "op": "in", "value": ["NPA", "SMA-2"]}],
                "sums": [], "metrics": []}
        assert validate_aggregation_spec(spec, cdf.columns) == []
        out, err = execute_aggregation(cdf, spec)
        assert err == ""
        assert out[out["Unit"] == "A"].iloc[0]["bad_buckets"] == 2

    def test_validator_flags_missing_prev_column(self):
        # If no prev file was uploaded, prev_* columns are absent  -  the validator
        # must reject a spec that references them rather than silently zero them.
        cols_no_prev = ["Loan No", "RegionName", "SOH"]
        spec = {
            "group_by": "RegionName",
            "counts": [],
            "sums": [{"alias": "prev_soh", "column": "prev_SOH"}],
            "metrics": [],
        }
        errs = validate_aggregation_spec(spec, cols_no_prev)
        assert any("prev_SOH" in e for e in errs)
