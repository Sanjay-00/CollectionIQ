"""
Month-over-month reduction / change capability.

When a previous-month file is uploaded, a curated set of numeric columns is
carried into the current-month DataFrame as prev_* columns (see
utils.PREV_CARRYOVER_COLS and the merge in app.py::_cached_filter). These tests
lock in the live execution path that answers multi-condition "case reduction"
questions (e.g. insurance cases): the step-plan engine's conditional 'where'
counts on prev_* and current columns.

They also guard the data-prep contract: PREV_CARRYOVER_COLS names must be
eval-safe (no spaces/slashes) so derive expressions and the validators work.

The equivalent amount-reduction capability (SOH reduction ranking, etc.) is
covered live via the compiler's time.compare path in tests/test_temporal.py -
not duplicated here.
"""
import re

import pandas as pd

from utils import PREV_CARRYOVER_COLS
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
