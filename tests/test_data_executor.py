import pandas as pd
import pytest

from agents.data_executor import (
    execute_aggregation,
    execute_filters,
    validate_aggregation_spec,
    validate_filter_spec,
)


def _agg_df():
    return pd.DataFrame({
        "Unit":          ["A", "A", "B", "B"],
        "prev_bucket":   ["STD", "NPA", "SMA-1", "NPA"],
        "curr_bucket":   ["NPA", "NPA", "STD", "NPA"],
        "Arrears / EMI": [5, 1, 0, 4],
        "SOH":           [100, 200, 50, 300],
        "Loan No":       ["1", "2", "3", "4"],
    })


def _reduction_df():
    # A: 2 prev NPA -> 1 curr (50%); B: 1 -> 0 (100%); C: 3 -> 1 (66.67%)
    return pd.DataFrame({
        "Unit":        ["A", "A", "B", "C", "C", "C"],
        "prev_bucket": ["NPA", "NPA", "NPA", "NPA", "NPA", "NPA"],
        "curr_bucket": ["NPA", "STD", "STD", "NPA", "STD", "STD"],
    })


_REDUCTION_SPEC = {
    "group_by": "Unit",
    "counts": [
        {"alias": "npa_prev", "column": "prev_bucket", "value": "NPA"},
        {"alias": "npa_curr", "column": "curr_bucket", "value": "NPA"},
    ],
    "sums": [],
    "metrics": [{"expr": "(npa_prev - npa_curr) / npa_prev * 100", "label": "NPA Reduction %"}],
    "sort_asc": False,
}


class TestExecuteAggregation:
    def test_total_and_equality_counts(self):
        spec = {"group_by": "Unit", "counts": [
            {"alias": "total", "column": "__total__"},
            {"alias": "npa_curr", "column": "curr_bucket", "value": "NPA"}],
            "sums": [], "metrics": [], "sort_asc": False}
        out, err = execute_aggregation(_agg_df(), spec)
        assert err == ""
        got = {r["Unit"]: (int(r["total"]), int(r["npa_curr"])) for _, r in out.iterrows()}
        assert got == {"A": (2, 2), "B": (2, 1)}

    def test_numeric_compare_bucket_move_and_sum(self):
        spec = {"group_by": "Unit", "counts": [
            {"alias": "high", "column": "Arrears / EMI", "op": ">", "value": 3},
            {"alias": "worse", "column": "curr_bucket", "op": "bucket_worse_than", "value": "prev_bucket"}],
            "sums": [{"alias": "soh", "column": "SOH"}], "metrics": [], "sort_asc": False}
        out, err = execute_aggregation(_agg_df(), spec)
        assert err == ""
        got = {r["Unit"]: (int(r["high"]), int(r["worse"]), float(r["soh"])) for _, r in out.iterrows()}
        # A: STD->NPA is worse (1); Arrears 5>3 (1); SOH 300. B: no worse move; Arrears 4>3 (1); SOH 350.
        assert got == {"A": (1, 1, 300.0), "B": (1, 0, 350.0)}

    def test_in_operator_count(self):
        spec = {"group_by": "Unit", "counts": [
            {"alias": "sma", "column": "prev_bucket", "op": "in", "value": ["SMA-1", "SMA-2"]}],
            "sums": [], "metrics": [], "sort_asc": False}
        out, err = execute_aggregation(_agg_df(), spec)
        assert {r["Unit"]: int(r["sma"]) for _, r in out.iterrows()} == {"A": 0, "B": 1}

    def test_metric_list_sort_regression(self):
        # Regression: sorting must use the metrics-list label, not the legacy metric_label.
        out, err = execute_aggregation(_reduction_df(), _REDUCTION_SPEC)
        assert err == ""
        assert list(out["Unit"]) == ["B", "C", "A"]          # 100, 66.67, 50 descending
        assert out.iloc[0]["NPA Reduction %"] == 100.0

    def test_limit_keeps_top_n_after_sort(self):
        spec = {**_REDUCTION_SPEC, "limit": 2}
        out, err = execute_aggregation(_reduction_df(), spec)
        assert err == ""
        assert list(out["Unit"]) == ["B", "C"]               # A dropped by limit

    def test_having_filters_groups(self):
        spec = {**_REDUCTION_SPEC, "having": [{"alias": "npa_prev", "op": ">=", "value": 2}]}
        out, err = execute_aggregation(_reduction_df(), spec)
        assert err == ""
        assert set(out["Unit"]) == {"A", "C"}                # B has npa_prev=1, filtered out

    def test_multi_column_group_by_combines_label(self):
        df = pd.DataFrame({"MNT NAME": ["X", "X", "Y"], "Unit": ["A", "A", "A"],
                           "curr_bucket": ["NPA", "STD", "NPA"]})
        spec = {"group_by": ["MNT NAME", "Unit"], "counts": [
            {"alias": "npa", "column": "curr_bucket", "value": "NPA"}],
            "sums": [], "metrics": [], "sort_asc": False}
        out, err = execute_aggregation(df, spec)
        assert err == ""
        assert "MNT NAME (Unit)" in out.columns
        assert out.iloc[0]["MNT NAME (Unit)"] in {"X (A)", "Y (A)"}

    def test_missing_group_by_column_errors(self):
        out, err = execute_aggregation(_agg_df(), {"group_by": "Nope", "counts": [], "sums": []})
        assert out.empty and err


class TestExecuteFilters:
    def test_basic_filter_and_display(self):
        df = _agg_df().assign(**{"Cust Name": ["a", "b", "c", "d"], "RegionName": ["W"] * 4,
                                 "Cust Mob No": ["1", "2", "3", "4"]})
        parsed = {"conditions": [{"column": "curr_bucket", "op": "==", "value": "NPA"}],
                  "sort_by": "SOH", "sort_asc": False}
        out, err = execute_filters(df, parsed)
        assert err == ""
        assert len(out) == 3                                  # 3 NPA rows
        assert list(out["SOH"]) == [300, 200, 100]           # sorted desc

    def test_no_match_returns_message(self):
        parsed = {"conditions": [{"column": "curr_bucket", "op": "==", "value": "ZZZ"}]}
        out, err = execute_filters(_agg_df(), parsed)
        assert out.empty and err


class TestValidators:
    def test_validate_aggregation_spec_flags_all(self):
        errs = validate_aggregation_spec(
            {"group_by": "NOPE",
             "counts": [{"alias": "n", "column": "ghost", "value": "x"}],
             "metrics": [{"expr": "n / total", "label": "m"}]},
            ["Unit", "curr_bucket"])
        assert any("NOPE" in e for e in errs)
        assert any("ghost" in e for e in errs)
        assert any("total" in e for e in errs)               # undefined metric alias

    def test_validate_aggregation_spec_ok(self):
        assert validate_aggregation_spec(_REDUCTION_SPEC, ["Unit", "prev_bucket", "curr_bucket"]) == []

    def test_validate_filter_spec(self):
        errs = validate_filter_spec({"conditions": [{"column": "ghost", "op": "==", "value": 1}]}, ["Unit"])
        assert any("ghost" in e for e in errs)
