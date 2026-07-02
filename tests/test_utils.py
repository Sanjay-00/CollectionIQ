import numpy as np
import pandas as pd
import pytest

from utils import assign_buckets, apply_filters, compute_metrics, fmt_value, to_num, account_count, is_yes, clean_mobile
from helpers import make_df


class TestAssignBuckets:
    """Bucket assignment is the foundation  -  every downstream calculation depends on it."""

    @pytest.mark.parametrize("arrears_emi,expected", [
        (np.nan,  "NA"),       # unknown → NA
        (-1.0,    "STD"),      # negative → STD
        (0.0,     "STD"),      # exactly zero → STD
        (0.001,   "1-30 DPD"), # just above 0
        (0.999,   "1-30 DPD"), # just below 1
        (1.0,     "SMA-1"),    # exactly 1
        (1.999,   "SMA-1"),    # just below 2
        (2.0,     "SMA-2"),    # exactly 2
        (2.999,   "SMA-2"),    # just below 3
        (3.0,     "NPA"),      # exactly 3
        (10.0,    "NPA"),      # deep NPA
    ])
    def test_bucket_boundaries(self, arrears_emi, expected):
        df = pd.DataFrame({
            "Arrears / EMI": [arrears_emi],
            "POS": [1_00_000.0],
            "Closing Arrears": [0.0],
        })
        result = assign_buckets(df)
        assert result["curr_bucket"].iloc[0] == expected

    def test_soh_equals_pos_plus_closing_arrears(self):
        df = pd.DataFrame({
            "Arrears / EMI": [0.0],
            "POS": [80_000.0],
            "Closing Arrears": [20_000.0],
        })
        assert assign_buckets(df)["SOH"].iloc[0] == 1_00_000.0

    def test_missing_pos_column_treated_as_zero(self):
        df = pd.DataFrame({
            "Arrears / EMI": [0.0],
            "Closing Arrears": [5_000.0],
        })
        assert assign_buckets(df)["SOH"].iloc[0] == 5_000.0

    def test_multiple_rows_assigned_independently(self):
        df = pd.DataFrame({
            "Arrears / EMI": [0.0, 1.5, 5.0],
            "POS": [0.0, 0.0, 0.0],
            "Closing Arrears": [0.0, 0.0, 0.0],
        })
        buckets = assign_buckets(df)["curr_bucket"].tolist()
        assert buckets == ["STD", "SMA-1", "NPA"]


class TestApplyFilters:
    def setup_method(self):
        self.df = make_df([
            {"Loan No": "L001", "RegionName": "WEST",  "Unit": "MAHAD",   "Loan Status": "Active"},
            {"Loan No": "L002", "RegionName": "WEST",  "Unit": "PUNE",    "Loan Status": "Active"},
            {"Loan No": "L003", "RegionName": "SOUTH", "Unit": "CHENNAI", "Loan Status": "Closed"},
        ])

    def test_all_filters_pass_through_all_rows(self):
        assert len(apply_filters(self.df.copy(), "All", "All", "All")) == 3

    def test_region_filter_keeps_matching_rows(self):
        result = apply_filters(self.df.copy(), "WEST", "All", "All")
        assert len(result) == 2
        assert set(result["RegionName"]) == {"WEST"}

    def test_branch_filter(self):
        result = apply_filters(self.df.copy(), "All", "MAHAD", "All")
        assert len(result) == 1
        assert result["Loan No"].iloc[0] == "L001"

    def test_status_filter(self):
        result = apply_filters(self.df.copy(), "All", "All", "Closed")
        assert len(result) == 1
        assert result["Loan No"].iloc[0] == "L003"

    def test_combined_region_and_branch(self):
        result = apply_filters(self.df.copy(), "WEST", "PUNE", "All")
        assert len(result) == 1
        assert result["Loan No"].iloc[0] == "L002"

    def test_no_match_returns_empty_dataframe(self):
        result = apply_filters(self.df.copy(), "NORTH", "All", "All")
        assert len(result) == 0


class TestComputeMetrics:
    def test_collection_pct(self):
        df = make_df([{
            "Net Collection Demand Inst+Exp+BC":                 10_000.0,
            "Month Collection (Excluding Reserve Collection)":    7_500.0,
        }])
        metrics = compute_metrics(df, make_df([]))
        assert metrics["Collection %"][0] == 75.0

    def test_collection_pct_zero_demand(self):
        df = make_df([{
            "Net Collection Demand Inst+Exp+BC":                0.0,
            "Month Collection (Excluding Reserve Collection)":  5_000.0,
        }])
        metrics = compute_metrics(df, make_df([]))
        assert metrics["Collection %"][0] == 0.0

    def test_npa_pct_counts_unique_loans(self):
        df = make_df([
            {"Loan No": "L001", "curr_bucket": "STD"},
            {"Loan No": "L002", "curr_bucket": "NPA"},
            {"Loan No": "L003", "curr_bucket": "NPA"},
        ])
        metrics = compute_metrics(df, make_df([]))
        # 2 NPA / 3 total = 66.67 %
        assert metrics["NPA %"][0] == pytest.approx(66.67, abs=0.01)

    def test_npa_pct_zero_when_no_npa(self):
        df = make_df([
            {"Loan No": "L001", "curr_bucket": "STD"},
            {"Loan No": "L002", "curr_bucket": "SMA-1"},
        ])
        metrics = compute_metrics(df, make_df([]))
        assert metrics["NPA %"][0] == 0.0

    def test_strike_pct_y_over_y_plus_n(self):
        df = make_df([
            {"Loan No": "L001", "Strike": "Y"},
            {"Loan No": "L002", "Strike": "Y"},
            {"Loan No": "L003", "Strike": "N"},
            {"Loan No": "L004", "Strike": "N"},
        ])
        metrics = compute_metrics(df, make_df([]))
        assert metrics["Strike %"][0] == 50.0

    def test_mom_positive_when_curr_exceeds_prev(self):
        curr = make_df([{
            "Net Collection Demand Inst+Exp+BC":                10_000.0,
            "Month Collection (Excluding Reserve Collection)":  10_000.0,
        }])
        prev = make_df([{
            "Net Collection Demand Inst+Exp+BC":                10_000.0,
            "Month Collection (Excluding Reserve Collection)":   8_000.0,
        }])
        metrics = compute_metrics(curr, prev)
        # curr = 100%, prev = 80%, MoM = (100-80)/80*100 = 25%
        assert metrics["Collection %"][1] == 25.0

    def test_mom_negative_when_curr_below_prev(self):
        curr = make_df([{
            "Net Collection Demand Inst+Exp+BC":                10_000.0,
            "Month Collection (Excluding Reserve Collection)":   6_000.0,
        }])
        prev = make_df([{
            "Net Collection Demand Inst+Exp+BC":                10_000.0,
            "Month Collection (Excluding Reserve Collection)":  10_000.0,
        }])
        metrics = compute_metrics(curr, prev)
        assert metrics["Collection %"][1] < 0

    def test_empty_prev_gives_zero_mom(self):
        df = make_df([{
            "Month Collection (Excluding Reserve Collection)": 5_000.0,
        }])
        metrics = compute_metrics(df, make_df([]))
        assert metrics["Collection %"][1] == 0.0

    def test_result_has_all_expected_keys(self):
        df = make_df([{"Loan No": "L001"}])
        metrics = compute_metrics(df, make_df([]))
        expected_keys = {
            "Month Demand", "Total Collection", "Collection %", "Strike %",
            "NPA %", "Hard Bucket %", "SMA-2 %", "Count", "SOH", "LCC%", "CMD %",
        }
        assert set(metrics.keys()) == expected_keys

    def test_each_metric_is_value_mom_tuple(self):
        df = make_df([{"Loan No": "L001"}])
        metrics = compute_metrics(df, make_df([]))
        for key, val in metrics.items():
            assert isinstance(val, tuple) and len(val) == 2, f"{key} is not a (value, mom) tuple"


class TestFmtValue:
    def test_money_crore(self):
        assert fmt_value(1_00_00_000, "money") == "₹1.00Cr"

    def test_money_lakh(self):
        assert fmt_value(1_00_000, "money") == "₹1.00L"

    def test_money_below_lakh(self):
        assert fmt_value(999, "money") == "₹999"

    def test_pct_two_decimal_places(self):
        assert fmt_value(72.5, "pct") == "72.50"

    def test_count_lakh(self):
        assert fmt_value(2_00_000, "count") == "2.00L"

    def test_count_small(self):
        assert fmt_value(500, "count") == "500"

    def test_count_crore(self):
        result = fmt_value(1_00_00_000, "count")
        assert "Cr" in result


class TestToNum:
    """Shared numeric-coercion helper used across analysis/ and smart_alerts.py."""

    def test_coerces_valid_and_invalid_strings(self):
        df = pd.DataFrame({"x": ["10", "20.5", "bad", None]})
        result = to_num(df, "x")
        assert result.tolist()[:2] == [10.0, 20.5]
        assert pd.isna(result.iloc[2]) and pd.isna(result.iloc[3])

    def test_missing_column_returns_nan_series_aligned_to_df(self):
        df = pd.DataFrame({"other": [1, 2, 3]})
        result = to_num(df, "missing")
        assert len(result) == len(df)
        assert result.index.equals(df.index)
        assert result.isna().all()

    def test_missing_column_with_fill_returns_filled_series_aligned_to_df(self):
        df = pd.DataFrame({"other": [1, 2, 3]})
        result = to_num(df, "missing", fill=0)
        assert result.tolist() == [0, 0, 0]

    def test_present_column_with_fill_fills_only_invalid_values(self):
        df = pd.DataFrame({"x": ["10", "bad", None]})
        result = to_num(df, "x", fill=0)
        assert result.tolist() == [10.0, 0.0, 0.0]

    def test_result_stays_usable_in_boolean_mask_against_df(self):
        # Regression guard: a naive `df.get(col, pd.Series(dtype=float))` fallback
        # returns an EMPTY series when col is missing, which silently breaks any
        # boolean mask built against it (misaligned index -> mask always empty).
        df = pd.DataFrame({"other": [1, 2, 3]})
        mask = to_num(df, "missing", fill=0) > 5
        assert mask.tolist() == [False, False, False]
        assert len(df[mask]) == 0  # doesn't raise, just filters everything out


class TestAccountCount:
    def test_counts_distinct_loan_numbers(self):
        df = pd.DataFrame({"Loan No": ["L1", "L1", "L2"]})
        assert account_count(df) == 2

    def test_falls_back_to_row_count_when_column_missing(self):
        df = pd.DataFrame({"other": [1, 2, 3]})
        assert account_count(df) == 3

    def test_custom_column_name(self):
        df = pd.DataFrame({"Cust Mob No": ["A", "A", "B", "C"]})
        assert account_count(df, col="Cust Mob No") == 3


class TestIsYes:
    def test_matches_y_case_and_whitespace_insensitive(self):
        df = pd.DataFrame({"flag": [" y", "Y", "n", "N", "yes", None]})
        result = is_yes(df, "flag")
        assert result.tolist() == [True, True, False, False, False, False]

    def test_missing_column_returns_all_false_aligned_to_df(self):
        df = pd.DataFrame({"other": [1, 2, 3]})
        result = is_yes(df, "missing")
        assert result.tolist() == [False, False, False]
        assert result.index.equals(df.index)


class TestCleanMobile:
    """Regression guard: any blank mobile number in the Excel file upgrades the
    whole column to float64, which renders every number as '9876543210.0'."""

    def test_strips_decimal_artifact_from_float_upgraded_column(self):
        # This is exactly what pandas produces when one row is blank (NaN).
        series = pd.Series([9876543210.0, 9123456780.0, float("nan"), 9988776655.0])
        result = clean_mobile(series)
        assert result.tolist() == ["9876543210", "9123456780", "", "9988776655"]

    def test_already_clean_strings_untouched(self):
        series = pd.Series(["9876543210", "9123456780"])
        assert clean_mobile(series).tolist() == ["9876543210", "9123456780"]

    def test_clean_int_column_untouched(self):
        series = pd.Series([9876543210, 9123456780])
        assert clean_mobile(series).tolist() == ["9876543210", "9123456780"]

    def test_nan_becomes_empty_string_not_literal_nan(self):
        series = pd.Series([9876543210.0, float("nan")])
        result = clean_mobile(series)
        assert result.iloc[1] == ""
        assert "nan" not in result.tolist()
