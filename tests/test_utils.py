import io

import numpy as np
import pandas as pd
import pytest

from utils import assign_buckets, apply_filters, compute_metrics, fmt_value, to_num, account_count, is_yes, clean_mobile, REQUIRED_COLS, load_and_validate, build_html_export
from helpers import make_df


def _build_upload(overrides: dict, n: int = 3) -> io.BytesIO:
    """Minimal valid LCC-shaped upload (every REQUIRED_COLS column present) as an
    in-memory .xlsx, for testing load_and_validate's file-parsing path directly."""
    data = {c: ["x"] * n for c in REQUIRED_COLS}
    data.update({
        "Loan No": [f"L{i}" for i in range(n)],
        "Arrears / EMI": [0.0] * n,
        "Month Receipt Amount": [100.0] * n,
        "Month Collection (Excluding Reserve Collection)": [100.0] * n,
        "Net Collection Demand Inst+Exp+BC": [100.0] * n,
        "POS": [1000.0] * n,
        "LCC%": [100.0] * n,
        "Closing Arrears": [0.0] * n,
        "Month Due-Inst": [100.0] * n,
        "Month Due-Exp": [0.0] * n,
        "Total Cum Collection": [1000.0] * n,
        "Strike": ["Y"] * n,
        "Due Dt": [5] * n,
    })
    data.update(overrides)
    df = pd.DataFrame(data)
    buf = io.BytesIO()
    df.to_excel(buf, index=False, engine="openpyxl")
    buf.seek(0)
    buf.name = "test.xlsx"
    return buf


class TestLoadAndValidateSerialDates:
    """Regression: a single out-of-range Excel serial date value used to crash the
    ENTIRE upload (pandas.errors.OutOfBoundsDatetime) instead of just nulling that
    one row - a bad cell in one loan's Ag_Date took down every user's session on a
    shared Streamlit deployment. load_and_validate.__wrapped__ bypasses the
    @st.cache_data decorator so this can be called directly without a live session."""

    @pytest.mark.parametrize("serial,expected", [
        (999999999999, None),                  # overflows OutOfBoundsDatetime guard
        (-5,            None),                  # negative serial
        (45000,         pd.Timestamp("2023-03-15")),  # valid, Excel epoch 1899-12-30
    ], ids=["overflow", "negative", "valid"])
    def test_serial_value_handling(self, serial, expected):
        buf = _build_upload({"Ag_Date": [45000, serial, 45500]})
        result_df, errs = load_and_validate.__wrapped__(buf)
        assert errs == []
        # Neighboring good rows must never be collateral damage from one bad cell.
        assert result_df["Ag_Date"].iloc[0] == pd.Timestamp("2023-03-15")
        assert pd.notna(result_df["Ag_Date"].iloc[2])
        if expected is None:
            assert pd.isna(result_df["Ag_Date"].iloc[1])
        else:
            assert result_df["Ag_Date"].iloc[1] == expected

    def test_object_dtype_column_of_mostly_numeric_serials_is_still_parsed_as_dates(self):
        # Regression: pyxlsb can hand back an "object" dtype column (one stray
        # blank/string cell upgrades the whole column from int/float to object)
        # that's still 99%+ real Excel serial dates. is_numeric_dtype(col) alone
        # was False for that column, so it fell into pd.to_datetime(), which
        # doesn't know these are day-counts and silently reinterprets a bare
        # number as NANOSECONDS SINCE 1970 -- every date in the column came out
        # wrong (~1970-01-01), not just the one bad cell.
        buf = _build_upload({"ParentLDueDate": pd.array([45000, 45500, None], dtype=object)})
        result_df, errs = load_and_validate.__wrapped__(buf)
        assert errs == []
        assert result_df["ParentLDueDate"].iloc[0] == pd.Timestamp("2023-03-15")
        assert result_df["ParentLDueDate"].dt.year.iloc[0] != 1970

    def test_implausible_date_becomes_nat_even_within_timestamp_bounds(self):
        # Regression: a rupee amount (e.g. 150000) that ends up in a date column
        # in the source LCC extract is numerically small enough to not overflow
        # pandas' Timestamp range, so it silently parses into a "valid" but
        # nonsense date (e.g. year 2170) instead of erroring. That's dangerous
        # because Last Receipt Date feeds live business logic ("paid this month"
        # AI Query filters compare it directly against a cutoff) - a garbage
        # far-future date makes an account look paid when it isn't.
        buf = _build_upload({"Last Receipt Date": [45000, 150000, 45500]})
        result_df, errs = load_and_validate.__wrapped__(buf)
        assert errs == []
        assert pd.isna(result_df["Last Receipt Date"].iloc[1])
        assert pd.notna(result_df["Last Receipt Date"].iloc[0])

    def test_mnt_name_normalized_case_and_whitespace_at_load(self):
        # MNT NAME is manually retyped each month (not a controlled vocabulary), so
        # normalize once here rather than in every downstream consumer that groups by it.
        buf = _build_upload({"MNT NAME": ["Sunil Waghmare", " SUNIL WAGHMARE ", "Raj"]})
        result_df, errs = load_and_validate.__wrapped__(buf)
        assert errs == []
        assert result_df["MNT NAME"].iloc[0] == "SUNIL WAGHMARE"
        assert result_df["MNT NAME"].iloc[1] == "SUNIL WAGHMARE"

    def test_mnt_name_blank_stays_null_not_literal_nan_string(self):
        buf = _build_upload({"MNT NAME": ["Raj", None, "Sam"]})
        result_df, errs = load_and_validate.__wrapped__(buf)
        assert errs == []
        assert pd.isna(result_df["MNT NAME"].iloc[1])


class TestLoadAndValidateDuplicateLoanNos:
    """Regression: duplicate Loan Nos were dropped with zero count surfaced
    anywhere - a real data-quality signal (why does the extract have dupes?)
    thrown away silently. The count now rides along on df.attrs rather than
    the errs list, since errs is treated as fatal-per-file by _load_and_concat."""

    def test_no_duplicates_reports_zero(self):
        buf = _build_upload({"Loan No": ["L1", "L2", "L3"]})
        result_df, errs = load_and_validate.__wrapped__(buf)
        assert errs == []
        assert result_df.attrs["dropped_duplicate_loans"] == 0

    def test_duplicate_loan_nos_are_counted_and_first_occurrence_kept(self):
        buf = _build_upload({"Loan No": ["L1", "L1", "L2"], "POS": [1000.0, 9999.0, 1000.0]})
        result_df, errs = load_and_validate.__wrapped__(buf)
        assert errs == []
        assert len(result_df) == 2
        assert result_df.attrs["dropped_duplicate_loans"] == 1
        assert result_df.loc[result_df["Loan No"] == "L1", "POS"].iloc[0] == 1000.0


class TestAssignBuckets:
    """Bucket assignment is the foundation - every downstream calculation depends on it."""

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

    def test_multiple_rows_assigned_independently(self):
        df = pd.DataFrame({
            "Arrears / EMI": [0.0, 1.5, 5.0],
            "POS": [0.0, 0.0, 0.0],
            "Closing Arrears": [0.0, 0.0, 0.0],
        })
        buckets = assign_buckets(df)["curr_bucket"].tolist()
        assert buckets == ["STD", "SMA-1", "NPA"]

    @pytest.mark.parametrize("row,expected_soh", [
        ({"POS": 80_000.0, "Closing Arrears": 20_000.0}, 1_00_000.0),  # SOH = POS + arrears
        ({"Closing Arrears": 5_000.0},                    5_000.0),     # missing POS -> treated as 0
        ({"POS": 5_000.0},                                5_000.0),     # missing arrears -> treated as 0
    ], ids=["pos_plus_arrears", "missing_pos", "missing_arrears"])
    def test_soh_formula_handles_missing_columns(self, row, expected_soh):
        base = {"Arrears / EMI": 0.0}
        df = pd.DataFrame({k: [v] for k, v in {**base, **row}.items()})
        assert assign_buckets(df)["SOH"].iloc[0] == expected_soh


class TestApplyFilters:
    def setup_method(self):
        self.df = make_df([
            {"Loan No": "L001", "RegionName": "WEST",  "Unit": "MAHAD",   "Loan Status": "Active"},
            {"Loan No": "L002", "RegionName": "WEST",  "Unit": "PUNE",    "Loan Status": "Active"},
            {"Loan No": "L003", "RegionName": "SOUTH", "Unit": "CHENNAI", "Loan Status": "Closed"},
        ])

    @pytest.mark.parametrize("region,branch,status,expected_loans", [
        ("All",   "All",     "All",    {"L001", "L002", "L003"}),  # no-op
        ("WEST",  "All",     "All",    {"L001", "L002"}),           # region only
        ("All",   "MAHAD",   "All",    {"L001"}),                   # branch only
        ("All",   "All",     "Closed", {"L003"}),                   # status only
        ("WEST",  "PUNE",    "All",    {"L002"}),                   # combined region+branch
        ("NORTH", "All",     "All",    set()),                      # no match -> empty, not an error
    ], ids=["all", "region", "branch", "status", "combined", "no_match"])
    def test_filter_combinations(self, region, branch, status, expected_loans):
        result = apply_filters(self.df.copy(), region, branch, status)
        assert set(result["Loan No"]) == expected_loans


class TestComputeMetrics:
    @pytest.mark.parametrize("demand,collection,expected_pct", [
        (10_000.0, 7_500.0, 75.0),  # normal ratio
        (0.0,      5_000.0, 0.0),   # zero demand -> 0, not a division error
    ], ids=["normal", "zero_demand"])
    def test_collection_pct(self, demand, collection, expected_pct):
        df = make_df([{
            "Net Collection Demand Inst+Exp+BC": demand,
            "Month Collection (Excluding Reserve Collection)": collection,
        }])
        metrics = compute_metrics(df, make_df([]))
        assert metrics["Collection %"][0] == expected_pct

    @pytest.mark.parametrize("buckets,expected_pct", [
        (["STD", "NPA", "NPA"], pytest.approx(66.67, abs=0.01)),  # 2/3, unique loans
        (["STD", "SMA-1"],      0.0),                              # no NPA at all
    ], ids=["mixed", "none"])
    def test_npa_pct(self, buckets, expected_pct):
        df = make_df([{"Loan No": f"L{i}", "curr_bucket": b} for i, b in enumerate(buckets)])
        metrics = compute_metrics(df, make_df([]))
        assert metrics["NPA %"][0] == expected_pct

    def test_strike_pct_y_over_y_plus_n(self):
        df = make_df([
            {"Loan No": "L001", "Strike": "Y"},
            {"Loan No": "L002", "Strike": "Y"},
            {"Loan No": "L003", "Strike": "N"},
            {"Loan No": "L004", "Strike": "N"},
        ])
        metrics = compute_metrics(df, make_df([]))
        assert metrics["Strike %"][0] == 50.0

    @pytest.mark.parametrize("curr_coll,prev_coll,check", [
        (10_000.0, 8_000.0,  lambda mom: mom == 25.0),   # curr 100% vs prev 80% -> +25%
        (6_000.0,  10_000.0, lambda mom: mom < 0),        # curr below prev -> negative
    ], ids=["improved", "worsened"])
    def test_mom_reflects_direction_of_change(self, curr_coll, prev_coll, check):
        curr = make_df([{"Net Collection Demand Inst+Exp+BC": 10_000.0, "Month Collection (Excluding Reserve Collection)": curr_coll}])
        prev = make_df([{"Net Collection Demand Inst+Exp+BC": 10_000.0, "Month Collection (Excluding Reserve Collection)": prev_coll}])
        metrics = compute_metrics(curr, prev)
        assert check(metrics["Collection %"][1])

    def test_empty_prev_gives_zero_mom(self):
        df = make_df([{"Month Collection (Excluding Reserve Collection)": 5_000.0}])
        metrics = compute_metrics(df, make_df([]))
        assert metrics["Collection %"][1] == 0.0

    def test_result_shape_is_stable(self):
        # Every consumer (dashboard cards, report portfolio_health section)
        # destructures metrics[key] as a (value, mom) tuple for a fixed key set --
        # a missing key or a bare scalar instead of a tuple breaks every consumer.
        df = make_df([{"Loan No": "L001"}])
        metrics = compute_metrics(df, make_df([]))
        expected_keys = {
            "Month Demand", "Total Collection", "Collection %", "Strike %",
            "NPA %", "Hard Bucket %", "SMA-2 %", "Count", "SOH", "LCC%", "CMD %",
        }
        assert set(metrics.keys()) == expected_keys
        for key, val in metrics.items():
            assert isinstance(val, tuple) and len(val) == 2, f"{key} is not a (value, mom) tuple"

    def test_lcc_pct_uses_cum_coll_inst_exp_not_total_cum_collection(self):
        # Deliberately make "Total Cum Collection" (a broader figure, includes BC)
        # differ from "Cum Coll (Inst+Exp)" -- the documented LCC% numerator
        # (agents/domain_expert.py, registry/ontology.py's lcc_pct METRIC). LCC%
        # must be computed off Cum Coll (Inst+Exp), never Total Cum Collection.
        df = make_df([{
            "Cum Coll (Inst+Exp)":   80_000.0,
            "Total Cum Collection": 150_000.0,
            "Cum Due-Inst":          80_000.0,
            "Cum Due-Exp":           20_000.0,
        }])
        metrics = compute_metrics(df, make_df([]))
        # 80,000 / (80,000 + 20,000) * 100 = 80.0 -- NOT 150.
        assert metrics["LCC%"][0] == 80.0

    def test_lcc_pct_capped_at_100(self):
        # A customer who has paid ahead of cumulative dues can push the raw ratio
        # above 100%; the documented definition and the AI Query path's lcc_pct
        # METRIC (cap=100) both cap it, so the dashboard must too.
        df = make_df([{
            "Cum Coll (Inst+Exp)": 150_000.0,
            "Cum Due-Inst":         80_000.0,
            "Cum Due-Exp":          20_000.0,
        }])
        metrics = compute_metrics(df, make_df([]))
        assert metrics["LCC%"][0] == 100.0

    def test_full_metrics_computed_correctly_for_a_mixed_portfolio(self):
        # Dense integration test: exercises Collection%, NPA%, Strike%, Count and
        # MoM together against one realistic mixed portfolio (2 NPA of 4 loans,
        # 2 Strike=Y of 4), rather than isolating each metric behind its own
        # single-purpose fixture. Expected values hand-verified against the
        # underlying formulas, not just "whatever the code currently returns".
        curr = make_df([
            {"Loan No": "L001", "curr_bucket": "STD", "Strike": "Y",
             "Net Collection Demand Inst+Exp+BC": 5_000.0, "Month Collection (Excluding Reserve Collection)": 5_000.0},
            {"Loan No": "L002", "curr_bucket": "NPA", "Strike": "N",
             "Net Collection Demand Inst+Exp+BC": 5_000.0, "Month Collection (Excluding Reserve Collection)": 2_500.0},
            {"Loan No": "L003", "curr_bucket": "NPA", "Strike": "N",
             "Net Collection Demand Inst+Exp+BC": 0.0, "Month Collection (Excluding Reserve Collection)": 0.0},
            {"Loan No": "L004", "curr_bucket": "SMA-1", "Strike": "Y",
             "Net Collection Demand Inst+Exp+BC": 10_000.0, "Month Collection (Excluding Reserve Collection)": 10_000.0},
        ])
        prev = make_df([
            {"Net Collection Demand Inst+Exp+BC": 20_000.0, "Month Collection (Excluding Reserve Collection)": 15_000.0},
        ])
        metrics = compute_metrics(curr, prev)
        assert metrics["Count"][0] == 4
        assert metrics["Month Demand"][0] == 20_000.0        # 5000+5000+0+10000
        assert metrics["Total Collection"][0] == 17_500.0    # 5000+2500+0+10000
        assert metrics["Collection %"][0] == 87.5            # 17500/20000
        assert metrics["NPA %"][0] == 50.0                   # 2 of 4
        assert metrics["Strike %"][0] == 50.0                # 2 Y of 4 valid
        # prev collection% = 15000/20000 = 75%; MoM = (87.5-75)/75*100
        assert metrics["Collection %"][1] == pytest.approx(16.67, abs=0.01)


class TestFmtValue:
    @pytest.mark.parametrize("val,expected", [
        (1_00_00_000, "₹1.00Cr"),
        (1_00_000,    "₹1.00L"),
        (999,         "₹999"),
    ], ids=["crore", "lakh", "below_lakh"])
    def test_money_formatting_picks_correct_unit(self, val, expected):
        assert fmt_value(val, "money") == expected

    @pytest.mark.parametrize("val,expected", [
        (2_00_000,    "2.00L"),
        (500,         "500"),
    ], ids=["lakh", "small"])
    def test_count_formatting_picks_correct_unit(self, val, expected):
        assert fmt_value(val, "count") == expected

    def test_count_crore_includes_unit_suffix(self):
        assert "Cr" in fmt_value(1_00_00_000, "count")

    def test_pct_two_decimal_places(self):
        assert fmt_value(72.5, "pct") == "72.50"


class TestToNum:
    """Shared numeric-coercion helper used across analysis/ and smart_alerts.py."""

    def test_coerces_valid_and_invalid_strings(self):
        df = pd.DataFrame({"x": ["10", "20.5", "bad", None]})
        result = to_num(df, "x")
        assert result.tolist()[:2] == [10.0, 20.5]
        assert pd.isna(result.iloc[2]) and pd.isna(result.iloc[3])

    @pytest.mark.parametrize("fill,expected", [
        (None, [True, True, True]),   # NaN series, index-aligned
        (0,    [0, 0, 0]),             # filled series
    ], ids=["no_fill", "with_fill"])
    def test_missing_column_returns_series_aligned_to_df(self, fill, expected):
        df = pd.DataFrame({"other": [1, 2, 3]})
        result = to_num(df, "missing", fill=fill)
        assert len(result) == len(df)
        assert result.index.equals(df.index)
        if fill is None:
            assert result.isna().tolist() == expected
        else:
            assert result.tolist() == expected

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
    @pytest.mark.parametrize("df,kwargs,expected", [
        (pd.DataFrame({"Loan No": ["L1", "L1", "L2"]}), {}, 2),                          # distinct loans
        (pd.DataFrame({"other": [1, 2, 3]}), {}, 3),                                      # column missing -> row count
        (pd.DataFrame({"Cust Mob No": ["A", "A", "B", "C"]}), {"col": "Cust Mob No"}, 3),  # custom column
    ], ids=["distinct_loans", "missing_column", "custom_column"])
    def test_account_count(self, df, kwargs, expected):
        assert account_count(df, **kwargs) == expected


class TestIsYes:
    def test_matches_y_case_and_whitespace_insensitive(self):
        df = pd.DataFrame({"flag": [" y", "Y", "n", "N", "yes", None]})
        result = is_yes(df, "flag")
        assert result.tolist() == [True, True, False, False, True, False]

    def test_matches_yes_spelled_out_full_word(self):
        # Some monthly extracts spell the flag as "Yes"/"No" instead of "Y"/"N"
        # (e.g. Non Starter column) -- both spellings must count as true.
        df = pd.DataFrame({"flag": ["Yes", "YES", " yes ", "No", "NO"]})
        result = is_yes(df, "flag")
        assert result.tolist() == [True, True, True, False, False]

    def test_missing_column_returns_all_false_aligned_to_df(self):
        df = pd.DataFrame({"other": [1, 2, 3]})
        result = is_yes(df, "missing")
        assert result.tolist() == [False, False, False]
        assert result.index.equals(df.index)


class TestCleanMobile:
    """Regression guard: any blank mobile number in the Excel file upgrades the
    whole column to float64, which renders every number as '9876543210.0'."""

    @pytest.mark.parametrize("series,expected", [
        (pd.Series([9876543210.0, 9123456780.0, float("nan"), 9988776655.0]),
         ["9876543210", "9123456780", "", "9988776655"]),                  # float-upgraded column
        (pd.Series(["9876543210", "9123456780"]), ["9876543210", "9123456780"]),  # already clean strings
        (pd.Series([9876543210, 9123456780]), ["9876543210", "9123456780"]),       # clean int column
    ], ids=["float_upgraded", "clean_strings", "clean_ints"])
    def test_clean_mobile(self, series, expected):
        assert clean_mobile(series).tolist() == expected

    def test_nan_becomes_empty_string_not_literal_nan(self):
        result = clean_mobile(pd.Series([9876543210.0, float("nan")]))
        assert result.iloc[1] == ""
        assert "nan" not in result.tolist()


class TestBuildHtmlExportEscaping:
    """Regression: build_html_export interpolates filter values and the
    scorecard's free-text 'Executive (Branch)' column into raw f-string HTML,
    same bug class already fixed in report_agent/nodes/report_builder.py.
    A manually-entered name like "RAJESH & SONS <TRANSPORT>" must not break
    the surrounding table markup, and the file is offered as a raw download."""

    def _metrics(self):
        keys = ["Month Demand", "Total Collection", "Collection %", "Strike %",
                 "NPA %", "Hard Bucket %", "Count", "SOH", "LCC%", "CMD %"]
        return {k: (1.0, 0.0) for k in keys}

    def test_filter_values_are_escaped(self):
        import plotly.graph_objects as go

        html_out = build_html_export(
            make_df([]), make_df([]), self._metrics(),
            go.Figure(), go.Figure(), go.Figure(),
            filters={"Branch": "RAJESH & SONS <TRANSPORT>"},
            curr_month="Jan-2026",
        )
        assert "RAJESH & SONS <TRANSPORT>" not in html_out
        assert "RAJESH &amp; SONS &lt;TRANSPORT&gt;" in html_out

    def test_scorecard_executive_name_is_escaped(self):
        import plotly.graph_objects as go

        scorecard_df = pd.DataFrame([{
            "Executive (Branch)": "A & B <script>alert(1)</script>",
            "Accounts": 10, "Collection %": 95.0, "Strike Rate %": 80.0,
            "Tier": "top",
        }])
        html_out = build_html_export(
            make_df([]), make_df([]), self._metrics(),
            go.Figure(), go.Figure(), go.Figure(),
            filters={}, scorecard_df=scorecard_df,
        )
        assert "<script>alert(1)</script>" not in html_out
        assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html_out
        # Table structure survives - matching open/close tags around the row.
        assert html_out.count("<tr") == html_out.count("</tr>")
