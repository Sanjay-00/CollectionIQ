import io

import numpy as np
import pandas as pd
import pytest

from utils import assign_buckets, apply_filters, compute_metrics, fmt_value, to_num, account_count, is_yes, clean_mobile, REQUIRED_COLS, load_and_validate, build_html_export, compute_overdue_demand_pct, normalize_truncated_names
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


class TestLoadAndValidateColumnOrder:
    """Regression: different regional files (or a re-exported sheet) can hand
    back the same ~85 columns in a different order. Without reordering to the
    canonical template sequence, an AI Query result that includes every column
    (e.g. "give me all cases with all columns") rendered a visibly jumbled,
    upload-dependent column order instead of the familiar template layout."""

    def test_shuffled_upload_is_restored_to_canonical_order(self):
        upload = _build_upload({})
        # Deliberately scramble the column order before it's ever read by
        # load_and_validate -- a real regional file could easily have its own
        # internal column order that doesn't match REQUIRED_COLS at all.
        raw = pd.read_excel(upload, engine="openpyxl")
        shuffled_cols = list(reversed(raw.columns))
        raw = raw[shuffled_cols]
        buf = io.BytesIO()
        raw.to_excel(buf, index=False, engine="openpyxl")
        buf.seek(0)
        buf.name = "shuffled.xlsx"

        df, errs = load_and_validate.__wrapped__(buf)
        assert errs == []
        canon_subset = [c for c in REQUIRED_COLS if c in df.columns]
        actual_subset = [c for c in df.columns if c in REQUIRED_COLS]
        assert actual_subset == canon_subset

    def test_unknown_extra_columns_kept_but_appended_after_known_ones(self):
        upload = _build_upload({"Some Legacy Column": ["x", "x", "x"]})
        df, errs = load_and_validate.__wrapped__(upload)
        assert errs == []
        assert "Some Legacy Column" in df.columns
        known_cols = [c for c in df.columns if c in REQUIRED_COLS]
        extra_pos = list(df.columns).index("Some Legacy Column")
        assert extra_pos > list(df.columns).index(known_cols[-1])


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

    def test_already_datetime64_column_is_not_misread_as_excel_serials(self):
        # Regression: openpyxl (unlike pyxlsb) hands back REAL Timestamps
        # directly for date-formatted .xlsx cells -- a proper datetime64
        # column, not raw serial numbers. pd.to_numeric() on a datetime64
        # column doesn't fail, it silently casts each date to a microsecond
        # -since-epoch integer, which used to read as "100% numeric" and get
        # WRONGLY treated as an Excel serial-date column, reinterpreted as
        # "days since 1899" -- a value in the trillions, instantly out of
        # range, nulled to NaT. Confirmed on a real file: 100% of Ag_Date
        # (13,205/13,205 rows) silently vanished this way.
        buf = _build_upload({"Ag_Date": [pd.Timestamp("2011-01-25"), pd.Timestamp("2022-09-15"), pd.Timestamp("2023-03-15")]})
        result_df, errs = load_and_validate.__wrapped__(buf)
        assert errs == []
        assert result_df["Ag_Date"].isna().sum() == 0
        assert result_df["Ag_Date"].iloc[0] == pd.Timestamp("2011-01-25")
        assert result_df["Ag_Date"].iloc[1] == pd.Timestamp("2022-09-15")
        assert result_df["Ag_Date"].iloc[2] == pd.Timestamp("2023-03-15")

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


class TestLoadAndValidateMissingOptionalCols:
    """Regression: utils.py's own module docstring claims a missing non
    -critical column (REQUIRED_COLS minus CRITICAL_COLS) 'shows a warning, not
    an error' -- no such warning existed anywhere in the codebase. A missing
    CoLending_Loans/LGL_FLAG/SegmentName was previously indistinguishable from
    a real zero-count answer (e.g. a co-lending query silently reporting 'no
    co-lending accounts' when the column simply isn't in this upload at all)."""

    def test_all_columns_present_reports_none_missing(self):
        buf = _build_upload({})
        result_df, errs = load_and_validate.__wrapped__(buf)
        assert errs == []
        assert result_df.attrs["missing_optional_cols"] == []

    def test_missing_non_critical_columns_are_listed(self):
        data = {c: ["x"] * 3 for c in REQUIRED_COLS if c not in ("CoLending_Loans", "LGL_FLAG")}
        data.update({
            "Loan No": ["L1", "L2", "L3"], "Arrears / EMI": [0.0] * 3,
            "Month Receipt Amount": [100.0] * 3,
            "Month Collection (Excluding Reserve Collection)": [100.0] * 3,
            "Net Collection Demand Inst+Exp+BC": [100.0] * 3,
            "POS": [1000.0] * 3, "LCC%": [100.0] * 3, "Closing Arrears": [0.0] * 3,
            "Month Due-Inst": [100.0] * 3, "Month Due-Exp": [0.0] * 3,
            "Total Cum Collection": [1000.0] * 3, "Strike": ["Y"] * 3, "Due Dt": [5] * 3,
        })
        buf = io.BytesIO()
        pd.DataFrame(data).to_excel(buf, index=False, engine="openpyxl")
        buf.seek(0)
        buf.name = "missing_cols.xlsx"

        result_df, errs = load_and_validate.__wrapped__(buf)
        assert errs == []
        assert "CoLending_Loans" in result_df.attrs["missing_optional_cols"]
        assert "LGL_FLAG" in result_df.attrs["missing_optional_cols"]
        # Never flag a CRITICAL column as merely "optional" -- that path is a
        # hard failure (missing critical column(s)) long before this point.
        assert "Loan No" not in result_df.attrs["missing_optional_cols"]

    def test_missing_critical_column_still_hard_fails_not_just_warns(self):
        data = {c: ["x"] * 3 for c in REQUIRED_COLS if c != "POS"}
        data.update({"Loan No": ["L1", "L2", "L3"], "Arrears / EMI": [0.0] * 3})
        buf = io.BytesIO()
        pd.DataFrame(data).to_excel(buf, index=False, engine="openpyxl")
        buf.seek(0)
        buf.name = "missing_critical.xlsx"

        result_df, errs = load_and_validate.__wrapped__(buf)
        assert result_df is None
        assert errs and "POS" in errs[0]


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

    def test_na_bucket_has_no_severity_score(self):
        # Regression: BUCKET_SCORE used to map "NA" (Arrears/EMI missing/
        # unparseable that period -- not a real delinquency state) to -1, lower
        # than every real bucket. That made a loan going from "NA" (unknown) to
        # STD (the healthiest real bucket) register as "rolled forward"
        # (worsened) purely because -1 < 0 -- confirmed on real production data.
        # "NA" must map to NaN (not comparable), never a real number.
        df = pd.DataFrame({
            "Arrears / EMI": [np.nan],
            "POS": [0.0], "Closing Arrears": [0.0],
        })
        result = assign_buckets(df)
        assert result["curr_bucket"].iloc[0] == "NA"
        assert pd.isna(result["curr_score"].iloc[0])

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


def _overdue_demand_row(
    arrear_opening: float, month_due_inst: float = 0.0, month_due_exp: float = 0.0,
    month_due_bc: float = 0.0, month_due_pc: float = 0.0, paid: float = 0.0,
) -> pd.DataFrame:
    """One-loan DataFrame through the REAL assign_buckets path (not a hand-built
    'Overdue'/'MonthDemandExclPC' column), so these tests exercise the exact same
    code production uses."""
    return assign_buckets(pd.DataFrame({
        "Arrears / EMI": [0.0], "POS": [0.0], "Closing Arrears": [0.0],
        "Arrear Opening": [arrear_opening],
        "Month Due-Inst": [month_due_inst], "Month Due-Exp": [month_due_exp],
        "MONTH DUE (BC)": [month_due_bc], "MONTH DUE PC": [month_due_pc],
        "Month Collection (Excluding Reserve Collection)": [paid],
    }))


class TestOverdueDemandCollectionPct:
    """The overdue-first collection waterfall: a payment clears last month's
    carried-over overdue (Arrear Opening) BEFORE any of it counts against this
    month's own EMI demand -- confirmed against real business examples worked
    through with the user, not invented after the fact."""

    def test_example_1_overdue_cleared_then_partial_demand(self):
        # prev overdue 10k, month demand 20k, paid 20k -> overdue 100%, demand 50%
        df = _overdue_demand_row(arrear_opening=10_000, month_due_inst=20_000, paid=20_000)
        stats = compute_overdue_demand_pct(df)
        assert stats["overdue_pct"] == 100.0
        assert stats["demand_pct"] == 50.0

    def test_example_2_overdue_partial_no_demand_collected(self):
        # prev overdue 40k, month demand 10k, paid 10k -> overdue 25%, demand 0%
        df = _overdue_demand_row(arrear_opening=40_000, month_due_inst=10_000, paid=10_000)
        stats = compute_overdue_demand_pct(df)
        assert stats["overdue_pct"] == 25.0
        assert stats["demand_pct"] == 0.0

    def test_example_3_overdue_cleared_then_60pct_demand(self):
        # prev overdue 10k, month demand 50k, paid 40k -> overdue 100%, demand 60%
        df = _overdue_demand_row(arrear_opening=10_000, month_due_inst=50_000, paid=40_000)
        stats = compute_overdue_demand_pct(df)
        assert stats["overdue_pct"] == 100.0
        assert stats["demand_pct"] == 60.0

    def test_example_4_overpayment_caps_both_at_100(self):
        # prev overdue 20k, month demand 50k, paid 70k+ -> both 100%, never > 100%
        df = _overdue_demand_row(arrear_opening=20_000, month_due_inst=50_000, paid=75_000)
        stats = compute_overdue_demand_pct(df)
        assert stats["overdue_pct"] == 100.0
        assert stats["demand_pct"] == 100.0

    def test_negative_overdue_clipped_both_full_paid(self):
        # Arrear Opening -20k (customer already in credit) clips to 0 -- the
        # whole 60k payment is free to apply to demand. overdue_pct is 100%
        # (nothing outstanding), not 0% and not N/A.
        df = _overdue_demand_row(arrear_opening=-20_000, month_due_inst=60_000, paid=60_000)
        stats = compute_overdue_demand_pct(df)
        assert stats["overdue_pct"] == 100.0
        assert stats["demand_pct"] == 100.0

    def test_negative_overdue_clipped_partial_demand(self):
        # Same negative-overdue case, but only 50k of the 60k demand gets paid.
        # overdue_pct is still 100% (nothing outstanding); demand_pct = 50/60.
        df = _overdue_demand_row(arrear_opening=-20_000, month_due_inst=60_000, paid=50_000)
        stats = compute_overdue_demand_pct(df)
        assert stats["overdue_pct"] == 100.0
        assert stats["demand_pct"] == round(50_000 / 60_000 * 100, 2)

    def test_zero_overdue_and_zero_demand_is_100_not_na_not_0(self):
        # A loan with no carried-over overdue AND nothing due this month (e.g. a
        # closed/matured account) must never divide by zero -- 100%, not N/A.
        df = _overdue_demand_row(arrear_opening=0, month_due_inst=0, paid=0)
        stats = compute_overdue_demand_pct(df)
        assert stats["overdue_pct"] == 100.0
        assert stats["demand_pct"] == 100.0

    def test_penal_charges_excluded_from_month_demand(self):
        # MONTH DUE PC must NOT inflate Month Demand -- per the business rule,
        # penal charges are not core EMI demand for this metric.
        df = _overdue_demand_row(arrear_opening=0, month_due_inst=10_000, month_due_pc=999_999, paid=10_000)
        stats = compute_overdue_demand_pct(df)
        assert stats["demand_total"] == 10_000
        assert stats["demand_pct"] == 100.0

    def test_group_level_sums_not_averaged(self):
        # Two loans in the same group: one fully collected, one not -- the group
        # % must come from SUMMING overdue/collected across loans, then dividing
        # once, never averaging each loan's own %.
        df = pd.concat([
            _overdue_demand_row(arrear_opening=10_000, paid=10_000),   # overdue fully collected
            _overdue_demand_row(arrear_opening=10_000, paid=0),        # nothing collected
        ], ignore_index=True)
        stats = compute_overdue_demand_pct(df)
        assert stats["overdue_total"] == 20_000
        assert stats["overdue_collected"] == 10_000
        assert stats["overdue_pct"] == 50.0  # not (100+0)/2

    def test_one_loans_credit_does_not_offset_anothers_overdue(self):
        # A credit-balance loan (Arrear Opening -5k, clipped to 0) must contribute
        # ZERO to the group's total overdue -- not a negative amount that would
        # silently reduce another loan's real overdue in the same group.
        df = pd.concat([
            _overdue_demand_row(arrear_opening=-5_000, paid=0),   # in credit, no real overdue
            _overdue_demand_row(arrear_opening=10_000, paid=5_000),  # real overdue, half collected
        ], ignore_index=True)
        stats = compute_overdue_demand_pct(df)
        assert stats["overdue_total"] == 10_000  # NOT 10_000 - 5_000
        assert stats["overdue_collected"] == 5_000
        assert stats["overdue_pct"] == 50.0

    def test_empty_df_returns_100_defaults(self):
        stats = compute_overdue_demand_pct(pd.DataFrame())
        assert stats["overdue_pct"] == 100.0
        assert stats["demand_pct"] == 100.0

    def test_per_loan_pct_columns_exist_for_loan_table_display(self):
        # A loan_table ("case wise") AI Query can only SELECT a real per-row
        # column, never an aggregate ratio measure -- these two literal columns
        # are what makes "show overdue collection % case wise" possible at all.
        df = _overdue_demand_row(arrear_opening=10_000, month_due_inst=20_000, paid=20_000)
        assert df["Overdue Collection %"].iloc[0] == 100.0
        assert df["Month Demand Collection %"].iloc[0] == 50.0

    def test_per_loan_pct_columns_zero_denominator_is_100(self):
        df = _overdue_demand_row(arrear_opening=0, month_due_inst=0, paid=0)
        assert df["Overdue Collection %"].iloc[0] == 100.0
        assert df["Month Demand Collection %"].iloc[0] == 100.0

    def test_per_loan_pct_columns_match_group_level_for_single_loan(self):
        df = _overdue_demand_row(arrear_opening=40_000, month_due_inst=10_000, paid=10_000)
        stats = compute_overdue_demand_pct(df)
        assert df["Overdue Collection %"].iloc[0] == stats["overdue_pct"]
        assert df["Month Demand Collection %"].iloc[0] == stats["demand_pct"]


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

    def test_strike_pct_spelled_out_yes_no(self):
        # Some monthly LCC extracts spell the flag out as "Yes"/"No" instead of
        # abbreviating it (real production file: 100% of a real Strike column
        # was "YES"/"NO", zero exact "Y"/"N" values). is_yes() was patched to
        # accept both spellings, but compute_strike_pct's own "valid" filter
        # used to still gate strictly on {"Y", "N"} -- on such a file the
        # denominator came back empty and Strike % silently reported 0.0
        # everywhere (dashboard, scorecard) instead of the real rate.
        df = make_df([
            {"Loan No": "L001", "Strike": "Yes"},
            {"Loan No": "L002", "Strike": "Yes"},
            {"Loan No": "L003", "Strike": "No"},
            {"Loan No": "L004", "Strike": "No"},
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

    def test_empty_prev_gives_none_mom(self):
        # No previous-month file uploaded (or a prior-period value of exactly
        # 0) is "no prior data to compare against", not "0% change" -- a
        # metric that goes from 0 to something must not read as flat/no-move.
        # ui/components.py::_kpi_card_html renders delta=None as "no prev
        # data" instead of a misleading 0.00% arrow.
        df = make_df([{"Month Collection (Excluding Reserve Collection)": 5_000.0}])
        metrics = compute_metrics(df, make_df([]))
        assert metrics["Collection %"][1] is None

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


class TestNormalizeTruncatedNames:
    """Regression: SegmentName values truncated at different lengths by the
    source system for the SAME real segment (confirmed on real production
    data -- "Passenger Commerc" / "Passenger Commerci" / "Passenger
    Commercial") used to split one segment's NPA/SOH numbers across several
    rows in every segment-wise breakdown instead of one."""

    def test_truncated_variants_merge_to_longest(self):
        s = pd.Series(["Passenger Commerc", "Passenger Commerci", "Passenger Commercial"])
        result = normalize_truncated_names(s)
        assert (result == "Passenger Commercial").all()

    def test_distinct_segments_are_not_merged(self):
        s = pd.Series(["Heavy Goods Vehicle", "Light Goods Vehicle", "Private Car", "Machinery"])
        result = normalize_truncated_names(s)
        assert result.tolist() == s.tolist()

    def test_nan_left_untouched(self):
        s = pd.Series(["Passenger Commerc", None, "Passenger Commercial"])
        result = normalize_truncated_names(s)
        assert pd.isna(result.iloc[1])
        assert result.iloc[0] == "Passenger Commercial"

    def test_empty_series_returns_as_is(self):
        s = pd.Series([], dtype=object)
        assert normalize_truncated_names(s).empty

    def test_all_nan_series_returns_untouched(self):
        s = pd.Series([None, None])
        result = normalize_truncated_names(s)
        assert result.isna().all()

    def test_custom_prefix_length(self):
        s = pd.Series(["ABCDEFGHIJ-1", "ABCDEFGHIJ-2"])
        result = normalize_truncated_names(s, prefix_chars=10)
        assert (result == "ABCDEFGHIJ-1").all() or (result == "ABCDEFGHIJ-2").all()
        assert result.iloc[0] == result.iloc[1]  # merged either way


class TestLoadAndValidateSegmentNormalization:
    def test_segment_name_truncation_variants_merged_on_load(self):
        buf = _build_upload({
            "SegmentName": ["Passenger Commerc", "Passenger Commerci", "Passenger Commercial"],
        })
        result_df, errs = load_and_validate.__wrapped__(buf)
        assert errs == []
        assert (result_df["SegmentName"] == "Passenger Commercial").all()


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
