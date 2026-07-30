import pytest

from smart_alerts import (
    alert_non_starters,
    alert_easy_settlements,
    alert_colending_at_risk,
    alert_insurance_delinquency,
    alert_high_arrears_ratio,
    alert_recent_advances_at_risk,
    run_all_alerts,
)
from helpers import make_df


class TestAlertNonStarters:
    @pytest.mark.parametrize("value,expected_count", [
        ("Y", 1), ("y", 1), ("N", 0),
    ], ids=["upper_y", "lower_y_case_insensitive", "n_not_flagged"])
    def test_flags_only_yes_case_insensitive(self, value, expected_count):
        df = make_df([{"Loan No": "L001", "Non Starter": value}])
        assert alert_non_starters(df)["count"] == expected_count

    def test_result_shape(self):
        df = make_df([{"Non Starter": "Y", "SOH": 50_000.0, "Closing Arrears": 5_000.0}])
        result = alert_non_starters(df)
        assert {"count", "pos", "closing_arrears", "df"}.issubset(result.keys())


class TestAlertEasySettlements:
    @pytest.mark.parametrize("closing_arrears,expected_count", [
        (0.0,      0),  # zero -> not flagged
        (500.0,    1),  # within window
        (999.0,    1),  # inclusive upper boundary
        (1_000.0,  0),  # exclusive upper boundary
        (1_500.0,  0),  # too large
    ], ids=["zero", "mid", "boundary_999", "boundary_1000", "too_large"])
    def test_flags_small_positive_arrears_below_threshold(self, closing_arrears, expected_count):
        df = make_df([{"Closing Arrears": closing_arrears}])
        assert alert_easy_settlements(df)["count"] == expected_count


class TestAlertColending:
    @pytest.mark.parametrize("colending,arrears,expected_count", [
        ("Y", 2.5, 1),  # delinquent co-lending loan -> flagged
        ("Y", 0.0, 0),  # co-lending but current -> not flagged
        ("N", 3.0, 0),  # delinquent but not co-lending -> not flagged
    ], ids=["delinquent_colending", "current_colending", "not_colending"])
    def test_flags_only_delinquent_colending_loans(self, colending, arrears, expected_count):
        df = make_df([{"CoLending_Loans": colending, "Arrears / EMI": arrears}])
        assert alert_colending_at_risk(df)["count"] == expected_count


class TestAlertInsuranceDelinquency:
    @pytest.mark.parametrize("inst,exp,arrears_emi,expected_count", [
        (0.0,     8_000.0, 0.5, 1),  # pure insurance delinquency -> flagged
        (5_000.0, 8_000.0, 1.5, 0),  # instalment arrears present -> not pure insurance
        (0.0,     3_000.0, 0.5, 0),  # exp arrears below threshold
        (0.0,     8_000.0, 0.0, 0),  # no overall delinquency at all
    ], ids=["pure_insurance", "inst_arrears_present", "below_threshold", "no_delinquency"])
    def test_flags_exp_arrears_with_no_inst_arrears(self, inst, exp, arrears_emi, expected_count):
        df = make_df([{
            "ARREARS AGAINST INST": inst, "ARREARS AGAINST EXP": exp, "Arrears / EMI": arrears_emi,
        }])
        assert alert_insurance_delinquency(df)["count"] == expected_count


class TestAlertHighArrears:
    @pytest.mark.parametrize("loan_amount,inst,exp,bc,expected_count", [
        (2_00_000.0, 1_10_000.0, 0.0,      0.0,      1),  # 55% of loan -> flagged
        (2_00_000.0, 80_000.0,   0.0,      0.0,      0),  # 40% -> not flagged
        (3_00_000.0, 60_000.0,   60_000.0, 60_000.0, 1),  # 20%+20%+20% summed -> 60% flagged
        (0.0,        1_00_000.0, 0.0,      0.0,      0),  # zero loan amount -> not flagged, no div/0
    ], ids=["over_50pct", "under_50pct", "components_summed", "zero_loan_amount"])
    def test_flags_when_combined_arrears_exceed_ratio(self, loan_amount, inst, exp, bc, expected_count):
        df = make_df([{
            "Loan Amount": loan_amount, "ARREARS AGAINST INST": inst,
            "ARREARS AGAINST EXP": exp, "ARREARS AGAINST BC": bc,
        }])
        assert alert_high_arrears_ratio(df)["count"] == expected_count


class TestRunAllAlerts:
    def test_returns_six_alerts(self):
        df = make_df([{"Loan No": "L001"}])
        assert len(run_all_alerts(df)) == 6

    def test_all_clear_on_clean_account(self):
        df = make_df([{
            "Non Starter":          "N",
            "Closing Arrears":      0.0,
            "CoLending_Loans":      "N",
            "Arrears / EMI":        0.0,
            "ARREARS AGAINST INST": 0.0,
            "ARREARS AGAINST EXP":  0.0,
            "ARREARS AGAINST BC":   0.0,
            "Loan Amount":          2_00_000.0,
        }])
        for alert in run_all_alerts(df):
            assert alert["count"] == 0, f"Alert '{alert['title']}' unexpectedly triggered"

    def test_each_alert_has_required_keys_and_correct_severity(self):
        # Consolidates what used to be five separate "test_severity_is_X" tests
        # (one per alert function) into one check against the real pipeline output.
        expected_severity = {
            "Non Starters": "critical",
            "Easy Settlements": "medium",
            "Co-lending Loans at Risk": "critical",
            "Insurance-Driven Delinquency": "high",
            "High Arrears: Loan at Risk": "critical",
        }
        df = make_df([{"Loan No": "L001"}])
        required = {"title", "subtitle", "severity", "count", "pos", "closing_arrears", "df", "df_full", "icon", "action"}
        seen_titles = set()
        for alert in run_all_alerts(df):
            assert required.issubset(alert.keys()), f"Alert '{alert.get('title')}' missing keys"
            title = alert["title"]
            seen_titles.add(title)
            if title in expected_severity:
                assert alert["severity"] == expected_severity[title], title
        assert expected_severity.keys() <= seen_titles

    def test_df_full_carries_every_column_df_only_carries_curated_subset(self):
        # Regression: the on-screen table (alert["df"]) is deliberately limited
        # to ALERT_DISPLAY_COLS for a fast scan, but the download must carry
        # every column (raw Excel + derived, e.g. SOH/curr_bucket) for the SAME
        # filtered rows -- df_full must never be column-limited like df is.
        df = make_df([{
            "Loan No": "L001", "Non Starter": "Y",
            # A column deliberately NOT in ALERT_DISPLAY_COLS.
            "StateName": "MAHARASHTRA",
        }])
        result = alert_non_starters(df)
        assert "StateName" not in result["df"].columns
        assert "StateName" in result["df_full"].columns
        assert len(result["df_full"]) == len(result["df"])  # same rows, more columns

    def test_high_arrears_df_full_includes_computed_ratio_column(self):
        # Arrears Ratio % is computed (not a raw Excel column) but is the exact
        # figure that made the row qualify -- deliberately included in the full
        # download, not just the curated on-screen table.
        df = make_df([{
            "Loan No": "L001", "Loan Amount": 100_000.0,
            "ARREARS AGAINST INST": 60_000.0, "ARREARS AGAINST EXP": 0.0, "ARREARS AGAINST BC": 0.0,
        }])
        result = alert_high_arrears_ratio(df)
        assert "Arrears Ratio %" in result["df"].columns
        assert "Arrears Ratio %" in result["df_full"].columns


class TestAlertRecentAdvancesAsOf:
    """Regression: this alert used to anchor "last N months" to wall-clock
    date.today() instead of the report's own reporting month -- the one
    date-anchored function in this codebase that didn't follow its own
    established as_of convention (every sibling new-advances function in
    analysis/portfolio_intelligence.py already does). Re-analyzing an old
    file used to silently compute the window from TODAY, not the file's own
    month, with no error and no indication anything was off."""

    def test_as_of_anchors_the_window_not_wall_clock_today(self):
        import pandas as pd
        # A loan sanctioned "6 months before" an OLD reporting month (Jan
        # 2024) is long outside any real "last 12 months" window measured
        # from wall-clock today (2026+), but well within it when correctly
        # anchored to as_of=2024-01.
        df = make_df([{
            "Loan No": "L1", "Ag_Date": pd.Timestamp("2023-07-01"), "Arrears / EMI": 1.0,
        }])
        result_wrong_anchor = alert_recent_advances_at_risk(df, months=12, as_of=None)
        result_correct_anchor = alert_recent_advances_at_risk(df, months=12, as_of="2024-01-01")
        assert result_wrong_anchor["count"] == 0     # wall-clock today: window doesn't reach 2023
        assert result_correct_anchor["count"] == 1   # anchored to the file's own month: it does

    def test_as_of_none_falls_back_to_todays_date_unchanged(self):
        # Backward compatibility: existing callers that don't pass as_of must
        # keep exactly their prior behavior (wall-clock today), not suddenly
        # exclude everything.
        import pandas as pd
        df = make_df([{
            "Loan No": "L1", "Ag_Date": pd.Timestamp.today(), "Arrears / EMI": 1.0,
        }])
        result = alert_recent_advances_at_risk(df, months=12, as_of=None)
        assert result["count"] == 1

    def test_run_all_alerts_threads_as_of_through(self):
        import pandas as pd
        df = make_df([{
            "Loan No": "L1", "Ag_Date": pd.Timestamp("2023-07-01"), "Arrears / EMI": 1.0,
        }])
        alerts = run_all_alerts(df, as_of="2024-01-01")
        recent = next(a for a in alerts if a["title"] == "Recent Advances at Risk")
        assert recent["count"] == 1
