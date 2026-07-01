import pytest

from smart_alerts import (
    alert_non_starters,
    alert_easy_settlements,
    alert_colending_at_risk,
    alert_insurance_delinquency,
    alert_high_arrears_ratio,
    run_all_alerts,
)
from helpers import make_df


class TestAlertNonStarters:
    def test_flags_y(self):
        df = make_df([
            {"Loan No": "L001", "Non Starter": "Y"},
            {"Loan No": "L002", "Non Starter": "N"},
        ])
        result = alert_non_starters(df)
        assert result["count"] == 1

    def test_zero_when_no_non_starters(self):
        df = make_df([{"Loan No": "L001", "Non Starter": "N"}])
        assert alert_non_starters(df)["count"] == 0

    def test_case_insensitive_match(self):
        # "y" (lowercase) should still be caught
        df = make_df([{"Loan No": "L001", "Non Starter": "y"}])
        assert alert_non_starters(df)["count"] == 1

    def test_severity_is_critical(self):
        df = make_df([{"Non Starter": "Y"}])
        assert alert_non_starters(df)["severity"] == "critical"

    def test_result_shape(self):
        df = make_df([{"Non Starter": "Y", "SOH": 50_000.0, "Closing Arrears": 5_000.0}])
        result = alert_non_starters(df)
        assert "count" in result
        assert "pos" in result
        assert "closing_arrears" in result
        assert "df" in result


class TestAlertEasySettlements:
    def test_flags_small_positive_arrears(self):
        df = make_df([
            {"Loan No": "L001", "Closing Arrears": 500.0},   # flagged
            {"Loan No": "L002", "Closing Arrears": 0.0},     # zero  -  not flagged
            {"Loan No": "L003", "Closing Arrears": 1_500.0}, # too large  -  not flagged
        ])
        assert alert_easy_settlements(df)["count"] == 1

    def test_boundary_999_is_flagged(self):
        df = make_df([{"Closing Arrears": 999.0}])
        assert alert_easy_settlements(df)["count"] == 1

    def test_boundary_1000_not_flagged(self):
        df = make_df([{"Closing Arrears": 1_000.0}])
        assert alert_easy_settlements(df)["count"] == 0

    def test_zero_arrears_not_flagged(self):
        df = make_df([{"Closing Arrears": 0.0}])
        assert alert_easy_settlements(df)["count"] == 0

    def test_severity_is_medium(self):
        df = make_df([{"Closing Arrears": 500.0}])
        assert alert_easy_settlements(df)["severity"] == "medium"


class TestAlertColending:
    def test_flags_delinquent_colending_loan(self):
        df = make_df([
            {"Loan No": "L001", "CoLending_Loans": "Y", "Arrears / EMI": 2.5},  # flagged
            {"Loan No": "L002", "CoLending_Loans": "Y", "Arrears / EMI": 0.0},  # no arrears
            {"Loan No": "L003", "CoLending_Loans": "N", "Arrears / EMI": 2.5},  # not colending
        ])
        assert alert_colending_at_risk(df)["count"] == 1

    def test_zero_when_colending_but_current(self):
        df = make_df([{"CoLending_Loans": "Y", "Arrears / EMI": 0.0}])
        assert alert_colending_at_risk(df)["count"] == 0

    def test_zero_when_not_colending(self):
        df = make_df([{"CoLending_Loans": "N", "Arrears / EMI": 3.0}])
        assert alert_colending_at_risk(df)["count"] == 0

    def test_severity_is_critical(self):
        df = make_df([{"CoLending_Loans": "Y", "Arrears / EMI": 1.0}])
        assert alert_colending_at_risk(df)["severity"] == "critical"


class TestAlertInsuranceDelinquency:
    def test_flags_exp_arrears_with_no_inst_arrears(self):
        df = make_df([{
            "ARREARS AGAINST INST": 0.0,
            "ARREARS AGAINST EXP":  8_000.0,  # > 5000
            "Arrears / EMI":        0.5,       # > 0
        }])
        assert alert_insurance_delinquency(df)["count"] == 1

    def test_not_flagged_when_inst_arrears_present(self):
        # If there are instalment arrears, this isn't pure insurance delinquency
        df = make_df([{
            "ARREARS AGAINST INST": 5_000.0,
            "ARREARS AGAINST EXP":  8_000.0,
            "Arrears / EMI":        1.5,
        }])
        assert alert_insurance_delinquency(df)["count"] == 0

    def test_not_flagged_when_exp_arrears_below_threshold(self):
        df = make_df([{
            "ARREARS AGAINST INST": 0.0,
            "ARREARS AGAINST EXP":  3_000.0,  # <= 5000
            "Arrears / EMI":        0.5,
        }])
        assert alert_insurance_delinquency(df)["count"] == 0

    def test_not_flagged_when_no_overall_arrears(self):
        df = make_df([{
            "ARREARS AGAINST INST": 0.0,
            "ARREARS AGAINST EXP":  8_000.0,
            "Arrears / EMI":        0.0,  # no delinquency
        }])
        assert alert_insurance_delinquency(df)["count"] == 0

    def test_severity_is_high(self):
        df = make_df([{
            "ARREARS AGAINST INST": 0.0,
            "ARREARS AGAINST EXP":  8_000.0,
            "Arrears / EMI":        0.5,
        }])
        assert alert_insurance_delinquency(df)["severity"] == "high"


class TestAlertHighArrears:
    def test_flags_when_arrears_exceed_50pct_of_loan(self):
        df = make_df([{
            "Loan Amount":          2_00_000.0,
            "ARREARS AGAINST INST": 1_10_000.0,  # 55% of loan
            "ARREARS AGAINST EXP":  0.0,
            "ARREARS AGAINST BC":   0.0,
        }])
        assert alert_high_arrears_ratio(df)["count"] == 1

    def test_not_flagged_below_50pct(self):
        df = make_df([{
            "Loan Amount":          2_00_000.0,
            "ARREARS AGAINST INST": 80_000.0,  # 40%  -  not flagged
            "ARREARS AGAINST EXP":  0.0,
            "ARREARS AGAINST BC":   0.0,
        }])
        assert alert_high_arrears_ratio(df)["count"] == 0

    def test_combined_arrears_components_summed(self):
        # 20% inst + 20% exp + 20% bc = 60% total → should be flagged
        df = make_df([{
            "Loan Amount":          3_00_000.0,
            "ARREARS AGAINST INST": 60_000.0,
            "ARREARS AGAINST EXP":  60_000.0,
            "ARREARS AGAINST BC":   60_000.0,
        }])
        assert alert_high_arrears_ratio(df)["count"] == 1

    def test_not_flagged_when_loan_amount_zero(self):
        df = make_df([{
            "Loan Amount":          0.0,
            "ARREARS AGAINST INST": 1_00_000.0,
        }])
        assert alert_high_arrears_ratio(df)["count"] == 0

    def test_severity_is_critical(self):
        df = make_df([{
            "Loan Amount":          2_00_000.0,
            "ARREARS AGAINST INST": 1_50_000.0,
        }])
        assert alert_high_arrears_ratio(df)["severity"] == "critical"


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

    def test_each_alert_has_required_keys(self):
        df = make_df([{"Loan No": "L001"}])
        required = {"title", "subtitle", "severity", "count", "pos", "closing_arrears", "df", "icon", "action"}
        for alert in run_all_alerts(df):
            assert required.issubset(alert.keys()), f"Alert '{alert.get('title')}' missing keys"
