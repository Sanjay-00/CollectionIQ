"""Report fixes from the audit (2026-10-16): what the report SAYS must match
what the numbers are, and a file meant for one region must be named and sent
as that region's."""
import pytest

from helpers import make_df


class TestDefinitionsSayWhatTheNumbersAre:
    def test_strike_means_current_on_the_instalment(self):
        # It used to say "a collection made this month", which is not the rule.
        from report_agent.story import _MEANING
        assert "current on their instalment" in _MEANING["Strike"]
        assert "collection made this month" not in _MEANING["Strike"]

    def test_thresholds_come_from_config(self):
        from config import HARD_BUCKET_ARREARS_EMI_MIN, INSURANCE_EXP_ARREARS_MIN
        from report_agent.story import _MEANING
        assert f"₹{INSURANCE_EXP_ARREARS_MIN:,}" in _MEANING["Insurance charge only"]
        assert f"₹{INSURANCE_EXP_ARREARS_MIN:,}" in _MEANING["Other charges"]
        assert f"{HARD_BUCKET_ARREARS_EMI_MIN} or more EMIs" in _MEANING["Hard bucket"]

    def test_month_table_says_last_month_is_last_months_whole_file(self):
        from report_agent.story import _month_vs_month
        c = make_df([{"Loan No": "L1"}, {"Loan No": "L2"}])
        blocks = _month_vs_month(c, make_df([{"Loan No": "L1"}]), "2026-08", "2026-07", {})
        note = blocks[0]["note"]
        assert "Same filters both months" in note and "Same loans" not in note


class TestReportFileNames:
    def test_two_units_with_the_same_safe_name_get_two_files(self):
        # "A/B" and "A B" both become "A_B"; in a ZIP the second overwrote the first.
        from report_agent.packs import build_files
        c = make_df([{"Loan No": "L1", "RegionName": "A/B"}, {"Loan No": "L2", "RegionName": "A B"}])
        out = build_files(c, c.iloc[0:0], "2026-08", None, "Regional", sections=["glance"], fmt="HTML")
        names = [n for n, _ in out["files"]]
        assert len(names) == 2 and len(set(names)) == 2
        assert sorted(out["titles"]) == ["Regional Report: A B", "Regional Report: A/B"]


class TestEmailAddresses:
    @pytest.fixture(autouse=True)
    def _smtp(self, monkeypatch):
        monkeypatch.setenv("SMTP_HOST", "smtp.invalid")     # checks run before any connection

    @pytest.mark.parametrize("to", ["evil@x.com\nBcc: spy@x.com", "ok@x.com, not-an-address", "a@b"])
    def test_bad_addresses_are_refused_before_sending(self, to):
        from report_agent.nodes.email_dispatcher import send_report_email
        ok, err = send_report_email("<p>report</p>", to, "2026-08")
        assert not ok and "Not a valid email address" in err


class TestLabelsFollowTheirThreshold:
    def test_hard_bucket_labels_and_definitions_use_the_same_number(self):
        from config import HARD_BUCKET_ARREARS_EMI_MIN as H
        from report_agent.story import _MEANING, _WHY_LABEL, SILENT_LABEL
        for label in (_WHY_LABEL["Hard + Not Paying"], _WHY_LABEL["Hard + Still Paying"],
                      _WHY_LABEL["Early Delinquency"], SILENT_LABEL):
            assert str(H) in label and label in _MEANING      # each label is defined, with the same number
