import pandas as pd

from investigator.branch_recipients import BRANCH_RECIPIENTS, get_branch_recipients
from investigator.email_draft import build_eml_bytes, build_priority_email_html


class TestGetBranchRecipients:
    def test_unmapped_branch_returns_empty_lists_not_an_error(self):
        assert get_branch_recipients("NOT_A_REAL_BRANCH") == ([], [])

    def test_mapped_branch_returns_its_to_and_cc(self):
        BRANCH_RECIPIENTS["MAHAD"] = {"to": ["manager@example.com"], "cc": ["zonal@example.com"]}
        try:
            assert get_branch_recipients("mahad") == (["manager@example.com"], ["zonal@example.com"])
            assert get_branch_recipients("  Mahad  ") == (["manager@example.com"], ["zonal@example.com"])
        finally:
            del BRANCH_RECIPIENTS["MAHAD"]

    def test_blank_branch_returns_empty_lists(self):
        assert get_branch_recipients("") == ([], [])
        assert get_branch_recipients(None) == ([], [])


class TestBuildPriorityEmailHtml:
    def test_renders_a_row_per_loan(self):
        df = pd.DataFrame([
            {"Cust Name": "Cust A", "Loan No": "L1", "Arrears / EMI": 10.0},
            {"Cust Name": "Cust B", "Loan No": "L2", "Arrears / EMI": 7.0},
        ])
        html = build_priority_email_html(df, "MAHAD")
        assert "Cust A" in html
        assert "Cust B" in html
        assert "MAHAD" in html

    def test_escapes_manually_typed_fields_against_html_injection(self):
        # Cust Name/MNT NAME are free-typed LCC fields -- unescaped, a stray
        # "&"/"<"/">" breaks the surrounding table markup (same convention
        # report_agent/nodes/report_builder.py's _esc() already follows).
        df = pd.DataFrame([{"Cust Name": "A & B <Corp>", "Loan No": "L1"}])
        html = build_priority_email_html(df, "MAHAD")
        assert "A &amp; B &lt;Corp&gt;" in html
        assert "<Corp>" not in html

    def test_empty_df_renders_a_no_cases_message_not_an_empty_table(self):
        html = build_priority_email_html(pd.DataFrame(), "MAHAD")
        assert "no priority cases" in html.lower()

    def test_branch_name_itself_is_escaped(self):
        html = build_priority_email_html(pd.DataFrame([{"Cust Name": "A"}]), "M&A BRANCH")
        assert "M&amp;A BRANCH" in html


class TestBuildEmlBytes:
    def test_produces_a_valid_unsent_mime_message(self):
        raw = build_eml_bytes(
            subject="Priority cases for Mahad",
            html_body="<html><body>Test</body></html>",
            to_list=["manager@example.com"],
            cc_list=["zonal@example.com"],
        )
        text = raw.decode("utf-8", errors="replace")
        assert "Subject: Priority cases for Mahad" in text
        assert "manager@example.com" in text
        assert "zonal@example.com" in text
        assert "X-Unsent: 1" in text

    def test_blank_recipients_produce_empty_to_cc_not_a_crash(self):
        raw = build_eml_bytes(subject="X", html_body="<html></html>", to_list=[], cc_list=[])
        assert isinstance(raw, bytes)
        assert len(raw) > 0
