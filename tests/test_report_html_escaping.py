"""Regression: raw data (customer/executive/branch names, Gemini narrative text)
must be HTML-escaped before entering the report's f-string HTML fragments.

Bug: an unescaped `&`, `<`, or `>` in a manually-entered LCC field (a real
possibility - e.g. "RAJESH & SONS TRANSPORT", or a vehicle description with a
stray "<") breaks the surrounding table markup from that point on. The report
is also offered as a raw .html download/email attachment, not just rendered
inside an email client's sandboxed viewer, so this isn't purely cosmetic.
"""
from report_agent.nodes.report_builder import _esc, _render_top_accounts, _render_narrative


class TestEscHelper:
    def test_escapes_html_special_characters(self):
        assert _esc("RAJESH & SONS") == "RAJESH &amp; SONS"
        assert _esc("<script>alert(1)</script>") == "&lt;script&gt;alert(1)&lt;/script&gt;"
        assert _esc("A < B > C") == "A &lt; B &gt; C"

    def test_none_becomes_empty_string(self):
        assert _esc(None) == ""

    def test_plain_text_unchanged(self):
        assert _esc("Yash Bhagoji Deve") == "Yash Bhagoji Deve"


class TestReportSectionsEscapeRawData:
    def test_top_accounts_customer_name_is_escaped(self):
        data = {
            "rows": [{
                "loan_no": "L1", "customer": "RAJESH & SONS <TRANSPORT>",
                "region": "WEST", "branch": "MAHAD", "bucket": "NPA", "soh": 100000.0,
            }],
            "summary": {"total_soh_cr": 0.01, "pct_of_portfolio": 5.0, "npa_count": 1},
            "n": 1,
        }
        html_out = _render_top_accounts(data)
        assert "RAJESH & SONS <TRANSPORT>" not in html_out
        assert "RAJESH &amp; SONS &lt;TRANSPORT&gt;" in html_out
        # Table structure survives - no unclosed/broken tag from the raw "<".
        assert html_out.count("<tr>") == html_out.count("</tr>")

    def test_narrative_text_is_escaped(self):
        narrative = "- Branch A & B showed <strong> improvement this month"
        html_out = _render_narrative(narrative)
        assert "<strong>" not in html_out or "&lt;strong&gt;" in html_out
        assert "A &amp; B" in html_out
