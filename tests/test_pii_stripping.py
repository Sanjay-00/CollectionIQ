"""Customer PII must never reach the Gemini Insight Generator prompt or a
LangSmith trace via result_kpis["_agg_rows"].

Real incident: 4 fast-path views (top_accounts, fleet_exposure, repossession,
good_customers) are customer/loan-grain, so their underlying tables carry
real Cust Name/Cust Mob No columns. graph.py's view_node fallback samples the
first 5 rows of that real table into result_kpis["_agg_rows"] so the Insight
Generator has something concrete to describe -- confirmed, on real data, that
this included actual customer names and phone numbers before this fix, sent
directly into the Gemini prompt (agents/insight_generator.py) and, since
result_kpis wasn't on _LARGE_STATE_FIELDS, into LangSmith traces too.
"""
from graph import _strip_pii_from_agg_rows, _strip_large_state_fields, _PII_COLS_IN_AGG_ROWS


class TestStripPiiFromAggRows:
    def test_removes_all_known_pii_columns(self):
        rows = [{
            "Loan No": "L1", "Cust Name": "Rajesh Kumar", "Cust Mob No": "9876543210",
            "Guar Name": "Suresh Kumar", "Guar Mob No": "9123456789",
            "SOH": 500000, "RegionName": "PUNE", "curr_bucket": "NPA",
        }]
        stripped = _strip_pii_from_agg_rows(rows)
        for col in _PII_COLS_IN_AGG_ROWS:
            assert col not in stripped[0]

    def test_preserves_analytical_columns(self):
        rows = [{
            "Loan No": "L1", "Cust Name": "Rajesh Kumar", "SOH": 500000,
            "RegionName": "PUNE", "curr_bucket": "NPA", "Arrears / EMI": 3.5,
        }]
        stripped = _strip_pii_from_agg_rows(rows)
        assert stripped[0]["Loan No"] == "L1"
        assert stripped[0]["SOH"] == 500000
        assert stripped[0]["RegionName"] == "PUNE"
        assert stripped[0]["curr_bucket"] == "NPA"
        assert stripped[0]["Arrears / EMI"] == 3.5

    def test_handles_rows_with_no_pii_columns(self):
        rows = [{"Branch": "AKOLA", "NPA%": 14.77}]
        assert _strip_pii_from_agg_rows(rows) == rows

    def test_empty_and_none_are_safe(self):
        assert _strip_pii_from_agg_rows([]) == []
        assert _strip_pii_from_agg_rows(None) is None


class TestLangSmithTraceStripping:
    """The second, independent layer -- protects even if some future code
    path populates _agg_rows with raw PII without going through
    _strip_pii_from_agg_rows first."""

    def test_agg_rows_pii_stripped_before_tracing(self):
        state = {
            "result_kpis": {
                "Count": 2,
                "_agg_rows": [
                    {"Loan No": "L1", "Cust Name": "Rajesh Kumar", "Cust Mob No": "9876543210", "SOH": 500000},
                ],
            },
        }
        out = _strip_large_state_fields(state)
        assert "Cust Name" not in out["result_kpis"]["_agg_rows"][0]
        assert "Cust Mob No" not in out["result_kpis"]["_agg_rows"][0]
        assert out["result_kpis"]["_agg_rows"][0]["SOH"] == 500000

    def test_other_result_kpis_fields_untouched(self):
        state = {"result_kpis": {"Count": 5, "Total POS": 1000000, "_agg_rows": [{"Cust Name": "X"}]}}
        out = _strip_large_state_fields(state)
        assert out["result_kpis"]["Count"] == 5
        assert out["result_kpis"]["Total POS"] == 1000000

    def test_no_agg_rows_is_a_noop(self):
        state = {"result_kpis": {"Count": 5, "Total POS": 1000000}}
        out = _strip_large_state_fields(state)
        assert out["result_kpis"] == {"Count": 5, "Total POS": 1000000}

    def test_missing_result_kpis_does_not_crash(self):
        assert _strip_large_state_fields({"other_field": 1}) == {"other_field": 1}
