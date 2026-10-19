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
import pandas as pd

from graph import _strip_pii_from_agg_rows, _strip_large_state_fields, _PII_COLS_IN_AGG_ROWS, execute_node


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


def _customer_agg_df():
    return pd.DataFrame({
        "Cust Mob No": [111, 222, 333],
        "Cust Name": ["Alice", None, "  "],
        "count": [4, 3, 3],
    })


def _customer_agg_state(df):
    return {
        "ir1": {"intent": "aggregation", "dimensions": ["customer"]},
        "plan": [{"op": "select", "columns": list(df.columns)}],
        "result_df_full": df,
        "aggregation_spec": {"group_by": "Cust Name (Cust Mob No)", "metrics": [{"label": "count"}]},
    }


class TestCustomerNameDisplayMerge:
    """execute_node's customer-grain display merge (see graph.py, right after
    the pre-existing MNT NAME/Unit merge) -- covers a blank Cust Name and a
    missing Cust Name column, both real-world cases since Cust Name is only
    in utils.py's REQUIRED_COLS (warn-only), never CRITICAL_COLS."""

    def test_blank_name_falls_back_to_bare_number_not_literal_nan(self):
        out = execute_node(_customer_agg_state(_customer_agg_df()))
        assert out["error"] == ""
        label_col = out["result_df"]["Cust Name (Cust Mob No)"]
        assert label_col.tolist() == ["Alice (111)", "222", "333"]
        assert "nan" not in " ".join(label_col.astype(str)).lower()

    def test_merged_column_stripped_from_agg_rows_pii_sample(self):
        out = execute_node(_customer_agg_state(_customer_agg_df()))
        agg_rows = out["result_kpis"]["_agg_rows"]
        assert all("Cust Name (Cust Mob No)" not in row for row in agg_rows)

    def test_missing_cust_name_column_falls_back_header_label_and_does_not_crash(self):
        df = _customer_agg_df().drop(columns=["Cust Name"])
        out = execute_node(_customer_agg_state(df))
        assert out["error"] == ""
        assert "Cust Mob No" in out["result_df"].columns
        assert "Cust Name (Cust Mob No)" not in out["result_df"].columns
        # Header label state corrected to match what's actually in result_df.
        assert out["aggregation_spec"]["group_by"] == "Cust Mob No"
