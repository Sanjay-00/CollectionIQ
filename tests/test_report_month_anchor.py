"""Every surface must anchor the "recent advances" window to the file's
reporting month, not to today.

Before this was fixed, the Alerts tab used the reporting month while AI Query
(compiler + priority mode), the Investigator and Root Cause used today, so the
same file gave 10 matching loans on one tab and 1 on another.

The reporting month here is deliberately far in the past (March 2020): a
today-anchored window can never contain these 2019 loans, so any surface that
slips back to "today" fails these tests no matter when they run.
"""
import pandas as pd
import pytest

from utils import CUTOFF_PLACEHOLDER, recent_advances_cutoff, resolve_dynamic_values

AS_OF = "2020-03-01"          # what app.py passes as the reporting month
EXPECTED = {"L_in", "L_boundary"}


def _df():
    # recent_advance_high_bucket = Ag_Date within the window AND Arrears/EMI >= 1.
    return pd.DataFrame({
        "Loan No":       ["L_in", "L_boundary", "L_old", "L_current"],
        "Cust Name":     ["A", "B", "C", "D"],
        "Cust Mob No":   ["1", "2", "3", "4"],
        "RegionName":    ["R1"] * 4,
        "Unit":          ["U1"] * 4,
        "MNT NAME":      ["Exec"] * 4,
        "Ag_Date": pd.to_datetime(["2019-09-15", "2019-03-01", "2018-06-01", "2019-10-01"]),
        "Arrears / EMI": [2.0, 1.5, 2.0, 0.0],
    })


def _loans(df):
    return set(df["Loan No"])


class TestCutoffHelper:
    def test_window_is_counted_back_from_reporting_month(self):
        assert recent_advances_cutoff(AS_OF) == pd.Timestamp("2019-03-01")

    def test_month_string_and_time_of_day_give_the_same_date_level_cutoff(self):
        assert recent_advances_cutoff("2020-03") == pd.Timestamp("2019-03-01")
        assert recent_advances_cutoff(pd.Timestamp("2020-03-01 17:45")) == pd.Timestamp("2019-03-01")

    def test_resolve_replaces_placeholder_in_nested_structures(self):
        plan = [{"op": "filter", "conditions": [{"column": "Ag_Date", "op": ">=", "value": CUTOFF_PLACEHOLDER}]}]
        resolved = resolve_dynamic_values(plan, AS_OF)
        assert resolved[0]["conditions"][0]["value"] == pd.Timestamp("2019-03-01")
        assert plan[0]["conditions"][0]["value"] == CUTOFF_PLACEHOLDER  # input not mutated


class TestEverySurfaceAgrees:
    def test_ai_query_compiler(self):
        from compiler.core import compile_logical
        from agents.plan_executor import execute_plan
        df = _df()
        ir = {"intent": "loan_table", "filters": [{"concept": "recent_advance_high_bucket"}]}
        plan, errs = compile_logical(ir, list(df.columns), as_of=AS_OF)
        assert errs == []
        assert CUTOFF_PLACEHOLDER not in repr(plan)
        out, err = execute_plan(df, plan)
        assert err == ""
        assert _loans(out) == EXPECTED

    def test_ai_query_graph_compile_node_reads_snapshot_dates(self):
        from graph import compile_and_validate_node
        from agents.plan_executor import execute_plan
        df = _df()
        state = {
            "ir1": {"intent": "loan_table", "filters": [{"concept": "recent_advance_high_bucket"}]},
            "result_df_full": df,
            "snapshot_dates": {"curr": AS_OF},
        }
        out = compile_and_validate_node(state)
        assert not out.get("error")
        result, err = execute_plan(df, out["plan"])
        assert err == ""
        assert _loans(result) == EXPECTED

    def test_ai_query_priority_mode(self):
        from graph import execute_node
        state = {
            "result_df_full": _df(),
            "ir1": {"intent": "priority_action", "display_columns": []},
            "plan": [],
            "snapshot_dates": {"curr": AS_OF},
        }
        out = execute_node(state)
        assert out["error"] == ""
        assert _loans(out["result_df"]) == EXPECTED

    def test_investigator_concept_filter_and_priority_accounts(self):
        from investigator.steps import concept_filter, priority_accounts, priority_menu
        df = _df()
        assert _loans(concept_filter(df, "recent_advance_high_bucket", as_of=AS_OF)) == EXPECTED
        assert _loans(priority_accounts(df, as_of=AS_OF)) == EXPECTED
        menu = priority_menu(df, as_of=AS_OF)
        row = menu[menu["Category"] == "Recent Advances: High Bucket"]
        assert int(row["Count"].iloc[0]) == len(EXPECTED)

    def test_alerts_tab_uses_the_same_cutoff(self):
        # The alert's own rule is Arrears/EMI > 0 (not >= 1), so it's compared
        # on the window only: both in-window delinquent loans, never the old one.
        from smart_alerts import alert_recent_advances_at_risk
        alert = alert_recent_advances_at_risk(_df(), as_of=AS_OF)
        assert set(alert["df_full"]["Loan No"]) == EXPECTED

    def test_root_cause_recent_vintage_share(self):
        from analysis.root_cause import compute_region_why_table
        df = _df()
        scorecard = pd.DataFrame([{"Region": "R1", "NPA%": 0.0, "Δ NPA%": None, "Collection%": 0.0, "Status": "-"}])
        why = compute_region_why_table(df, scorecard, as_of=AS_OF)
        row = why.iloc[0]
        # 3 delinquent loans (L_in, L_boundary, L_old); the 2 in-window ones are recent.
        assert row["Dominant Driver"] == "Recent-advance (sourcing/underwriting) quality"
        assert row["Driver Share %"] == 66.7   # 2/3, not 67.0 (no rounding before scaling)


class TestUnresolvedPlaceholderFailsLoudly:
    def test_apply_condition_rejects_raw_placeholder(self):
        # On a non-date column the placeholder would otherwise fall through to
        # the string branch and silently return every row.
        from agents.data_executor import _apply_condition
        df = _df().assign(Ag_Date=lambda d: d["Ag_Date"].astype(str))
        with pytest.raises(ValueError, match="unresolved"):
            _apply_condition(df, {"column": "Ag_Date", "op": ">=", "value": CUTOFF_PLACEHOLDER})
