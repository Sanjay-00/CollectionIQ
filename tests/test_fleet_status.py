"""A fleet operator (>= FLEET_MIN_LOANS loans) is a fact about the CUSTOMER,
counted once over the whole upload -- never re-counted inside whatever slice
is on screen. Before this, filtering the sidebar to one branch (or scoping an
Investigator step / AI Query dimension) silently demoted a customer whose
other loans sat in another branch, and the Root Cause "why" table merged every
loan with no mobile number into one phantom fleet operator.

Customer 111 below holds 4 loans split 2/2 across branches A and B: seen
through branch A alone that's only 2 loans, yet it's still a fleet operator.
"""
import pandas as pd
import pytest

from utils import CUSTOMER_LOAN_COUNT, add_customer_loan_count, fleet_loan_mask
from tests.test_utils import _build_upload


def _book():
    rows = [
        # loan, mobile, branch, region, arrears/emi
        ("L1", "111", "A", "R1", 2.0),
        ("L2", "111", "A", "R1", 0.0),
        ("L3", "111", "B", "R2", 1.0),
        ("L4", "111", "B", "R2", 0.0),
        ("L5", "222", "A", "R1", 3.0),   # 2 loans: not a fleet operator
        ("L6", "222", "A", "R1", 0.0),
        ("N1", "", "A", "R1", 2.0),      # blank mobiles: never one merged "customer"
        ("N2", "", "A", "R1", 2.0),
        ("N3", "", "A", "R1", 2.0),
    ]
    df = pd.DataFrame(rows, columns=["Loan No", "Cust Mob No", "Unit", "RegionName", "Arrears / EMI"])
    df["Cust Name"] = df["Cust Mob No"].map({"111": "Ramesh", "222": "Suresh"}).fillna("")
    df["SOH"] = 100.0
    df["curr_bucket"] = "STD"
    return add_customer_loan_count(df)


def _branch_a(df):
    return df[df["Unit"] == "A"]


class TestCustomerLoanCount:
    def test_counts_across_the_whole_upload(self):
        df = _book()
        assert set(df.loc[df["Cust Mob No"] == "111", CUSTOMER_LOAN_COUNT]) == {4}
        assert set(df.loc[df["Cust Mob No"] == "222", CUSTOMER_LOAN_COUNT]) == {2}

    def test_blank_mobile_gets_no_count(self):
        df = _book()
        assert df.loc[df["Cust Mob No"] == "", CUSTOMER_LOAN_COUNT].isna().all()

    def test_filtering_keeps_the_whole_upload_count(self):
        view = _branch_a(_book())
        assert fleet_loan_mask(view)[view["Cust Mob No"] == "111"].all()
        assert not fleet_loan_mask(view)[view["Cust Mob No"] != "111"].any()

    def test_load_and_validate_adds_the_column(self):
        from utils import load_and_validate
        upload = _build_upload({"Cust Mob No": ["9876543210", "9876543210", ""]})
        df, errs = load_and_validate.__wrapped__(upload)
        assert errs == []
        counts = df[CUSTOMER_LOAN_COUNT]
        assert counts.iloc[0] == 2 and counts.iloc[1] == 2 and pd.isna(counts.iloc[2])

    def test_multi_file_upload_counts_across_files(self, monkeypatch):
        import ui.components as components
        east = add_customer_loan_count(pd.DataFrame({"Loan No": ["E1", "E2"], "Cust Mob No": ["111", "111"]}))
        west = add_customer_loan_count(pd.DataFrame({"Loan No": ["W1"], "Cust Mob No": ["111"]}))
        files = {"east": east, "west": west}
        monkeypatch.setattr(components, "load_and_validate", lambda f: (files[f].copy(), []))
        combined, errs = components._load_and_concat(["east", "west"])
        assert errs == []
        assert set(combined[CUSTOMER_LOAN_COUNT]) == {3}


class TestEveryFleetViewAgrees:
    def test_portfolio_fleet_exposure_under_branch_filter(self):
        from analysis.portfolio_intelligence import compute_fleet_exposure
        out = compute_fleet_exposure(_branch_a(_book()))
        assert out["count"] == 1                                   # Ramesh stays a fleet operator
        assert out["top_df"]["Loans"].tolist() == [2]              # his loans in this view

    def test_root_cause_ignores_blank_mobiles(self):
        from analysis.root_cause import compute_region_why_table
        df = _book()
        scorecard = pd.DataFrame([{"Region": r, "NPA%": 0.0, "Δ NPA%": None, "Collection%": 0.0, "Status": "-"}
                                  for r in ("R1", "R2")])
        why = compute_region_why_table(df, scorecard).set_index("Region")
        # R1 delinquent: L1 (Ramesh, fleet), L5 (Suresh), N1-N3 (blank). Fleet share
        # is 1/5 -- it was 4/5 when the three blank-mobile loans merged into one "fleet".
        assert why.loc["R1", "Dominant Driver"] == "Fleet-operator concentration"
        assert why.loc["R1", "Driver Share %"] == 20.0

    def test_investigator_fleet_defaulters_scoped_to_branch(self):
        from investigator.steps import fleet_defaulters
        out = fleet_defaulters(_book(), scope_col="branch", scope_value="A")
        assert out["Loan No"].tolist() == ["L1"]

    def test_investigator_scoping_without_precomputed_count(self):
        # Hand-built frames without the column are counted BEFORE scoping.
        from investigator.steps import fleet_defaulters
        out = fleet_defaulters(_book().drop(columns=[CUSTOMER_LOAN_COUNT]), scope_col="branch", scope_value="A")
        assert out["Loan No"].tolist() == ["L1"]

    @pytest.mark.parametrize("view", ["full", "branch_a"])
    def test_ai_query_fleet_operators_per_branch(self, view):
        from compiler.core import compile_logical
        from agents.plan_executor import execute_plan
        df = _book() if view == "full" else _branch_a(_book())
        plan, errs = compile_logical({
            "dimensions": ["branch"],
            "entity_filters": [{"concept": "fleet_operator"}],
            "measures": [{"agg": "count", "distinct": "Cust Mob No", "alias": "fleet_operators"}],
        }, df.columns)
        assert errs == []
        out, err = execute_plan(df, plan)
        assert err == ""
        got = dict(zip(out["Unit"], out["fleet_operators"]))
        # Ramesh has only 2 loans in each branch but 4 overall: a fleet operator in both.
        assert got == ({"A": 1, "B": 1} if view == "full" else {"A": 1})
