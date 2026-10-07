"""Large customers: a customer's exposure is the SOH of ALL their loans in the
upload (matched by mobile), the delinquent ones are flagged, and the loan
list puts each customer's delinquent loans first."""
import pandas as pd

from analysis.customer_exposure import large_customers
from utils import CUSTOMER_SOH, add_customer_loan_count

CR = 1e7


def _book():
    rows = [
        # loan, mobile, branch, arrears/emi, bucket, SOH
        ("A1", "111", "X", 0.0, "STD", 1.5 * CR),
        ("A2", "111", "Y", 4.0, "NPA", 1.0 * CR),     # 111: 2.5 Cr across two branches, one NPA loan
        ("B1", "222", "X", 0.0, "STD", 2.2 * CR),     # 222: 2.2 Cr, all paying
        ("C1", "333", "X", 2.0, "SMA-2", 0.5 * CR),   # 333: small
        ("N1", "", "X", 3.0, "NPA", 3.0 * CR),        # no mobile: never a "customer"
    ]
    df = pd.DataFrame(rows, columns=["Loan No", "Cust Mob No", "Unit", "Arrears / EMI", "curr_bucket", "SOH"])
    df["Cust Name"] = df["Cust Mob No"].map({"111": "Ramesh", "222": "Suresh", "333": "Mahesh"}).fillna("")
    return add_customer_loan_count(df)


def test_exposure_counts_every_loan_of_the_customer():
    df = _book()
    assert df.loc[df["Cust Mob No"] == "111", CUSTOMER_SOH].eq(2.5 * CR).all()
    assert df.loc[df["Cust Mob No"] == "", CUSTOMER_SOH].isna().all()


def test_customers_above_the_threshold_with_delinquency():
    cust = large_customers(_book(), 2.0)["customers"].set_index("Mobile")
    assert list(cust.index) == ["111", "222"]                       # largest exposure first; 333 and no-mobile out
    assert cust.loc["111", "Exposure (Cr)"] == 2.5 and cust.loc["111", "Delinquent"] == 1
    assert cust.loc["111", "Delinquent SOH (Cr)"] == 1.0 and cust.loc["111", "Delinquent SOH %"] == 40.0
    assert cust.loc["111", "Worst Bucket"] == "NPA" and cust.loc["111", "Branches"] == "X, Y"
    assert bool(cust.loc["111", "Has Delinquent Loan"]) and not bool(cust.loc["222", "Has Delinquent Loan"])


def test_a_branch_filter_does_not_shrink_the_exposure():
    df = _book()
    cust = large_customers(df[df["Unit"] == "X"], 2.0)["customers"].set_index("Mobile")
    assert cust.loc["111", "Exposure (Cr)"] == 2.5 and cust.loc["111", "Loans"] == 1


def test_loans_listed_delinquent_first_within_customer():
    loans = large_customers(_book(), 2.0)["loans"]
    assert list(loans["Loan No"]) == ["A2", "A1", "B1"]
    assert list(loans["Delinquent"]) == ["Yes", "No", "No"]


def test_nothing_above_threshold_reports_the_largest():
    r = large_customers(_book(), 5.0)
    assert r["customers"].empty and r["largest_cr"] == 2.5


def test_highlighted_rows_are_tinted():
    from ui.components import html_table
    df = pd.DataFrame({"Customer": ["A", "B"], "Bad": [True, False]})
    html = html_table(df, [{"key": "Customer"}], highlight=lambda r: r["Bad"])
    assert html.count("#fff1f2") == 1
