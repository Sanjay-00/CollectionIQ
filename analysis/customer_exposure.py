"""Large customers: borrowers whose loans add up to a big SOH, and which of
them have a loan behind on payment.

A customer is matched by mobile number (Cust Mob No), the same rule as fleet
operators. "Exposure" is the customer's total SOH across the WHOLE upload
(utils.CUSTOMER_SOH, worked out at load time), so a sidebar filter on one
branch never hides a customer whose other loans sit in another branch. The
loans listed are the ones in view.
"""
import pandas as pd

from utils import BUCKET_SCORE, CUSTOMER_LOAN_COUNT, CUSTOMER_SOH, add_customer_loan_count, loan_flags, to_num

_COLLECTED = "Month Collection (Excluding Reserve Collection)"

# Loan-level columns, in reading order: who, where, how bad, how much, proof.
LOAN_COLS = [
    "Cust Name", "Cust Mob No", "Loan No", "RegionName", "Unit", "MNT NAME", "curr_bucket", "Delinquent",
    "SOH", "POS", "Arrears / EMI", "Closing Arrears", "ARREARS AGAINST INST", "ARREARS AGAINST EXP",
    "Month Due-Inst", _COLLECTED, "Last Receipt Date", "Last Receipt Amount",
    "Ag_Date", "Loan Amount", "Make", "Vehicle Description", "Veh ID",
]


def _join(values: pd.Series) -> str:
    seen = [str(v) for v in dict.fromkeys(values.dropna()) if str(v).strip() and str(v).lower() != "nan"]
    return ", ".join(seen)


def large_customers(df: pd.DataFrame, min_soh_cr: float) -> dict:
    """{"customers": one row per customer, largest exposure first;
        "loans": every loan of those customers in view, the customer's
                 delinquent loans first;
        "largest_cr": the biggest customer exposure in the upload (so an empty
                      result can say how far off the threshold is)}."""
    empty = {"customers": pd.DataFrame(), "loans": pd.DataFrame(), "largest_cr": 0.0}
    if df.empty or "Cust Mob No" not in df.columns or "Loan No" not in df.columns:
        return empty
    if CUSTOMER_SOH not in df.columns:          # hand-built frames: count within themselves
        df = add_customer_loan_count(df)
    if CUSTOMER_SOH not in df.columns:
        return empty
    exposure = pd.to_numeric(df[CUSTOMER_SOH], errors="coerce")
    largest = float(exposure.max() / 1e7) if exposure.notna().any() else 0.0
    d = df[exposure >= min_soh_cr * 1e7].copy()
    if d.empty:
        return {**empty, "largest_cr": round(largest, 2)}

    f = loan_flags(d)
    mob = d["Cust Mob No"].astype(str).str.strip()
    d["_soh"], d["_delq"], d["_npa"] = f["soh"], f["delinquent"], f["npa"]
    d["_delq_soh"] = f["soh"].where(f["delinquent"], 0.0)
    d["_arrears"] = to_num(d, "Closing Arrears", fill=0)
    d["_score"] = d["curr_bucket"].map(BUCKET_SCORE) if "curr_bucket" in d.columns else 0
    g = d.groupby(mob, sort=False)

    worst = g["_score"].max()
    score_to_bucket = {v: k for k, v in BUCKET_SCORE.items()}
    cust = pd.DataFrame({
        "Customer": g["Cust Name"].first() if "Cust Name" in d.columns else "",
        "Mobile": g["Cust Mob No"].first(),
        "Exposure (Cr)": (g[CUSTOMER_SOH].first() / 1e7).round(2),
        "Loans in Book": g[CUSTOMER_LOAN_COUNT].first() if CUSTOMER_LOAN_COUNT in d.columns else g["Loan No"].nunique(),
        "Loans": g["Loan No"].nunique(),
        "Delinquent": g["_delq"].sum().astype(int),
        "NPA": g["_npa"].sum().astype(int),
        "SOH (Cr)": (g["_soh"].sum() / 1e7).round(2),
        "Delinquent SOH (Cr)": (g["_delq_soh"].sum() / 1e7).round(2),
        "Overdue (Closing Arrears)": g["_arrears"].sum().round(0),
        "Worst Bucket": worst.map(score_to_bucket).fillna("-"),
        "Branches": g["Unit"].agg(_join) if "Unit" in d.columns else "",
        "Regions": g["RegionName"].agg(_join) if "RegionName" in d.columns else "",
        "Executives": g["MNT NAME"].agg(_join) if "MNT NAME" in d.columns else "",
    })
    cust["Delinquent %"] = (cust["Delinquent"] / cust["Loans"] * 100).round(2)
    cust["Delinquent SOH %"] = (cust["Delinquent SOH (Cr)"] / cust["SOH (Cr)"].where(cust["SOH (Cr)"] > 0) * 100) \
        .round(2).fillna(0.0)
    cust["Has Delinquent Loan"] = cust["Delinquent"] > 0
    cust = cust.sort_values(["Exposure (Cr)", "Delinquent SOH (Cr)"], ascending=False, kind="stable").reset_index(drop=True)

    # Loans: grouped by customer (largest exposure first), delinquent loans
    # on top within a customer, then by SOH.
    order = {m: i for i, m in enumerate(cust["Mobile"].astype(str).str.strip())}
    d["Delinquent"] = d["_delq"].map({True: "Yes", False: "No"})
    d["_order"] = mob.map(order)
    loans = d.sort_values(["_order", "_delq", "_soh"], ascending=[True, False, False], kind="stable")
    loans = loans[[c for c in LOAN_COLS if c in loans.columns]].reset_index(drop=True)
    return {"customers": cust, "loans": loans, "largest_cr": round(largest, 2)}
