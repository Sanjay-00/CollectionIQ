"""
Shared test helpers. Not pytest fixtures  -  plain functions imported directly.
"""
import pandas as pd


# Minimal column set that satisfies all tested functions.
_DEFAULTS: dict = {
    "Loan No":                                           "PLACEHOLDER",
    "RegionName":                                        "WEST",
    "Unit":                                              "MAHAD",
    "Loan Status":                                       "Active",
    "Ag_Date":                                           pd.Timestamp("2024-01-01"),
    "Arrears / EMI":                                     0.0,
    "Month Receipt Amount":                              10_000.0,
    "Month Collection (Excluding Reserve Collection)":   10_000.0,
    "Net Collection Demand Inst+Exp+BC":                 10_000.0,
    "POS":                                               1_00_000.0,
    "LCC%":                                              85.0,
    "Strike":                                            "Y",
    "Closing Arrears":                                   0.0,
    "Month Due-Inst":                                    9_500.0,
    "Month Due-Exp":                                     500.0,
    "Total Cum Collection":                              50_000.0,
    "Cum Coll (Inst+Exp)":                               50_000.0,
    "Non Starter":                                       "N",
    "CoLending_Loans":                                   "N",
    "ARREARS AGAINST INST":                              0.0,
    "ARREARS AGAINST EXP":                               0.0,
    "ARREARS AGAINST BC":                                0.0,
    "Loan Amount":                                       2_00_000.0,
    "SOH":                                               1_00_000.0,
    "curr_bucket":                                       "STD",
    "curr_score":                                        0,
}


def make_df(rows: list[dict]) -> pd.DataFrame:
    """Build a minimal valid LCC DataFrame from a list of row dicts.

    Each dict is merged over _DEFAULTS so only the fields under test need
    to be specified. Loan No is auto-assigned (L001, L002, …) unless provided.
    Returns an empty DataFrame with all columns when rows==[].
    """
    if not rows:
        return pd.DataFrame(columns=list(_DEFAULTS.keys()))

    result = []
    for i, row in enumerate(rows):
        r = _DEFAULTS.copy()
        r["Loan No"] = f"L{i + 1:03d}"
        r.update(row)
        result.append(r)
    return pd.DataFrame(result)
