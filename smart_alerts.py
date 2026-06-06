"""
Smart Alerts - pre-computed, rule-based flags that run on every LCC upload.
No LLM. Pure pandas. Always accurate.
"""
import pandas as pd
from datetime import date
from dateutil.relativedelta import relativedelta

from config import (
    EASY_SETTLEMENT_MAX_ARREARS,
    HIGH_ARREARS_LOAN_RATIO,
    INSURANCE_EXP_ARREARS_MIN,
    RECENT_ADVANCES_MONTHS,
)


def _fmt_dates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.date
    return df

# Fixed columns shown in every alert drilldown table
ALERT_DISPLAY_COLS = [
    "Loan No", "Zone", "RegionName", "Unit", "Ag_Date", "MNT NAME", "curr_bucket",
    "Due Dt", "Tenure", "Loan Status", "Loan Amount", "Veh ID", "Cust Name",
    "Guar Name", "Cust Mob No", "Guar Mob No", "Vehicle Description",
    "Month Due-Inst", "Month Due-Exp", "MONTH DUE (BC)", "MONTH DUE PC",
    "Month Receipt Amount", "Closing Arrears", "Arrears against Inst+Exp",
    "ARREARS AGAINST INST", "ARREARS AGAINST EXP",
    "LCC%", "Arrears / EMI", "DelinquencyDays", "VehEMI Accrued", "ClosingPC",
    "POS", "Non Starter", "Strike", "Last Receipt Date", "Last Receipt Amount",
    "ParentLDueDate", "No Coll 3 Months and >6 EMI", "NACHStatus",
    "TyreFlag", "FUEL_TYPE",
]


def _safe_cols(df: pd.DataFrame, cols: list) -> list:
    seen = set()
    result = []
    for c in cols:
        if c in df.columns and c not in seen:
            seen.add(c)
            result.append(c)
    return result


def _to_num(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df[col], errors="coerce").fillna(0) if col in df.columns else pd.Series(0, index=df.index)


def alert_non_starters(df: pd.DataFrame) -> dict:
    """Customers who have never paid even their 1st EMI."""
    mask = df["Non Starter"].astype(str).str.strip().str.upper() == "Y" if "Non Starter" in df.columns else pd.Series(False, index=df.index)
    subset = df[mask]
    return {
        "title": "Non Starters",
        "subtitle": "Never paid 1st EMI",
        "severity": "critical",
        "count": subset["Loan No"].nunique() if "Loan No" in subset.columns else len(subset),
        "pos": _to_num(subset, "SOH").sum(),
        "closing_arrears": _to_num(subset, "Closing Arrears").sum(),
        "df": _fmt_dates(subset[_safe_cols(df, ALERT_DISPLAY_COLS)]),
        "icon": "🚨",
        "action": "Immediate field visit required. Check if disbursement reached customer.",
    }


def alert_insurance_delinquency(df: pd.DataFrame) -> dict:
    """Customer has no arrears against installment but has arrears against expenses (insurance).
    EMI is being paid but unpaid insurance charge is creating artificial delinquency."""
    arr_inst = _to_num(df, "ARREARS AGAINST INST")
    arr_exp  = _to_num(df, "ARREARS AGAINST EXP")
    arrears  = _to_num(df, "Arrears / EMI")
    mask = (arr_inst <= 0) & (arr_exp > INSURANCE_EXP_ARREARS_MIN) & (arrears > 0)
    subset = df[mask]
    return {
        "title": "Insurance-Driven Delinquency",
        "subtitle": "EMI paid but unpaid insurance charge causing delinquency",
        "severity": "high",
        "count": subset["Loan No"].nunique() if "Loan No" in subset.columns else len(subset),
        "pos": _to_num(subset, "SOH").sum(),
        "closing_arrears": _to_num(subset, "Closing Arrears").sum(),
        "df": _fmt_dates(subset[_safe_cols(df, ALERT_DISPLAY_COLS)]),
        "icon": "⚠️",
        "action": "Settle insurance by cash or convert to child loan (EMI). Do not mark as willful default.",
    }


def alert_easy_settlements(df: pd.DataFrame) -> dict:
    """Accounts where closing arrears < 1000 - small amount, easy to clear."""
    closing = _to_num(df, "Closing Arrears")
    mask = (closing > 0) & (closing < EASY_SETTLEMENT_MAX_ARREARS)
    subset = df[mask]
    return {
        "title": "Easy Settlements",
        "subtitle": "Closing arrears < ₹1,000 - quick wins",
        "severity": "medium",
        "count": subset["Loan No"].nunique() if "Loan No" in subset.columns else len(subset),
        "pos": _to_num(subset, "SOH").sum(),
        "closing_arrears": _to_num(subset, "Closing Arrears").sum(),
        "df": _fmt_dates(subset[_safe_cols(df, ALERT_DISPLAY_COLS)]),
        "icon": "💡",
        "action": "One call / one visit can clear these. Assign to executives for same-day closure.",
    }


def alert_recent_advances_at_risk(df: pd.DataFrame, months: int = RECENT_ADVANCES_MONTHS) -> dict:
    """Loans sanctioned in the last N months that already have delinquencies."""
    cutoff = pd.Timestamp(date.today() - relativedelta(months=months))
    ag = df["Ag_Date"] if "Ag_Date" in df.columns else pd.Series(pd.NaT, index=df.index)
    arrears = _to_num(df, "Arrears / EMI")
    mask = (ag >= cutoff) & (arrears > 0)
    subset = df[mask]
    return {
        "title": "Recent Advances at Risk",
        "subtitle": f"Sanctioned in last {months} months - already delinquent",
        "severity": "high",
        "count": subset["Loan No"].nunique() if "Loan No" in subset.columns else len(subset),
        "pos": _to_num(subset, "SOH").sum(),
        "closing_arrears": _to_num(subset, "Closing Arrears").sum(),
        "df": _fmt_dates(subset[_safe_cols(df, ALERT_DISPLAY_COLS)]),
        "icon": "📉",
        "action": "Review sourcing quality. Engage field executive and check NACH status immediately.",
    }


def alert_colending_at_risk(df: pd.DataFrame) -> dict:
    """High-priority co-lending loans with any delinquency."""
    colend = df["CoLending_Loans"].astype(str).str.strip().str.upper() if "CoLending_Loans" in df.columns else pd.Series("N", index=df.index)
    arrears = _to_num(df, "Arrears / EMI")
    mask = (colend == "Y") & (arrears > 0)
    subset = df[mask]
    return {
        "title": "Co-lending Loans at Risk",
        "subtitle": "Partner bank exposure - must not default",
        "severity": "critical",
        "count": subset["Loan No"].nunique() if "Loan No" in subset.columns else len(subset),
        "pos": _to_num(subset, "SOH").sum(),
        "closing_arrears": _to_num(subset, "Closing Arrears").sum(),
        "df": _fmt_dates(subset[_safe_cols(df, ALERT_DISPLAY_COLS)]),
        "icon": "🏦",
        "action": "Escalate immediately to Regional Manager. Partner bank SLA may be breached.",
    }


def alert_high_arrears_ratio(df: pd.DataFrame) -> dict:
    """Accounts where Inst+Exp+BC arrears exceed 50% of original loan amount.
    Signals deep distress regardless of DPD bucket — potential write-off risk."""
    arr_inst = _to_num(df, "ARREARS AGAINST INST")
    arr_exp  = _to_num(df, "ARREARS AGAINST EXP")
    arr_bc   = _to_num(df, "ARREARS AGAINST BC")
    loan_amt = pd.to_numeric(df["Loan Amount"], errors="coerce") if "Loan Amount" in df.columns else pd.Series(0.0, index=df.index)

    total_arr = arr_inst + arr_exp + arr_bc
    mask = (total_arr > HIGH_ARREARS_LOAN_RATIO * loan_amt) & (loan_amt > 0)
    subset = df[mask].copy()

    # Add ratio column — sorted worst-first so >100% cases surface immediately
    if len(subset) > 0:
        loan = pd.to_numeric(subset["Loan Amount"], errors="coerce").replace(0, float("nan"))
        t_arr = (_to_num(subset, "ARREARS AGAINST INST") +
                 _to_num(subset, "ARREARS AGAINST EXP") +
                 _to_num(subset, "ARREARS AGAINST BC"))
        subset["Arrears Ratio %"] = (t_arr / loan * 100).round(1)
        subset = subset.sort_values("Arrears Ratio %", ascending=False)

    display_cols = ["Arrears Ratio %"] + _safe_cols(df, ALERT_DISPLAY_COLS)
    return {
        "title": "High Arrears: Loan at Risk",
        "subtitle": "Inst+Exp+BC arrears exceed 50% of original loan - Highly critical cases: Potential Write-off",
        "severity": "critical",
        "count": subset["Loan No"].nunique() if "Loan No" in subset.columns else len(subset),
        "pos": _to_num(subset, "SOH").sum(),
        "closing_arrears": _to_num(subset, "Closing Arrears").sum(),
        "df": _fmt_dates(subset[[c for c in display_cols if c in subset.columns]]),
        "icon": "🔥",
        "action": "Prioritise >100% cases for legal/recovery, these may be unrecoverable without immediate escalation.",
    }


def run_all_alerts(df: pd.DataFrame, recent_months: int = 12) -> list:
    """Run all 6 alerts and return list sorted by severity."""
    alerts = [
        alert_non_starters(df),
        alert_colending_at_risk(df),
        alert_insurance_delinquency(df),
        alert_recent_advances_at_risk(df, recent_months),
        alert_easy_settlements(df),
        alert_high_arrears_ratio(df),
    ]
    return alerts
