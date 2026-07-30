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
from utils import to_num, account_count, is_yes


def _fmt_dates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.date
    return df

# Fixed columns shown in every alert drilldown table. One shared list across
# all 6 alerts (not per-alert) -- deliberately includes each alert's own
# trigger-formula column so a flagged row carries its own visible proof (e.g.
# CoLending_Loans/ARREARS AGAINST BC below were added because Co-lending Loans
# at Risk / High Arrears at Risk's own boolean conditions read them, and
# without them a flagged row showed no visible evidence of why it qualified).
ALERT_DISPLAY_COLS = [
    "Loan No", "Zone", "RegionName", "Unit", "Ag_Date", "MNT NAME", "curr_bucket",
    "Due Dt", "Tenure", "Loan Status", "Loan Amount", "Veh ID", "Cust Name",
    "Guar Name", "Cust Mob No", "Guar Mob No", "Vehicle Description",
    "Month Due-Inst", "Month Due-Exp", "MONTH DUE (BC)", "MONTH DUE PC",
    "Month Receipt Amount", "Closing Arrears", "Arrears against Inst+Exp",
    "ARREARS AGAINST INST", "ARREARS AGAINST EXP", "ARREARS AGAINST BC",
    "LCC%", "Arrears / EMI", "DelinquencyDays", "VehEMI Accrued", "ClosingPC",
    "POS", "Non Starter", "Strike", "Last Receipt Date", "Last Receipt Amount",
    "ParentLDueDate", "No Coll 3 Months and >6 EMI", "NACHStatus",
    "TyreFlag", "FUEL_TYPE", "CoLending_Loans",
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
    return to_num(df, col, fill=0)


def alert_non_starters(df: pd.DataFrame) -> dict:
    """Customers who have never paid even their 1st EMI."""
    mask = is_yes(df, "Non Starter")
    subset = df[mask]
    return {
        "title": "Non Starters",
        "subtitle": "Never paid 1st EMI",
        "severity": "critical",
        "count": account_count(subset),
        "pos": _to_num(subset, "SOH").sum(),
        "closing_arrears": _to_num(subset, "Closing Arrears").sum(),
        "df": _fmt_dates(subset[_safe_cols(df, ALERT_DISPLAY_COLS)]),
        "df_full": _fmt_dates(subset),
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
        "count": account_count(subset),
        "pos": _to_num(subset, "SOH").sum(),
        "closing_arrears": _to_num(subset, "Closing Arrears").sum(),
        "df": _fmt_dates(subset[_safe_cols(df, ALERT_DISPLAY_COLS)]),
        "df_full": _fmt_dates(subset),
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
        "subtitle": f"Closing arrears < ₹{EASY_SETTLEMENT_MAX_ARREARS:,} - quick wins",
        "severity": "medium",
        "count": account_count(subset),
        "pos": _to_num(subset, "SOH").sum(),
        "closing_arrears": _to_num(subset, "Closing Arrears").sum(),
        "df": _fmt_dates(subset[_safe_cols(df, ALERT_DISPLAY_COLS)]),
        "df_full": _fmt_dates(subset),
        "icon": "💡",
        "action": "One call / one visit can clear these. Assign to executives for same-day closure.",
    }


def alert_recent_advances_at_risk(df: pd.DataFrame, months: int = RECENT_ADVANCES_MONTHS, as_of=None) -> dict:
    """Loans sanctioned in the last N months that already have delinquencies.

    as_of anchors "last N months" to the report's own reporting month, not
    wall-clock today -- same convention as every other as_of-taking function
    in this codebase (e.g. analysis/portfolio_intelligence.py's
    compute_new_advances/compute_product_analysis), since re-analyzing an old
    file must use the month IT reports on, not the day this code happens to
    run. Falls back to real today only when the caller doesn't have a
    reporting month to pass (as_of=None), same fallback every sibling
    function already uses."""
    ref = pd.Timestamp(as_of) if as_of is not None else pd.Timestamp(date.today())
    cutoff = ref - relativedelta(months=months)
    ag = df["Ag_Date"] if "Ag_Date" in df.columns else pd.Series(pd.NaT, index=df.index)
    arrears = _to_num(df, "Arrears / EMI")
    mask = (ag >= cutoff) & (arrears > 0)
    subset = df[mask]
    return {
        "title": "Recent Advances at Risk",
        "subtitle": f"Sanctioned in last {months} months - already delinquent",
        "severity": "high",
        "count": account_count(subset),
        "pos": _to_num(subset, "SOH").sum(),
        "closing_arrears": _to_num(subset, "Closing Arrears").sum(),
        "df": _fmt_dates(subset[_safe_cols(df, ALERT_DISPLAY_COLS)]),
        "df_full": _fmt_dates(subset),
        "icon": "📉",
        "action": "Review sourcing quality. Engage field executive and check NACH status immediately.",
    }


def alert_colending_at_risk(df: pd.DataFrame) -> dict:
    """High-priority co-lending loans with any delinquency."""
    arrears = _to_num(df, "Arrears / EMI")
    mask = is_yes(df, "CoLending_Loans") & (arrears > 0)
    subset = df[mask]
    return {
        "title": "Co-lending Loans at Risk",
        "subtitle": "Partner bank exposure - must not default",
        "severity": "critical",
        "count": account_count(subset),
        "pos": _to_num(subset, "SOH").sum(),
        "closing_arrears": _to_num(subset, "Closing Arrears").sum(),
        "df": _fmt_dates(subset[_safe_cols(df, ALERT_DISPLAY_COLS)]),
        "df_full": _fmt_dates(subset),
        "icon": "🏦",
        "action": "Escalate immediately to Regional Manager. Partner bank SLA may be breached.",
    }


def alert_high_arrears_ratio(df: pd.DataFrame) -> dict:
    """Accounts where Inst+Exp+BC arrears exceed HIGH_ARREARS_LOAN_RATIO of original loan amount.
    Signals deep distress regardless of DPD bucket  -  potential write-off risk."""
    arr_inst = _to_num(df, "ARREARS AGAINST INST")
    arr_exp  = _to_num(df, "ARREARS AGAINST EXP")
    arr_bc   = _to_num(df, "ARREARS AGAINST BC")
    loan_amt = to_num(df, "Loan Amount", fill=0)

    total_arr = arr_inst + arr_exp + arr_bc
    mask = (total_arr > HIGH_ARREARS_LOAN_RATIO * loan_amt) & (loan_amt > 0)
    subset = df[mask].copy()

    # Add ratio column  -  sorted worst-first so >100% cases surface immediately
    if len(subset) > 0:
        loan = to_num(subset, "Loan Amount").replace(0, float("nan"))
        t_arr = (_to_num(subset, "ARREARS AGAINST INST") +
                 _to_num(subset, "ARREARS AGAINST EXP") +
                 _to_num(subset, "ARREARS AGAINST BC"))
        subset["Arrears Ratio %"] = (t_arr / loan * 100).round(1)
        subset = subset.sort_values("Arrears Ratio %", ascending=False)

    display_cols = ["Arrears Ratio %"] + _safe_cols(df, ALERT_DISPLAY_COLS)
    return {
        "title": "High Arrears: Loan at Risk",
        "subtitle": f"Inst+Exp+BC arrears exceed {HIGH_ARREARS_LOAN_RATIO:.0%} of original loan - Highly critical cases: Potential Write-off",
        "severity": "critical",
        "count": account_count(subset),
        "pos": _to_num(subset, "SOH").sum(),
        "closing_arrears": _to_num(subset, "Closing Arrears").sum(),
        "df": _fmt_dates(subset[[c for c in display_cols if c in subset.columns]]),
        # subset already carries the computed "Arrears Ratio %" column (added
        # above) -- included here deliberately, since it's the exact figure
        # that made each row qualify, not just a raw Excel column.
        "df_full": _fmt_dates(subset),
        "icon": "🔥",
        "action": "Prioritise >100% cases for legal/recovery, these may be unrecoverable without immediate escalation.",
    }


def run_all_alerts(df: pd.DataFrame, recent_months: int = RECENT_ADVANCES_MONTHS, as_of=None) -> list:
    """Run all 6 alerts and return list sorted by severity.

    as_of is passed straight through to alert_recent_advances_at_risk (the
    only one of the 6 that's date-anchored) -- see that function's own
    docstring. Optional, defaults to None (wall-clock today), so every
    existing caller that doesn't pass it keeps its prior behavior unchanged."""
    alerts = [
        alert_non_starters(df),
        alert_colending_at_risk(df),
        alert_insurance_delinquency(df),
        alert_recent_advances_at_risk(df, recent_months, as_of=as_of),
        alert_easy_settlements(df),
        alert_high_arrears_ratio(df),
    ]
    return alerts
