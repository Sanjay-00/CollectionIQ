"""
Smart Alerts — pre-computed, rule-based flags that run on every LCC upload.
No LLM. Pure pandas. Always accurate.
"""
import pandas as pd
from datetime import date
from dateutil.relativedelta import relativedelta


def _fmt_dates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.date
    return df

# Fixed columns shown in every alert drilldown table
ALERT_DISPLAY_COLS = [
    "Loan No", "Zone", "RegionName", "Unit", "Ag_Date", "MNT NAME",
    "Due Dt", "Tenure", "Loan Status", "Loan Amount", "Veh ID", "Cust Name",
    "Guar Name", "Cust Mob No", "Guar Mob No", "Vehicle Description",
    "Month Due-Inst", "Month Due-Exp", "MONTH DUE (BC)", "MONTH DUE PC",
    "Month Receipt Amount", "Closing Arrears", "Arrears against Inst+Exp",
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
        "pos": _to_num(subset, "POS").sum(),
        "closing_pc": _to_num(subset, "ClosingPC").sum(),
        "df": _fmt_dates(subset[_safe_cols(df, ALERT_DISPLAY_COLS)]),
        "icon": "🚨",
        "action": "Immediate field visit required. Check if disbursement reached customer.",
    }


def alert_insurance_delinquency(df: pd.DataFrame) -> dict:
    """Customers with zero/negative installment demand but positive expense (insurance) demand.
    Customer is paying EMI but insurance charge is creating artificial delinquency."""
    inst = _to_num(df, "Month Due-Inst")
    exp  = _to_num(df, "Month Due-Exp")
    arrears = _to_num(df, "Arrears / EMI")
    mask = (inst == 0) & (exp > 0) & (arrears > 0)
    subset = df[mask]
    return {
        "title": "Insurance-Driven Delinquency",
        "subtitle": "EMI paid but insurance charge causing default",
        "severity": "high",
        "count": subset["Loan No"].nunique() if "Loan No" in subset.columns else len(subset),
        "pos": _to_num(subset, "POS").sum(),
        "closing_pc": _to_num(subset, "ClosingPC").sum(),
        "df": _fmt_dates(subset[_safe_cols(df, ALERT_DISPLAY_COLS)]),
        "icon": "⚠️",
        "action": "Settle insurance by cash or convert to child loan (EMI). Do not mark as willful default.",
    }


def alert_easy_settlements(df: pd.DataFrame) -> dict:
    """Accounts where closing arrears < 1000 — small amount, easy to clear."""
    closing = _to_num(df, "Closing Arrears")
    mask = (closing > 0) & (closing < 1000)
    subset = df[mask]
    return {
        "title": "Easy Settlements",
        "subtitle": "Closing arrears < ₹1,000 — quick wins",
        "severity": "medium",
        "count": subset["Loan No"].nunique() if "Loan No" in subset.columns else len(subset),
        "pos": _to_num(subset, "POS").sum(),
        "closing_pc": _to_num(subset, "ClosingPC").sum(),
        "df": _fmt_dates(subset[_safe_cols(df, ALERT_DISPLAY_COLS)]),
        "icon": "💡",
        "action": "One call / one visit can clear these. Assign to executives for same-day closure.",
    }


def alert_recent_advances_at_risk(df: pd.DataFrame, months: int = 12) -> dict:
    """Loans sanctioned in the last N months that already have delinquencies."""
    cutoff = pd.Timestamp(date.today() - relativedelta(months=months))
    ag = df["Ag_Date"] if "Ag_Date" in df.columns else pd.Series(pd.NaT, index=df.index)
    arrears = _to_num(df, "Arrears / EMI")
    mask = (ag >= cutoff) & (arrears > 0)
    subset = df[mask]
    return {
        "title": "Recent Advances at Risk",
        "subtitle": f"Sanctioned in last {months} months — already delinquent",
        "severity": "high",
        "count": subset["Loan No"].nunique() if "Loan No" in subset.columns else len(subset),
        "pos": _to_num(subset, "POS").sum(),
        "closing_pc": _to_num(subset, "ClosingPC").sum(),
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
        "subtitle": "Partner bank exposure — must not default",
        "severity": "critical",
        "count": subset["Loan No"].nunique() if "Loan No" in subset.columns else len(subset),
        "pos": _to_num(subset, "POS").sum(),
        "closing_pc": _to_num(subset, "ClosingPC").sum(),
        "df": _fmt_dates(subset[_safe_cols(df, ALERT_DISPLAY_COLS)]),
        "icon": "🏦",
        "action": "Escalate immediately to Regional Manager. Partner bank SLA may be breached.",
    }


def run_all_alerts(df: pd.DataFrame, recent_months: int = 12) -> list:
    """Run all 5 alerts and return list sorted by severity."""
    alerts = [
        alert_non_starters(df),
        alert_colending_at_risk(df),
        alert_insurance_delinquency(df),
        alert_recent_advances_at_risk(df, recent_months),
        alert_easy_settlements(df),
    ]
    return alerts
