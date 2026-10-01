"""Plain-language definitions for the app's own analytical terms (not standard
NBFC vocabulary), shown as an ⓘ hover next to the term. Thresholds come from
config.py and group names from analysis/root_cause.py, so a definition can't
drift from the rule or label it describes."""
from analysis.root_cause import EARLY_DELINQUENCY, HARD_NOT_PAYING, HARD_STILL_PAYING, WAS_SILENT
from config import CONCERN_SCORE_WEIGHTS, FLEET_MIN_LOANS, HARD_BUCKET_ARREARS_EMI_MIN, INSURANCE_EXP_ARREARS_MIN, RECENT_ADVANCES_MONTHS
from ui.components import _esc

_H = HARD_BUCKET_ARREARS_EMI_MIN

_NOT_PAYING = f"No payment received in the last 3 months AND more than {_H} EMIs overdue."
_STILL_PAYING = (f"{_H} or more EMIs overdue, but the customer has paid something in the last 3 months: "
                 "trying, but not keeping up.")
_FLEET = f"A customer (same mobile number) with {FLEET_MIN_LOANS} or more loans in the whole upload."

GLOSSARY: dict[str, str] = {
    "Not Paying 3M+": f"Loans with {_NOT_PAYING[0].lower()}{_NOT_PAYING[1:]}",
    "Not Paying 3M+%": f"% of the unit's loans with {_NOT_PAYING[0].lower()}{_NOT_PAYING[1:]}",
    "NPA% (SOH)": "NPA measured by money: SOH of NPA loans as % of total SOH (NPA% counts loans instead).",
    "NPA % (SOH)": "NPA measured by money: SOH of NPA loans as % of total SOH (NPA % counts loans instead).",
    "Hard Bucket": f"{_H} or more EMIs overdue right now.",
    "Hard Bucket%": f"% of loans with {_H} or more EMIs overdue right now.",
    HARD_NOT_PAYING: (f"{_H} or more EMIs overdue AND no payment in the last 3 months. "
                      "Legal, repossession or write-off candidates."),
    HARD_STILL_PAYING: _STILL_PAYING + " Worth a restructuring conversation.",
    WAS_SILENT: (f"Paid nothing for 3+ months at some point, but now under {_H} EMIs overdue after "
                 "paying part of it down. Call before they go silent again."),
    EARLY_DELINQUENCY: (f"Behind on payments, but under {_H} EMIs overdue and not silent for 3 months. "
                        "Normal follow-up by the field executive."),
    "Insurance-Only": (f"EMI fully paid; only the insurance/expense charge (over ₹{INSURANCE_EXP_ARREARS_MIN:,}) "
                       "is unpaid. Fix with a cash or WCL adjustment, not a credit problem."),
    "Other": (f"Behind for another reason: a small insurance charge (₹{INSURANCE_EXP_ARREARS_MIN:,} or less) "
              "or only bounce/penal charges."),
    "Installment-Only": "EMI unpaid; no unpaid insurance/expense charge.",
    "Both": "Both the EMI and the insurance/expense charge are unpaid.",
    "Delinquent Accounts": "Loans with any EMI or charge overdue (Arrears/EMI above 0).",
    "Dominant Driver": ("The cause behind the biggest share of this region's delinquent loans: "
                        "insurance-only, not paying for 3+ months, deep arrears still paying, fleet operators "
                        "or recent advances."),
    "Driver Share %": "Share of the region's delinquent loans explained by the dominant driver.",
    "Fleet operator": _FLEET,
    "Concern Score": ("0-100 rank of this branch against all other branches (higher = worse), blending "
                      + ", ".join(f"{k} ({round(w * 100)}%)" for k, w in CONCERN_SCORE_WEIGHTS.items())
                      + ". A part with no data (e.g. Roll Fwd% without last month) is left out."),
}

# The Dominant Driver values (analysis/root_cause.py::_DRIVER_LABELS).
DRIVER_HELP: dict[str, str] = {
    "Insurance-driven delinquency": GLOSSARY["Insurance-Only"],
    "Not paying for 3+ months": _NOT_PAYING,
    "Deep arrears, still paying": _STILL_PAYING,
    "Fleet-operator concentration": "Delinquent loans belonging to fleet operators. " + _FLEET,
    "Recent-advance (sourcing/underwriting) quality": (
        f"Delinquent loans agreed in the {RECENT_ADVANCES_MONTHS} months before the reporting month: "
        "points to sourcing or underwriting, not collections."),
}


def help_for(term: str) -> str | None:
    """Definition for a column/label; a "... %" column shares its base term's."""
    return GLOSSARY.get(term) or GLOSSARY.get(term.removesuffix(" %")) or DRIVER_HELP.get(term)


def info_icon(text: str | None, color: str = "#9ca3af") -> str:
    """An ⓘ whose hover shows `text` (HTML, for st.markdown). Empty if no text."""
    if not text:
        return ""
    return (f'<span title="{_esc(text)}" style="cursor:help;color:{color};font-weight:400;'
            f'font-size:0.9em;margin-left:3px;">&#9432;</span>')
