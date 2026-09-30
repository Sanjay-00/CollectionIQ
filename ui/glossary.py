"""Plain-language definitions for the app's own analytical terms (not standard
NBFC vocabulary), shown as an ⓘ hover next to the term. Thresholds come from
config.py, so a definition can't drift from the rule it describes."""
from config import FLEET_MIN_LOANS, HARD_BUCKET_ARREARS_EMI_MIN, RECENT_ADVANCES_MONTHS
from ui.components import _esc

_H = HARD_BUCKET_ARREARS_EMI_MIN

_CHRONIC = f"No payment received in the last 3 months AND more than {_H} EMIs overdue."
_SHOCK = (f"{_H}+ EMIs overdue now, but NOT chronic: the customer paid something in the last "
          "3 months, so this is a recent slide, not a long silence.")
_FLEET = f"A customer (same mobile number) with {FLEET_MIN_LOANS} or more loans in the whole upload."

GLOSSARY: dict[str, str] = {
    "Chronic": _CHRONIC,
    "Chronic (3M+)": f"Number of chronic loans: {_CHRONIC[0].lower()}{_CHRONIC[1:]}",
    "Hard Bucket": f"{_H} or more EMIs overdue right now.",
    "Hard Bucket%": f"% of loans with {_H} or more EMIs overdue right now.",
    "Sudden shock": _SHOCK,
    "Chronic + Hard": f"Chronic and still {_H}+ EMIs overdue: legal / write-off candidates.",
    "Chronic + Not Hard": f"Chronic, but now under {_H} EMIs overdue: part-paid after a long silence. Call before they slip back.",
    "Shock + Hard": _SHOCK + " Worth a restructuring conversation.",
    "Neither": f"Behind on payments, but neither chronic nor {_H}+ EMIs overdue: normal early delinquency.",
    "Insurance-Only": ("EMI fully paid; only the insurance/expense charge is unpaid. "
                       "Fix with a cash or WCL adjustment, not a credit problem."),
    "Installment-Only": "EMI unpaid; no unpaid insurance/expense charge.",
    "Both": "Both the EMI and the insurance/expense charge are unpaid.",
    "Delinquent Accounts": "Loans with any EMI or charge overdue (Arrears/EMI above 0).",
    "Dominant Driver": ("The cause behind the biggest share of this region's delinquent loans: "
                        "insurance-only, chronic, sudden shock, fleet operators or recent advances."),
    "Driver Share %": "Share of the region's delinquent loans explained by the dominant driver.",
    "Fleet operator": _FLEET,
    "Concern Score": ("0-100 rank of this branch against all other branches (higher = worse), blending "
                      "NPA% (45%), Hard Bucket% (25%), Roll Fwd% (20%) and chronic loans (10%)."),
}

# The Dominant Driver values (analysis/root_cause.py::_DRIVER_LABELS).
DRIVER_HELP: dict[str, str] = {
    "Insurance-driven delinquency": GLOSSARY["Insurance-Only"],
    "Chronic non-payer buildup": _CHRONIC,
    "Sudden-shock deterioration": _SHOCK,
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
