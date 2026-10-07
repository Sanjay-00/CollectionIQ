"""Every loan-level action list in one place, for the Action Lists tab.

The lists used to live in four tabs (Dashboard call lists, Alerts, Migration's
"loans to call", Portfolio Intelligence's action lists) and three of them were
the same loans under two names. Here each list appears once, in four groups:

  Call this week        today's buckets (action_center's focus groups)
  Slipped this month    loans that moved to a worse bucket (roll_flow)
  Risk flags            the smart alerts not already covered above
  Recovery and relationship  repossession candidates, good customers

Each list: {"group", "name", "action", "loans" (display frame, largest SOH
first, except good customers: lowest first), "count", "soh_cr", "prev" (last
month's count, when known)}.
Pure pandas, no LLM.
"""
import pandas as pd

from analysis import roll_flow as rf
from analysis.action_center import _RENAME, _focus_groups, call_lists
from analysis.exposure import compute_good_customers, compute_repossession_list
from config import ROLL_STEPS
from utils import to_num

GROUPS = ("Call this week", "Slipped this month", "Risk flags", "Recovery and relationship")

# Alerts whose loans are exactly a "Call this week" list (same rule, same loans).
_SAME_AS_CALL_LIST = {"Non Starters", "Insurance-Driven Delinquency"}

_SLIP_ACTION = {
    "1-30 → SMA-1+": "Was 1-30 DPD last month, SMA-1 or worse now: get an EMI before it reaches SMA-2.",
    "SMA-1 → SMA-2+": "Was SMA-1 last month, SMA-2 or NPA now: field visit before it becomes NPA.",
    "SMA-2 → NPA": "Became NPA this month: recovery or repossession decision.",
}
_COLUMN_NAMES = {**_RENAME, "Last Month": "Last Month Bucket", "Now": "Bucket Now"}
# Back to the upload's own column names (for a download that adds every upload column).
RAW_NAMES = {shown: raw for raw, shown in _RENAME.items()}


def _entry(group: str, name: str, action: str, loans: pd.DataFrame, soh: pd.Series, prev=None,
           smallest_first: bool = False) -> dict:
    loans = loans.rename(columns=_COLUMN_NAMES)
    if "Loan No" in loans.columns:
        loans = loans.assign(SOH=loans["Loan No"].map(soh)).sort_values("SOH", ascending=smallest_first, kind="stable")
    loans = loans.reset_index(drop=True)
    count = int(loans["Loan No"].nunique()) if "Loan No" in loans.columns else len(loans)
    soh_cr = round(float(loans["SOH"].sum()) / 1e7, 2) if "SOH" in loans.columns else 0.0
    return {"group": group, "name": name, "action": action, "loans": loans,
            "count": count, "soh_cr": soh_cr, "prev": prev}


def _loan_count(loans: pd.DataFrame) -> int:
    return int(loans["Loan No"].nunique()) if "Loan No" in loans.columns else len(loans)


def _prev_counts(df_prev: pd.DataFrame | None, prev_as_of) -> dict:
    """Last month's size of each list, by name, from last month's file with the
    same rules. A list whose rule needs the bucket from the month before
    (new defaulters, the SMA lists that leave them out, the slips) is only
    counted when last month's file carries prev_bucket; otherwise it is left
    out, so the screen shows "-" rather than a count built on another rule."""
    if df_prev is None or df_prev.empty or "Loan No" not in df_prev.columns or "curr_bucket" not in df_prev.columns:
        return {}
    has_roll = rf.has_roll_data(df_prev)
    needs_roll = {"New defaulters", "SMA-2: last chance before NPA", "SMA-1: stop the slide",
                  "1-30 DPD: bring them current"}
    ids = df_prev["Loan No"]
    out = {name: int(ids[mask.fillna(False)].nunique()) for name, _, mask in _focus_groups(df_prev)
           if has_roll or name not in needs_roll}
    if has_roll:
        for label, from_bucket in list(ROLL_STEPS.items())[1:]:
            out[label] = _loan_count(rf.loans_that_rolled(df_prev, from_bucket))
    out["Repossession candidates"] = _loan_count(compute_repossession_list(df_prev, as_of=prev_as_of))
    out["Good customers"] = _loan_count(compute_good_customers(df_prev))
    return out


def build_lists(df_curr: pd.DataFrame, as_of, alerts: list, alerts_prev: list | None = None,
                df_prev: pd.DataFrame | None = None, prev_as_of=None) -> list[dict]:
    """All lists in display order. `alerts` / `alerts_prev`: smart_alerts.run_all_alerts
    for this month and last month (the app caches them). `df_prev` / `prev_as_of`:
    last month's file and month, for each list's count last month."""
    if df_curr.empty or "Loan No" not in df_curr.columns:
        return []
    soh = to_num(df_curr, "SOH", fill=0).groupby(df_curr["Loan No"]).first()
    prev = _prev_counts(df_prev, prev_as_of)
    out = []

    actions = {name: action for name, action, _ in _focus_groups(df_curr)}
    for t in call_lists(df_curr):
        out.append(_entry(GROUPS[0], t["name"], actions.get(t["name"], ""), t["loans"], soh, prev.get(t["name"])))

    if rf.has_roll_data(df_curr):
        for label, from_bucket in list(ROLL_STEPS.items())[1:]:   # STD → behind = "New defaulters" above
            out.append(_entry(GROUPS[1], label, _SLIP_ACTION.get(label, ""),
                              rf.loans_that_rolled(df_curr, from_bucket), soh, prev.get(label)))

    prev_alerts = {a["title"]: a["count"] for a in (alerts_prev or [])}
    for a in alerts:
        if a["title"] in _SAME_AS_CALL_LIST or not a["count"]:
            continue
        out.append(_entry(GROUPS[2], a["title"], a.get("action", ""), a["df"], soh, prev_alerts.get(a["title"])))

    repo = compute_repossession_list(df_curr, as_of=as_of)
    out.append(_entry(GROUPS[3], "Repossession candidates",
                      "SMA-2 or NPA, agreed recently (the vehicle still has value); seized-and-sold left out.",
                      repo, soh, prev.get("Repossession candidates")))
    out.append(_entry(GROUPS[3], "Good customers",
                      "Most of the tenure done and everything paid: offer a top-up or a new loan "
                      "(lowest SOH first: the easiest to refinance).",
                      compute_good_customers(df_curr), soh, prev.get("Good customers"), smallest_first=True))
    return [e for e in out if e["count"]]


def distinct_loans(lists: list[dict], group: str | None = None) -> int:
    """Loans across lists (a loan can be in several), optionally one group only."""
    ids = [e["loans"]["Loan No"] for e in lists
           if (group is None or e["group"] == group) and "Loan No" in e["loans"].columns]
    return int(pd.concat(ids).nunique()) if ids else 0
