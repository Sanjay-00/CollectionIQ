"""
The plain-words layer of an Investigator answer: a one-line headline read
straight off the step's own table (no AI call, so it can't invent a number),
the Root Cause "why" split for a single branch / region / executive, plain
step names for "How I got this", and the table as a business reader sees it
(internal columns out, file column names in plain words).

Everything here is display only. The stored result_df, the downloads and
the "Analyse further" controls keep the step's own columns.
"""
from __future__ import annotations

import pandas as pd

from registry.ontology import CONCEPTS

# "How I got this" names a step in plain words, not its code name.
STEP_LABELS: dict[str, str] = {
    "dimension_breakdown": "Compared units side by side",
    "entity_summary": "Summary of one unit",
    "underperformer_quartile": "Weakest quarter of executives",
    "bottom_n_per_group": "Weakest executives in each group",
    "executive_roster": "List of executives",
    "non_paying_customers": "Customers not paying 3 months",
    "concept_filter": "Loans matching a rule",
    "concept_breakdown": "Rule broken down by unit",
    "worst_loans_by_metric": "Loans driving a measure",
    "roll_rate_summary": "Bucket movement",
    "roll_rate_by_dimension": "Bucket movement by unit",
    "vintage_summary": "Results by disbursement month",
    "demand_summary": "Old arrears vs this month's EMI",
    "new_advances_summary": "New business",
    "new_advances_by_dimension": "New business by unit",
    "new_advances_trend": "New business by month",
    "product_analysis": "Results by product",
    "top_accounts": "Largest delinquent accounts",
    "fleet_exposure": "Customers with several loans",
    "fleet_defaulters": "Fleet owners who are delinquent",
    "repossession_list": "Accounts eligible for repossession",
    "good_customers": "Good customers",
    "priority_menu": "What to chase first",
    "priority_accounts": "Full priority list",
    "top_closing_arrears": "Largest closing arrears",
    "high_arrears_at_risk": "High arrears against loan amount",
    "priority_by_executive": "Priority cases by executive",
    "intersect_entities": "Executives on several lists",
    "customer_loan_book": "One customer's loans",
    "portfolio_query": "A list or count question",
}

_PLURAL = {"region": "regions", "branch": "branches", "zone": "zones", "bu": "business units",
           "executive": "executives"}

# The executive scorecard spells metrics with a space; everything else
# without. Shown one way.
_METRIC_SHOWN = {"Collection %": "Collection%", "NPA %": "NPA%", "Delinquency %": "Delinquency%",
                 "Strike Rate %": "Strike%", "Hard Bucket %": "Hard Bucket%", "SMA-2 %": "SMA-2%"}

# File column names in plain words, for the table on screen.
_COLUMN_SHOWN = {"RegionName": "Region", "Unit": "Branch", "MNT NAME": "Executive", "curr_bucket": "Bucket",
                 "Cust Name": "Customer", "Cust Mob No": "Mobile", "Delta": "Change MoM"}
_COLUMN_HIDDEN = {"Higher Is Better", "curr_score", "prev_score",
                  "Not Paying 3M (receipts)"}     # every row of a not-paying list says Y
# An executive ranking keeps the figures a leader reads; the download has all 18.
_EXECUTIVE_SHOWN = ["Executive (Branch)", "Accounts", "Collection%", "Delinquent", "Delinquency%", "NPA%",
                    "Strike%", "Hard Bucket%", "Total SOH (L)"]

# Rankings longer than this show their worst rows first; the rest is one tick away.
RANKING_STEP_TYPES = {"dimension_breakdown", "underperformer_quartile", "executive_roster", "bottom_n_per_group"}
RANKING_SHOWN = 10
RANKING_LONG = 15

_LOAN_LIST_LABEL = {
    "non_paying_customers": "loans not paying for 3 months",
    "worst_loans_by_metric": "loans driving {metric}",
    "top_accounts": "largest delinquent accounts",
    "repossession_list": "accounts eligible for repossession",
    "high_arrears_at_risk": "loans with high arrears against the loan amount",
    "fleet_defaulters": "delinquent loans of fleet owners",
    "top_closing_arrears": "loans with the largest closing arrears",
    "good_customers": "good customers",
    "customer_loan_book": "loans for this customer",
    "priority_accounts": "priority loans",
}


def metric_shown(metric) -> str:
    m = str(metric or "").strip()
    return _METRIC_SHOWN.get(m, m)


def _pct(v) -> str:
    return f"{float(v):,.2f}%"


def _num(v):
    v = pd.to_numeric(v, errors="coerce")
    return None if pd.isna(v) else float(v)


def _first_present(df: pd.DataFrame, names) -> str | None:
    return next((c for c in names if c in df.columns), None)


def _name_of(entity_value) -> str:
    if isinstance(entity_value, (list, tuple)):
        return f"{entity_value[0]} ({entity_value[1]})" if len(entity_value) > 1 and entity_value[1] else str(entity_value[0])
    return str(entity_value)


def _soh_cr(df: pd.DataFrame) -> float | None:
    if "SOH" not in df.columns:
        return None
    return float(pd.to_numeric(df["SOH"], errors="coerce").fillna(0).sum()) / 1e7


def _most_in(df: pd.DataFrame) -> str:
    """ ' Most in AKLUJ (12).' for a loan list spread over 2+ branches."""
    if "Unit" not in df.columns or df.empty:
        return ""
    counts = df["Unit"].astype(str).str.strip().value_counts()
    if len(counts) < 2:
        return ""
    return f" Most in {counts.index[0]} ({int(counts.iloc[0]):,})."


def _loan_list(step_type: str, df: pd.DataFrame, params: dict) -> str:
    if step_type == "concept_filter":
        concept = params.get("concept")
        what = f"loans: {CONCEPTS.get(concept, {}).get('label', str(concept).replace('_', ' '))}"
    else:
        what = _LOAN_LIST_LABEL[step_type].format(metric=metric_shown(params.get("metric")))
    if len(df) == 1:
        for many, one in (("loans", "loan"), ("accounts", "account"), ("customers", "customer")):
            what = what.replace(many, one, 1)
    where = f" in {params['scope_value']}" if params.get("scope_value") and not isinstance(
        params.get("scope_value"), (list, tuple)) else ""
    soh = _soh_cr(df)
    soh_txt = f", ₹{soh:,.2f} Cr SOH" if soh is not None else ""
    return f"{len(df):,} {what}{where}{soh_txt}.{_most_in(df)}"


def _ranking(df: pd.DataFrame, params: dict) -> str | None:
    """dimension_breakdown: rows already sorted worst first by its metric."""
    label_col = df.columns[0]
    dim = str(params.get("dimension") or "unit").lower()
    metric = metric_shown(params.get("metric")) if metric_shown(params.get("metric")) in df.columns else "Collection%"
    if metric not in df.columns or df.empty:
        return None
    where = f" in {params['scope_value']}" if params.get("scope_value") else ""
    if len(df) == 1:
        return f"{df.iloc[0][label_col]}: {metric} {_pct(df.iloc[0][metric])}."
    worst, best = df.iloc[0], df.iloc[-1]
    return (f"{len(df):,} {_PLURAL.get(dim, dim)}{where}. Weakest on {metric}: {worst[label_col]} at "
            f"{_pct(worst[metric])}. Strongest: {best[label_col]} at {_pct(best[metric])}.")


def _summary(df: pd.DataFrame, entity_value) -> str | None:
    """entity_summary: the three numbers a leader asks first, and how each
    moved MoM when last month's file is loaded. Changes are stated, not judged."""
    if "Metric" not in df.columns or df.empty:
        return None
    rows = {metric_shown(m): r for m, r in zip(df["Metric"], df.to_dict("records"))}
    parts, moves = [], []
    for m in ("Collection%", "NPA%", "Delinquency%"):
        r = rows.get(m)
        cur = _num(r["Current"]) if r else None
        if cur is None:
            continue
        parts.append(f"{m} {_pct(cur)}")
        d = _num(r.get("Delta")) if r else None
        if d is not None and d != 0:
            moves.append(f"{m} {'up' if d > 0 else 'down'} {abs(d):.2f} % points MoM")
    if not parts:
        return None
    text = f"{_name_of(entity_value)}: {', '.join(parts)}."
    return text + (f" {'; '.join(moves)}." if moves else "")


def _executives(step_type: str, df: pd.DataFrame, params: dict) -> str:
    metric = metric_shown(params.get("metric"))
    col = next((c for c in df.columns if metric_shown(c) == metric), None)
    who = "Executive (Branch)" if "Executive (Branch)" in df.columns else None
    where = f" in {params['scope_value']}" if params.get("scope_value") else ""
    first = f" First: {df.iloc[0][who]} at {_pct(df.iloc[0][col])}." if col and who and not df.empty else ""
    if step_type == "underperformer_quartile":
        return f"{len(df):,} executives{where} in the weakest quarter on {metric}.{first}"
    if step_type == "bottom_n_per_group":
        return f"{len(df):,} executives: the weakest {params.get('n', 2)} per {params.get('group_col', 'group')} on {metric}."
    if params.get("n") and col:
        return f"{len(df):,} executives{where} ranked by {metric}.{first}"
    return f"{len(df):,} executives{where}."


def headline(step_type: str, result_df: pd.DataFrame, params: dict | None, entity_value=None) -> str | None:
    """One sentence read off the step's own table, or None when a step has
    nothing simple to say (the table speaks for itself)."""
    params = params or {}
    if result_df is None or result_df.empty:
        return None
    try:
        if step_type == "dimension_breakdown":
            return _ranking(result_df, params)
        if step_type == "entity_summary":
            return _summary(result_df, entity_value if entity_value is not None else params.get("entity_value"))
        if step_type in ("underperformer_quartile", "bottom_n_per_group", "executive_roster"):
            return _executives(step_type, result_df, params)
        if step_type in _LOAN_LIST_LABEL or step_type == "concept_filter":
            return _loan_list(step_type, result_df, params)
        where = f" in {params['scope_value']}" if params.get("scope_value") and not isinstance(
            params.get("scope_value"), (list, tuple)) else ""
        if step_type == "vintage_summary" and {"Disbursement Month", "NPA%"} <= set(result_df.columns):
            worst = result_df.loc[pd.to_numeric(result_df["NPA%"], errors="coerce").idxmax()]
            return (f"Highest NPA% by disbursement month{where}: {worst['Disbursement Month']} "
                    f"at {_pct(worst['NPA%'])}.")
        if step_type == "priority_menu" and {"Category", "Count"} <= set(result_df.columns):
            top = result_df.loc[pd.to_numeric(result_df["Count"], errors="coerce").idxmax()]
            return (f"Largest group{where}: {top['Category']} ({int(top['Count']):,} loans). "
                    "Pick one below to see its loans.")
    except (KeyError, IndexError, TypeError, ValueError):
        return None                # a headline never stops the table from showing
    return None


def why_split(scope_df: pd.DataFrame) -> dict | None:
    """The Root Cause tab's own split of one unit's delinquent loans
    (analysis.root_cause.root_cause_snapshot, the same numbers as the
    Dashboard's "Why It's Moving"): {"delinquent": n, "rows": [(label, n, pct)]}."""
    from analysis.root_cause import (
        EARLY_DELINQUENCY, GROUP_LABELS, HARD_NOT_PAYING, HARD_STILL_PAYING, WAS_SILENT, root_cause_snapshot,
    )
    if scope_df is None or scope_df.empty:
        return None
    snap = root_cause_snapshot(scope_df)
    if not snap or not snap.get("delinquent"):
        return None
    order = (HARD_NOT_PAYING, HARD_STILL_PAYING, WAS_SILENT, EARLY_DELINQUENCY, "Insurance-Only")
    rows = [(GROUP_LABELS[g], snap["groups"][g][0], snap["groups"][g][1]) for g in order if g in snap["groups"]]
    return {"delinquent": snap["delinquent"], "rows": rows}


def display_table(step_type: str | None, result_df: pd.DataFrame) -> pd.DataFrame:
    """The step's table as a business reader sees it: internal and empty
    columns out, file column names in plain words, one metric spelling."""
    df = result_df
    hidden = [c for c in df.columns if c in _COLUMN_HIDDEN or str(c).startswith("_")]
    if len(df):
        hidden += [c for c in df.columns if c not in hidden and df[c].isna().all()]
    if "Executive (Branch)" in df.columns:          # the same two facts again, as separate columns
        hidden += [c for c in ("MNT NAME", "Unit") if c in df.columns]
    df = df.drop(columns=hidden)
    if step_type == "entity_summary" and "Metric" in df.columns:
        df = df.assign(Metric=df["Metric"].map(metric_shown))
    renamed = {c: _COLUMN_SHOWN[c] for c in df.columns if c in _COLUMN_SHOWN and _COLUMN_SHOWN[c] not in df.columns}
    renamed.update({c: metric_shown(c) for c in df.columns if c in _METRIC_SHOWN and metric_shown(c) not in df.columns})
    df = df.rename(columns=renamed)
    if "Executive (Branch)" in df.columns and "Metric" not in df.columns:
        df = df[[c for c in _EXECUTIVE_SHOWN if c in df.columns]]
    return df
