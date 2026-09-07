"""
Pure-pandas step functions for the Investigator drill-down assistant.

Each function is ONE vectorized pass over its input (never a Python loop
calling itself once per group/entity -- see compute_executive_scorecard's
own docstring, analysis/executive_scorecard.py:24-31, for why that matters),
returns a single flat pd.DataFrame (the auditable, downloadable table), and
contains zero LLM calls. Gemini's role (investigator/llm.py) is routing free
text to one of these functions with parameters -- never computing the
numbers itself. This keeps the same "vocabulary, not logic, in the LLM"
discipline CLAUDE.md documents for graph.py's IR-1/compiler split, applied
to the Investigator's own smaller, fixed step vocabulary.

Every step is dispatched through STEP_REGISTRY at the bottom of this file.
"""
from __future__ import annotations

import pandas as pd

from agents.data_executor import _COL_COMPARE_OPS, _build_mask
from analysis.executive_scorecard import compute_executive_scorecard, rank_by_metric
from analysis.portfolio_intelligence import (
    _branch_aggregates,
    compute_fleet_exposure,
    compute_good_customers,
    compute_new_advances,
    compute_new_advances_by_dimension,
    compute_new_advances_trend,
    compute_overdue_demand_scorecard,
    compute_product_analysis,
    compute_repossession_list,
    compute_top_accounts,
    roll_new_advances_trend,
)
from analysis.roll_rate import VALID_BUCKETS, compute_roll_rate_matrix
from compiler.core import _expand_filters
from config import FLEET_MIN_LOANS, HARD_BUCKET_ARREARS_EMI_MIN
from registry.semantic_model import resolve_dimension
from smart_alerts import alert_high_arrears_ratio
from utils import BUCKET_SCORE, _mom_pct, to_num

# ── Metric direction metadata ────────────────────────────────────────────────
# Single source of truth for "is a higher value of this metric good," shared
# by every step below that ranks/filters entities by a metric. Avoids the
# double-sign-flip bug class CLAUDE.md documents for _kpi_card_html's
# inverse/good_override contract -- callers may still override explicitly
# (e.g. for a metric not listed here), but nothing here silently assumes
# "lowest raw value = worst" the way a naive bottom-quartile cut would for
# NPA%/Hard Bucket% (where the highest value is the worst one).
METRIC_DIRECTION: dict[str, bool] = {
    "Collection %": True, "Collection%": True,
    "Strike Rate %": True, "Strike%": True,
    "NPA %": False, "NPA%": False,
    "Hard Bucket %": False, "Hard Bucket%": False,
    "SMA-2 %": False, "SMA-2%": False,
}

_BRANCH_METRICS  = ["Collection%", "NPA%", "Strike%", "Hard Bucket%", "SOH (Cr)", "Accounts"]
_EXEC_METRICS    = ["Collection %", "NPA %", "Strike Rate %", "Hard Bucket %", "Total SOH (L)", "Accounts"]
# region/zone/BU go through the generic _dimension_metrics aggregator
# (below), which doesn't compute Strike% -- hence a shorter metric list
# than _BRANCH_METRICS, not an oversight.
_GENERIC_DIMENSION_METRICS = ["Collection%", "NPA%", "Hard Bucket%", "SOH (Cr)", "Accounts"]
_GENERIC_ENTITY_TYPES = {"region", "zone", "bu"}
_MONEY_METRICS  = {"SOH (Cr)", "Total SOH (L)", "Total POS (L)", "Accounts", "Demand (L)", "Collected (L)"}

_CUSTOMER_DISPLAY_COLS = [
    "Cust Name", "Cust Mob No", "MNT NAME", "Unit", "RegionName",
    "Ag_Date", "POS", "Closing Arrears", "LCC%", "Arrears / EMI",
    "Last Receipt Date", "Last Receipt Amount", "Loan No", "curr_bucket", "SOH",
]


# Ops (agents/data_executor.py::_apply_condition) whose "value" is ITSELF a
# column name to compare against, not a literal -- a missing-column check
# must also validate these, not just each condition's own "column".
_CROSS_COLUMN_OPS = set(_COL_COMPARE_OPS) | {"bucket_worse_than", "bucket_better_than"}


def _missing_condition_columns(df: pd.DataFrame, conditions: list[dict]) -> list[str]:
    """Which columns a list of {"column", "op", "value"} conditions actually
    needs that AREN'T in df.columns -- catches the real silent-wrong-answer
    risk in agents/data_executor.py::_apply_condition, which SKIPS (treats
    as always-true) any single condition whose column is missing rather than
    erroring. For an AND-chain of conditions (every registered CONCEPT with
    more than one condition), silently dropping just one of them doesn't
    return zero rows, it returns MORE rows than should qualify -- a false
    positive at portfolio scale, worse than an empty table and just as
    undetectable without this check. Called BEFORE building any mask, never
    after, so a step can raise a clear "can't answer" error instead of
    quietly executing a filter it already knows is broken for this file."""
    needed: set[str] = set()
    for cond in conditions or []:
        col = cond.get("column")
        if col:
            needed.add(col)
        if cond.get("op") in _CROSS_COLUMN_OPS:
            ref_col = cond.get("value")
            if isinstance(ref_col, str):
                needed.add(ref_col)
    return sorted(c for c in needed if c not in df.columns)


def _higher_is_better(metric: str, override: bool | None = None) -> bool:
    """The registry (METRIC_DIRECTION) wins whenever the metric is
    registered, even if a caller passes a conflicting override -- override
    is a fallback for metrics with no registered direction, never a way to
    contradict one that exists. A live Gemini call gave
    higher_is_better=False for "Collection %" (registered as True,
    unambiguously -- collection% is always better higher, there's no
    genuine ambiguity here), which silently sorted best-performer-first for
    a question asking for the WORST performer. Trusting an override over a
    known, unambiguous registered direction defeats the entire point of
    METRIC_DIRECTION being a single source of truth immune to LLM error."""
    if metric in METRIC_DIRECTION:
        return METRIC_DIRECTION[metric]
    if override is not None:
        return override
    raise ValueError(
        f"Unknown metric direction for '{metric}' -- pass higher_is_better explicitly"
    )


# ── dimension_breakdown ───────────────────────────────────────────────────────

def dimension_breakdown(
    df_curr: pd.DataFrame,
    dimension: str,
    df_prev: pd.DataFrame | None = None,
    scope_col: str | None = None,
    scope_value: str | None = None,
    metric: str | None = None,
    higher_is_better: bool | None = None,
) -> pd.DataFrame:
    """One row per entity at `dimension`'s grain (branch/region/zone/bu) --
    for a PLURAL/comparative question ("how are my regions performing")
    with no single entity named, distinct from entity_summary (exactly ONE
    named entity) and underperformer_quartile (only the worst quartile).
    dimension is resolved via registry/semantic_model.py::resolve_dimension,
    the same org-hierarchy vocabulary (BU > Zone > Region > Branch) used
    elsewhere in this module. Delta columns are raw percentage-point
    differences (never a relative %-of-%% figure), same convention as
    entity_summary and compute_region_scorecard
    (analysis/portfolio_intelligence.py:401).

    Sorted worst-first by `metric` (defaulting to Collection%, ascending --
    the plain "how is everyone doing" case) -- a real gap this fixes: a
    question actually ABOUT a specific metric ("why is NPA more") used to
    still sort by Collection% regardless, which is irrelevant to what was
    asked. Direction is looked up via METRIC_DIRECTION unless
    higher_is_better is given explicitly, same convention as
    underperformer_quartile.

    scope_col/scope_value optionally narrow to one entity ONE LEVEL UP the
    hierarchy first (e.g. dimension="branch", scope_col="region",
    scope_value="CS Nagar" -> every branch within CS Nagar only) -- the
    natural "drill from a region into its branches" step, same scoping
    convention underperformer_quartile already uses.
    """
    col = resolve_dimension(dimension)[0]
    if df_curr is None or df_curr.empty or col not in df_curr.columns:
        return pd.DataFrame()

    if scope_col and scope_value:
        scope_resolved = resolve_dimension(scope_col)[0]
        if scope_resolved in df_curr.columns:
            df_curr = df_curr[df_curr[scope_resolved].astype(str).str.strip().str.upper() == str(scope_value).strip().upper()]
            if df_prev is not None and not df_prev.empty and scope_resolved in df_prev.columns:
                df_prev = df_prev[df_prev[scope_resolved].astype(str).str.strip().str.upper() == str(scope_value).strip().upper()]
    if df_curr.empty:
        return pd.DataFrame()

    curr = _dimension_metrics(df_curr, col)
    if curr.empty:
        return pd.DataFrame()

    if df_prev is not None and not df_prev.empty and col in df_prev.columns:
        prev = _dimension_metrics(df_prev, col)
        prev = prev.rename(columns={
            "Collection%": "_prev_coll", "NPA%": "_prev_npa", "Hard Bucket%": "_prev_hb",
        })[[col, "_prev_coll", "_prev_npa", "_prev_hb"]]
        curr = curr.merge(prev, on=col, how="left")
        curr["Δ Collection%"]  = (curr["Collection%"] - curr["_prev_coll"]).round(2)
        curr["Δ NPA%"]         = (curr["NPA%"] - curr["_prev_npa"]).round(2)
        curr["Δ Hard Bucket%"] = (curr["Hard Bucket%"] - curr["_prev_hb"]).round(2)
        curr = curr.drop(columns=["_prev_coll", "_prev_npa", "_prev_hb"])

    # Normalize the executive-grain metric spelling ("Collection %", with a
    # space -- compute_executive_scorecard's own column naming) to this
    # function's no-space column names ("Collection%") -- a live-Gemini
    # call produced the space-variant here, which silently fell back to the
    # default sort as if no metric were given at all, not a graceful
    # normalization. Code-level fix, not just prompt wording, since a
    # caller getting the spelling wrong should never silently do the wrong
    # thing.
    if metric:
        metric = metric.replace(" %", "%")
    sort_metric = metric if metric in curr.columns else "Collection%"
    hib = _higher_is_better(sort_metric, higher_is_better)
    return curr.sort_values(sort_metric, ascending=hib).reset_index(drop=True)


# ── roll_rate_summary ─────────────────────────────────────────────────────────

def roll_rate_summary(
    df_curr: pd.DataFrame,
    df_prev: pd.DataFrame,
    scope_col: str | None = None,
    scope_value: str | None = None,
) -> pd.DataFrame:
    """Root-cause MECHANISM for a moved NPA%/delinquency number, distinct
    from every other step here (all of which answer WHO/WHERE -- which
    branch, which executive). "Why is NPA up" decomposes into: roll-forward
    (existing accounts sliding into worse buckets -- a collections-
    execution problem), roll-backward (fewer accounts recovering -- also
    execution), and NPA formation (non-NPA accounts newly turning NPA this
    month). These need DIFFERENT fixes, so a leader deciding where to focus
    should see this before drilling into executives, not instead of it.

    Reuses analysis/roll_rate.py::compute_roll_rate_matrix (which itself
    calls compute_roll_rate_kpis) directly -- the SAME bucket-migration
    engine report_agent's bucket_migration section and the AI Query
    roll_rate_matrix VIEW already use, so "roll forward rate" can never
    drift into a second, independently-computed definition here.

    scope_col/scope_value optionally narrow to one branch/region/zone/BU
    BEFORE matching curr/prev on Loan No -- same scoping convention every
    other step in this module uses -- so the rates reflect just that
    entity's own accounts, not diluted by the rest of the portfolio.

    Returns a flat Metric/Value table (matching entity_summary's own
    Metric-column convention), not the wide migration matrix itself -- that
    matrix is a Plotly heatmap input (see build_roll_rate_heatmap), not a
    single downloadable "here's the audit trail" table, and this step's job
    is the headline numbers, not the full crosstab.
    """
    if df_curr is None or df_curr.empty or df_prev is None or df_prev.empty:
        return pd.DataFrame()

    if scope_col and scope_value:
        col = resolve_dimension(scope_col)[0]
        if col in df_curr.columns:
            df_curr = df_curr[df_curr[col].astype(str).str.strip().str.upper() == str(scope_value).strip().upper()]
        if col in df_prev.columns:
            df_prev = df_prev[df_prev[col].astype(str).str.strip().str.upper() == str(scope_value).strip().upper()]

    if df_curr.empty or df_prev.empty:
        return pd.DataFrame()

    _matrix, meta = compute_roll_rate_matrix(df_curr, df_prev)
    if meta["matched_count"] == 0:
        return pd.DataFrame()

    return pd.DataFrame([
        {"Metric": "Roll Forward Rate %",  "Value": meta["roll_forward_rate"]},
        {"Metric": "Roll Backward Rate %", "Value": meta["roll_backward_rate"]},
        {"Metric": "NPA Formation Rate %", "Value": meta["npa_formation_rate"]},
        {"Metric": "Matched Accounts",     "Value": meta["matched_count"]},
        {"Metric": "New Entries",          "Value": meta["new_entries"]},
        {"Metric": "Exits",                "Value": meta["exits"]},
    ])


def roll_rate_by_dimension(
    df_curr: pd.DataFrame,
    df_prev: pd.DataFrame,
    dimension: str,
) -> pd.DataFrame:
    """"Which branch had the most roll forward" -- roll_rate_summary alone
    can't answer this, it only ever computes ONE scope's rates at a time,
    never a ranked breakdown across many (confirmed as a real gap: a live
    user asked exactly this and got a clarification request, since no step
    could produce "Roll Forward Rate %" grouped by branch).

    Reuses BUCKET_SCORE (imported from utils, the SAME bucket-severity
    mapping analysis/roll_rate.py::compute_roll_rate_matrix/
    compute_roll_rate_kpis use internally) directly -- so "what counts as
    forward vs backward" can never drift into a second definition here,
    even though the GROUPED aggregation shape (one row per branch/region/
    zone/BU/executive) is new: compute_roll_rate_matrix itself only ever
    answers for one scope, by design (it returns a single portfolio-wide
    matrix, not a per-entity breakdown).

    ONE vectorized merge (curr x prev on Loan No) + ONE groupby(dimension)
    pass over the merged per-loan rows -- never a Python loop calling
    compute_roll_rate_matrix once per branch, which would re-scan the
    whole DataFrame per group (see this module's own docstring for why
    that pattern is avoided everywhere here).

    Groups by EVERY column resolve_dimension returns for `dimension`, not
    just the first -- "executive" resolves to ["MNT NAME", "Unit"]
    together, so two different people sharing a first name in different
    branches are never merged into one row (the same collision class
    investigator/guardrails.py's ambiguous-executive-name check exists to
    prevent, avoided here by construction).

    Sorted worst-first by Roll Forward Rate % (highest = most accounts
    sliding into worse buckets)."""
    if df_curr is None or df_curr.empty or df_prev is None or df_prev.empty:
        return pd.DataFrame()

    cols = [c for c in resolve_dimension(dimension) if c in df_curr.columns]
    if not cols:
        return pd.DataFrame()

    curr_slim = df_curr[["Loan No", "curr_bucket", *cols]]
    prev_slim = df_prev[["Loan No", "curr_bucket"]].rename(columns={"curr_bucket": "prev_bucket"})
    merged = curr_slim.merge(prev_slim, on="Loan No", how="inner")
    merged = merged[merged["prev_bucket"].isin(VALID_BUCKETS) & merged["curr_bucket"].isin(VALID_BUCKETS)]
    if merged.empty:
        return pd.DataFrame()

    prev_score = merged["prev_bucket"].map(BUCKET_SCORE)
    curr_score = merged["curr_bucket"].map(BUCKET_SCORE)
    merged["_forward"]     = curr_score > prev_score
    merged["_backward"]    = curr_score < prev_score
    merged["_pre_npa"]     = merged["prev_bucket"] != "NPA"
    merged["_formed_npa"]  = merged["_pre_npa"] & (merged["curr_bucket"] == "NPA")

    grouped = merged.groupby(cols, dropna=False).agg(
        **{"Matched Accounts": ("_forward", "size")},
        _forward_n=("_forward", "sum"),
        _backward_n=("_backward", "sum"),
        _pre_npa_n=("_pre_npa", "sum"),
        _formed_npa_n=("_formed_npa", "sum"),
    ).reset_index()

    grouped["Roll Forward Rate %"] = (
        grouped["_forward_n"] / grouped["Matched Accounts"] * 100
    ).round(2)
    grouped["Roll Backward Rate %"] = (
        grouped["_backward_n"] / grouped["Matched Accounts"] * 100
    ).round(2)
    grouped["NPA Formation Rate %"] = (
        grouped["_formed_npa_n"] / grouped["_pre_npa_n"].replace(0, pd.NA) * 100
    ).round(2).fillna(0.0)

    grouped = grouped.drop(columns=["_forward_n", "_backward_n", "_pre_npa_n", "_formed_npa_n"])
    return grouped.sort_values("Roll Forward Rate %", ascending=False).reset_index(drop=True)


def _dimension_metrics(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """One vectorized groupby(col) pass computing Accounts/Collection%/NPA%/
    Hard Bucket%/SOH -- precomputed boolean/numeric columns once, then a
    single .agg() per group, same pattern as compute_executive_scorecard's
    own docstring describes and justifies (analysis/executive_scorecard.py:24-31)."""
    d = df.copy()
    d["_demand"]    = to_num(d, "Net Collection Demand Inst+Exp+BC")
    d["_collected"] = to_num(d, "Month Collection (Excluding Reserve Collection)")
    d["_soh"]       = to_num(d, "SOH")
    d["_hard_bucket_flag"] = to_num(d, "Arrears / EMI") >= HARD_BUCKET_ARREARS_EMI_MIN
    has_bucket = "curr_bucket" in d.columns
    if has_bucket:
        d["_npa_flag"] = d["curr_bucket"] == "NPA"

    agg_cols = {"_demand": "sum", "_collected": "sum", "_soh": "sum", "_hard_bucket_flag": "sum"}
    if has_bucket:
        agg_cols["_npa_flag"] = "sum"
    grouped = d.groupby(col).agg(agg_cols)
    n = d.groupby(col).size()

    rows = []
    for key, row in grouped.iterrows():
        count = int(n.loc[key])
        demand = row["_demand"]
        rows.append({
            col:              key,
            "Accounts":       count,
            "Collection%":    round(row["_collected"] / demand * 100, 2) if demand > 0 else 0.0,
            "NPA%":           round(row["_npa_flag"] / count * 100, 2) if has_bucket and count > 0 else 0.0,
            "Hard Bucket%":   round(row["_hard_bucket_flag"] / count * 100, 2) if count > 0 else 0.0,
            "SOH (Cr)":       round(row["_soh"] / 1_00_00_000, 2),
        })
    return pd.DataFrame(rows)


# ── vintage_summary ───────────────────────────────────────────────────────────

def vintage_summary(
    df: pd.DataFrame,
    scope_col: str | None = None,
    scope_value: str | None = None,
    as_of=None,
) -> pd.DataFrame:
    """Root-cause MECHANISM's other half from roll_rate_summary: is
    delinquency concentrated in RECENT originations (an underwriting/
    sourcing-quality signal -- a DIFFERENT fix than a collections-execution
    problem) or spread evenly across the seasoned book? Neither
    dimension_breakdown (org grain) nor roll_rate_summary (this-month
    migration only) can answer that -- this groups by DISBURSEMENT cohort
    (Ag_Date's month) instead, one row per cohort with NPA%/SMA-2% for that
    cohort alone.

    Reuses analysis/portfolio_intelligence.py::compute_product_analysis's
    own "vintage" sub-table directly -- the SAME cohort computation the
    dashboard's Vintage chart (build_vintage_chart) and report_agent's
    vintage_analysis section already use, so a cohort's NPA%/SMA-2% can
    never drift into a second, independently-computed definition here. A
    cohort with fewer than MIN_ACCOUNTS_SOURCE_VINTAGE (config.py) accounts
    is silently excluded by that same shared function -- not a bug here,
    the same materiality floor the dashboard's own vintage chart already
    applies, so a 2-loan cohort doesn't produce a meaningless 100%/0% spike.

    as_of anchors "is this cohort's agreement date in the future" to the
    report's own reporting month, never wall-clock today -- same convention
    every other as_of-taking function in this codebase already follows
    (compute_product_analysis's own docstring; CLAUDE.md documents this
    exact bug class having been hit twice already). scope_col/scope_value
    optionally narrow to one branch/region/zone/BU first, same convention
    every other step in this module uses.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    if scope_col and scope_value:
        col = resolve_dimension(scope_col)[0]
        if col in df.columns:
            df = df[df[col].astype(str).str.strip().str.upper() == str(scope_value).strip().upper()]
    if df.empty:
        return pd.DataFrame()

    result = compute_product_analysis(df, as_of=as_of)
    return result.get("vintage", pd.DataFrame())


# ── demand_summary ────────────────────────────────────────────────────────────

_DEMAND_DIMENSIONS = {"region", "branch", "executive"}


def demand_summary(
    df: pd.DataFrame,
    dimension: str,
    scope_col: str | None = None,
    scope_value: str | None = None,
) -> pd.DataFrame:
    """Root-cause MECHANISM's third lens: is poor Overall Collection% driven
    by failing to recover OLD arrears (Overdue Collection% -- a legacy/
    recovery problem) or by failing to collect THIS MONTH's own fresh EMI
    demand (Month Demand Collection% -- an ongoing execution problem)? These
    need different fixes, same reason roll_rate_summary/vintage_summary
    exist -- neither of those, nor dimension_breakdown/entity_summary
    (plain Collection%, not decomposed into overdue vs fresh demand), can
    answer this.

    Reuses analysis/portfolio_intelligence.py::compute_overdue_demand_scorecard
    directly -- the SAME waterfall computation the dashboard's Overdue-vs-
    Month-Demand tables and report_agent's overdue_demand section already
    use, so these percentages can never drift into a second, independently
    -computed definition here.

    dimension picks WHICH of that function's three grains to return
    (region/branch/executive -- its own dict keys, not a resolve_dimension
    alias); scope_col/scope_value optionally narrow the input FIRST, same
    convention every other step in this module uses, so "demand breakdown
    by executive within Mahad" computes each executive's rates from ONLY
    Mahad's rows, not diluted by the rest of the portfolio.
    """
    if df is None or df.empty or dimension not in _DEMAND_DIMENSIONS:
        return pd.DataFrame()

    if scope_col and scope_value:
        col = resolve_dimension(scope_col)[0]
        if col in df.columns:
            df = df[df[col].astype(str).str.strip().str.upper() == str(scope_value).strip().upper()]
    if df.empty:
        return pd.DataFrame()

    result = compute_overdue_demand_scorecard(df)
    return result.get(dimension, pd.DataFrame())


# ── new_advances_summary / new_advances_by_dimension ────────────────────────
# Business VOLUME (originations), not collection performance -- a different
# axis entirely from every other step in this module, matching the
# dashboard's own Business tab (ui/tabs/business.py Sections 1 and 3). "How
# much new business did we originate this month/in this branch" and "which
# branch/executive originated the most" are real, distinct questions from
# "how well are we collecting," and were a confirmed, named gap in
# assistant_feature.md's honest critique ("new-advances tracking... still
# isn't reachable from the Investigator") before this was added.

_NEW_ADVANCES_DIMENSIONS = {"region", "branch", "executive"}


def new_advances_summary(
    df: pd.DataFrame,
    scope_col: str | None = None,
    scope_value: str | None = None,
    as_of=None,
) -> pd.DataFrame:
    """One scope's own new-business KPIs this reporting month (accounts,
    funded amount, avg ticket size) vs last month -- reuses
    analysis/portfolio_intelligence.py::compute_new_advances directly, the
    SAME KPI computation the Business tab's own headline cards use, so these
    numbers can never drift into a second, independently-computed
    definition here. Never gated by curr_bucket -- a fresh advance counts
    as new business even if already overdue by upload time, same as that
    tab.

    Reshaped into the same Metric/Current/Previous/Delta table shape
    entity_summary already returns, so this renders through the exact same
    generic table UI and narration path with no new rendering special case.
    as_of anchors "this reporting month" the same way every other as_of-
    taking step in this module does (see vintage_summary's own docstring --
    never wall-clock today). scope_col/scope_value optionally narrow the
    INPUT to one branch/region/zone/BU first, same convention every other
    step here uses; compute_new_advances itself has no scope concept of its
    own, it always summarizes whatever df it's given.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    if scope_col and scope_value:
        col = resolve_dimension(scope_col)[0]
        if col in df.columns:
            df = df[df[col].astype(str).str.strip().str.upper() == str(scope_value).strip().upper()]
    if df.empty:
        return pd.DataFrame()

    data = compute_new_advances(df, as_of=as_of)
    if not data.get("accounts") and not data.get("has_prev"):
        return pd.DataFrame()

    rows = [{
        "Metric": "New Advances (Accounts)", "Current": data["accounts"],
        "Previous": data["prev_accounts"] if data["has_prev"] else None,
        "Delta": (data["accounts"] - data["prev_accounts"]) if data["has_prev"] else None,
    }, {
        "Metric": "Funded Amount (Cr)", "Current": round(data["funded_cr"], 2),
        "Previous": round(data["prev_funded_cr"], 2) if data["has_prev"] else None,
        "Delta": round(data["funded_cr"] - data["prev_funded_cr"], 2) if data["has_prev"] else None,
    }, {
        "Metric": "Avg Ticket Size (L)", "Current": round(data["avg_ticket_l"], 2),
        "Previous": None, "Delta": None,
    }]
    return pd.DataFrame(rows)


def new_advances_by_dimension(
    df: pd.DataFrame,
    dimension: str,
    scope_col: str | None = None,
    scope_value: str | None = None,
    as_of=None,
) -> pd.DataFrame:
    """WHICH region/branch/executive originated the most new business this
    month -- the ranked-breakdown counterpart to new_advances_summary above
    (which only ever answers ONE scope's own totals), the same relationship
    dimension_breakdown has to entity_summary elsewhere in this module.
    Reuses analysis/portfolio_intelligence.py::compute_new_advances_by_dimension
    directly -- the SAME table the Business tab's own Section 3 renders, so
    these numbers can never drift into a second, independently-computed
    definition here. Already sorted Accounts-This-Month-descending by that
    shared function. scope_col/scope_value optionally narrow the INPUT
    first (e.g. "new advances by executive within Mahad").
    """
    if df is None or df.empty or dimension not in _NEW_ADVANCES_DIMENSIONS:
        return pd.DataFrame()

    if scope_col and scope_value:
        col = resolve_dimension(scope_col)[0]
        if col in df.columns:
            df = df[df[col].astype(str).str.strip().str.upper() == str(scope_value).strip().upper()]
    if df.empty:
        return pd.DataFrame()

    result = compute_new_advances_by_dimension(df, as_of=as_of)
    return result.get(dimension, pd.DataFrame())


# ── product_analysis / top_accounts / new_advances_trend ────────────────────
# Tier-1 gap-closing: each of these wraps a function ALREADY imported/used
# elsewhere in this module (compute_product_analysis by vintage_summary,
# compute_new_advances* by new_advances_summary/new_advances_by_dimension)
# under a different dict key, or a function already fully self-contained and
# tested (compute_top_accounts) -- zero new business logic, matching this
# module's reuse-first discipline.

_PRODUCT_AXES = {"segment", "fuel", "source"}


def product_analysis(
    df: pd.DataFrame,
    axis: str,
    scope_col: str | None = None,
    scope_value: str | None = None,
    as_of=None,
) -> pd.DataFrame:
    """Segment-wise NPA breakdown -- "which segment/fuel-type/source has the
    worst NPA%," a different axis from vintage_summary (origination COHORT,
    not product/channel attribute) even though both are sub-tables of the
    SAME compute_product_analysis call (vintage_summary already reuses this
    exact function for its own "vintage" key; this just exposes the other
    three keys that function already computes and returns but nothing in
    this module read before now). axis picks which of "segment"/"fuel"/
    "source" to return -- SegmentName/Segment, FUEL_TYPE, and SRC Name
    respectively (whichever of those columns exists in this file; a file
    missing one silently has no entry for that axis, not an error, same as
    compute_product_analysis's own per-axis column-presence check). A group
    below MIN_ACCOUNTS_PRODUCT_SEGMENT is silently excluded by that same
    shared function, same materiality floor the dashboard's own product
    tables already apply. scope_col/scope_value optionally narrow to one
    branch/region/zone/BU first, same convention every other step here
    uses. as_of anchors "vintage" cohort exclusion the same way every other
    as_of-taking step in this module does, even though it's irrelevant to
    the segment/fuel/source axes themselves -- passed straight through
    since compute_product_analysis takes one as_of for the whole call.
    """
    if df is None or df.empty or axis not in _PRODUCT_AXES:
        return pd.DataFrame()

    if scope_col and scope_value:
        col = resolve_dimension(scope_col)[0]
        if col in df.columns:
            df = df[df[col].astype(str).str.strip().str.upper() == str(scope_value).strip().upper()]
    if df.empty:
        return pd.DataFrame()

    result = compute_product_analysis(df, as_of=as_of)
    return result.get(axis, pd.DataFrame())


def top_accounts(
    df: pd.DataFrame,
    n: int = 20,
    scope_col: str | None = None,
    scope_value: str | None = None,
) -> pd.DataFrame:
    """Largest single exposures among DELINQUENT accounts (any non-STD
    bucket) by SOH -- distinct from top_closing_arrears (raw Closing
    Arrears value, ANY status, not restricted to delinquent) and from
    high_arrears_at_risk (a RATIO -- Inst+Exp+BC arrears vs loan amount,
    "potential write-off," not a plain SOH ranking). Reuses
    analysis/portfolio_intelligence.py::compute_top_accounts directly, the
    SAME function the older AI Query tool's registered view and the report's
    top_accounts section both already use, so this can never drift into a
    second, independently-computed definition. Only the ranked table is
    returned here (that function's own summary dict -- total SOH, % of
    portfolio, NPA count among the top N -- is dashboard/report-narrative
    scaffolding, not something a downloadable, auditable table needs to
    carry). scope_col/scope_value optionally narrow the INPUT first, same
    convention every other step here uses.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    if scope_col and scope_value:
        col = resolve_dimension(scope_col)[0]
        if col in df.columns:
            df = df[df[col].astype(str).str.strip().str.upper() == str(scope_value).strip().upper()]
    if df.empty:
        return pd.DataFrame()

    top_df, _summary = compute_top_accounts(df, n=n)
    return top_df


def new_advances_trend(
    df: pd.DataFrame,
    months: int | None = None,
    granularity: str = "Monthly",
    scope_col: str | None = None,
    scope_value: str | None = None,
    as_of=None,
) -> pd.DataFrame:
    """Monthly (or rolled-up) new-advances trend -- "show me the last 6
    months of new business," "quarterly new advances trend" -- a DIFFERENT
    axis from every collection-metric step in this module, which only ever
    compare current-vs-previous UPLOADED file (this codebase never holds
    more than two months' files in memory at once). This is NOT subject to
    that same 2-period ceiling: it's derived entirely from THIS SINGLE
    upload's own Ag_Date history (compute_new_advances_trend's own
    docstring -- at least 2 years of trend from one file, no df_prev
    needed), the same reason new_advances_summary/new_advances_by_dimension
    above don't need a previous file either. Reuses
    analysis/portfolio_intelligence.py::compute_new_advances_trend +
    roll_new_advances_trend directly -- the SAME two functions the Business
    tab's own Section 2 chart/table render from. months=None means "all
    history" (that function's own convention); granularity is "Monthly"
    (default), "Quarterly", "Half-Yearly", "Yearly", or "Financial Year" --
    same 5 options ui/tabs/business.py's own granularity radio offers, and
    an unrecognized value is a no-op passthrough in roll_new_advances_trend
    (falls through to the Monthly rows unchanged, never an error). as_of
    anchors "this reporting month" as the trend's upper bound the same way
    every other as_of-taking step in this module does. scope_col/
    scope_value optionally narrow the INPUT first.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    if scope_col and scope_value:
        col = resolve_dimension(scope_col)[0]
        if col in df.columns:
            df = df[df[col].astype(str).str.strip().str.upper() == str(scope_value).strip().upper()]
    if df.empty:
        return pd.DataFrame()

    trend_df = compute_new_advances_trend(df, as_of=as_of, months=months)
    return roll_new_advances_trend(trend_df, granularity)


# ── fleet_exposure / repossession_list ───────────────────────────────────────
# Tier-2 gap-closing: still pure reuse of already-tested analysis functions,
# no new business logic, but each needs a small reshape (fleet_exposure pulls
# one dict key; repossession_list picks a sort order the shared function
# deliberately leaves to its caller).


def fleet_exposure(
    df: pd.DataFrame,
    scope_col: str | None = None,
    scope_value: str | None = None,
) -> pd.DataFrame:
    """"Which customers hold multiple loans" (relationship/cross-sell,
    portfolio concentration) -- distinct from fleet_defaulters (loan-level,
    delinquency-filtered ONLY, hand-rolled separately because that use case
    genuinely needs raw loan rows scoped to defaulters, not an aggregate).
    This one reuses analysis/portfolio_intelligence.py::compute_fleet_exposure's
    own customer-level "top_df" aggregate directly -- ALL fleet operators
    (>= FLEET_MIN_LOANS distinct loans) regardless of delinquency status,
    ranked by Total SOH -- the SAME table the report's fleet_exposure
    section already renders, so these numbers can never drift into a
    second, independently-computed definition. Blank/missing mobile numbers
    are excluded before grouping by that same shared function (see its own
    docstring -- a real, confirmed production bug otherwise: unrelated
    customers with no mobile on file collapsing into one phantom "fleet
    operator"). scope_col/scope_value optionally narrow the INPUT first,
    same convention every other step here uses.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    if scope_col and scope_value:
        col = resolve_dimension(scope_col)[0]
        if col in df.columns:
            df = df[df[col].astype(str).str.strip().str.upper() == str(scope_value).strip().upper()]
    if df.empty:
        return pd.DataFrame()

    result = compute_fleet_exposure(df)
    return result.get("top_df", pd.DataFrame())


def repossession_list(
    df: pd.DataFrame,
    scope_col: str | None = None,
    scope_value: str | None = None,
    as_of=None,
) -> pd.DataFrame:
    """Accounts eligible for repossession -- SMA-2/NPA (deep delinquent)
    AND still within the collateral-value window (Ag_Date within the last
    REPOSSESSION_WINDOW_MONTHS, config.py -- an older loan's collateral has
    typically depreciated too far to be worth repossessing). Reuses
    analysis/portfolio_intelligence.py::compute_repossession_list directly,
    the SAME function the report's own repossession section uses, so this
    can never drift into a second, independently-computed definition. That
    function deliberately returns unsorted rows and leaves ordering to its
    caller (the report renders 3 different sorted views from the same
    list) -- this sorts worst-first by SOH, the same "biggest exposure
    first" convention every other loan-level step in this module uses
    (top_closing_arrears, fleet_defaulters, worst_loans_by_metric).
    scope_col/scope_value optionally narrow the INPUT first. as_of anchors
    "within the collateral window" to the report's own reporting month,
    same convention every other as_of-taking step in this module uses --
    never wall-clock today.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    if scope_col and scope_value:
        col = resolve_dimension(scope_col)[0]
        if col in df.columns:
            df = df[df[col].astype(str).str.strip().str.upper() == str(scope_value).strip().upper()]
    if df.empty:
        return pd.DataFrame()

    result = compute_repossession_list(df, as_of=as_of)
    if result.empty or "SOH" not in result.columns:
        return result
    return result.sort_values("SOH", ascending=False).reset_index(drop=True)


def good_customers(
    df: pd.DataFrame,
    scope_col: str | None = None,
    scope_value: str | None = None,
) -> pd.DataFrame:
    """Loyal / high-quality customers eligible for refinance or relationship
    management -- a business-development question, distinct from every
    other step in this module (all of which answer collection risk or
    origination volume, never "who should we proactively re-approach").
    Reuses analysis/portfolio_intelligence.py::compute_good_customers
    directly -- the SAME criteria (tenure completed >= GOOD_CUSTOMER_MIN_TENURE_PCT,
    LCC% >= GOOD_CUSTOMER_MIN_LCC_PCT, config.py) and the SAME sort (lowest
    SOH first -- the easiest refinance targets) the report's own
    good_customers section already uses, so this can never drift into a
    second, independently-computed definition. scope_col/scope_value
    optionally narrow the INPUT first, same convention every other step
    here uses.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    if scope_col and scope_value:
        col = resolve_dimension(scope_col)[0]
        if col in df.columns:
            df = df[df[col].astype(str).str.strip().str.upper() == str(scope_value).strip().upper()]
    if df.empty:
        return pd.DataFrame()

    return compute_good_customers(df)


# ── entity_summary ────────────────────────────────────────────────────────────

def entity_summary(
    entity_type: str,
    entity_value,
    df_curr: pd.DataFrame,
    df_prev: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """One row per key metric for a SINGLE branch or executive: Current,
    Previous, Delta, Higher Is Better. Delta is a raw percentage-point
    difference for %% metrics (matching compute_region_scorecard's own
    convention, analysis/portfolio_intelligence.py:401 -- never a relative
    %-of-%% figure), and a relative %% change (utils._mom_pct) for absolute
    metrics like SOH. None when no prior period is available for this entity.
    """
    if entity_type == "branch":
        curr_row = _branch_row(df_curr, entity_value)
        prev_row = _branch_row(df_prev, entity_value)
        metrics = _BRANCH_METRICS
    elif entity_type == "executive":
        # Never blindly unpack as name, unit = entity_value -- a live
        # Gemini call produced a bare string, a 1-element list, and a
        # 3-element list for this exact case (a name-only follow-up like
        # "why does pankaj have so low strike rate"), each of which crashed
        # with a raw Python unpacking error instead of degrading
        # gracefully. Extra elements are ignored; a missing unit is None.
        if isinstance(entity_value, (list, tuple)):
            name = entity_value[0] if len(entity_value) > 0 else None
            unit = entity_value[1] if len(entity_value) > 1 else None
        else:
            name, unit = entity_value, None
        curr_row = _executive_row(df_curr, name, unit)
        prev_row = _executive_row(df_prev, name, unit)
        metrics = _EXEC_METRICS
    elif entity_type in _GENERIC_ENTITY_TYPES:
        curr_row = _generic_dimension_row(df_curr, entity_type, entity_value)
        prev_row = _generic_dimension_row(df_prev, entity_type, entity_value)
        metrics = _GENERIC_DIMENSION_METRICS
    else:
        raise ValueError(f"Unknown entity_type '{entity_type}'")

    rows = []
    for metric in metrics:
        curr_val = curr_row.get(metric) if curr_row else None
        prev_val = prev_row.get(metric) if prev_row else None
        rows.append({
            "Metric":            metric,
            "Current":           curr_val,
            "Previous":          prev_val,
            "Delta":             _entity_delta(metric, curr_val, prev_val),
            "Higher Is Better":  METRIC_DIRECTION.get(metric),
        })
    return pd.DataFrame(rows)


def _entity_delta(metric: str, curr_val, prev_val):
    if curr_val is None or prev_val is None:
        return None
    if metric in _MONEY_METRICS:
        return _mom_pct(curr_val, prev_val)
    return round(curr_val - prev_val, 2)


def _branch_row(df: pd.DataFrame | None, branch) -> dict | None:
    if df is None or df.empty or "Unit" not in df.columns:
        return None
    sub = df[df["Unit"].astype(str).str.strip().str.upper() == str(branch).strip().upper()]
    if sub.empty:
        return None
    agg = _branch_aggregates(sub)
    if agg.empty:
        return None
    return agg.iloc[0].to_dict()


def _generic_dimension_row(df: pd.DataFrame | None, entity_type: str, value) -> dict | None:
    """The region/zone/BU counterpart to _branch_row -- one entity's row
    from _dimension_metrics (the same aggregator dimension_breakdown uses),
    filtered to a single value. This is the fix for a real bug: a user
    asked "why is CS Nagar not performing" (CS Nagar is a RegionName, not a
    Unit/branch) right after a region-level dimension_breakdown, and
    entity_summary only supporting entity_type "branch"/"executive" meant
    it silently filtered Unit=="CS Nagar", found zero rows, and returned an
    all-None table with no error at all."""
    col = resolve_dimension(entity_type)[0]
    if df is None or df.empty or col not in df.columns:
        return None
    sub = df[df[col].astype(str).str.strip().str.upper() == str(value).strip().upper()]
    if sub.empty:
        return None
    agg = _dimension_metrics(sub, col)
    if agg.empty:
        return None
    return agg.iloc[0].to_dict()


def _scope_executive_rows(df: pd.DataFrame | None, name, unit=None) -> pd.DataFrame:
    """Loan-level rows matching an executive name by `contains`
    (case-insensitive substring), same convention agents/logical_planner.py's
    prompt already mandates for person-name filters -- these are free-typed
    fields, not a controlled vocabulary, so a caller giving just "Pankaj" for
    "PANKAJ DANDGE" must still resolve. unit narrows the match when given AND
    it actually matches something; a missing or non-matching unit (e.g. a
    guessed-wrong branch, or None) falls back to name-only rather than
    returning nothing. Shared by _executive_row (turns this into one
    scorecard row) and distinct_executive_matches (counts how many DISTINCT
    people are still in this same scoped set) -- one algorithm for "what
    counts as a match," never two that could silently drift apart."""
    if df is None or df.empty or "MNT NAME" not in df.columns or not name:
        return pd.DataFrame()
    name_mask = df["MNT NAME"].astype(str).str.strip().str.upper().str.contains(
        str(name).strip().upper(), na=False, regex=False,
    )
    sub = df[name_mask]
    if unit:
        unit_col = df["Unit"] if "Unit" in df.columns else pd.Series([""] * len(df), index=df.index)
        unit_mask = unit_col.astype(str).str.strip().str.upper() == str(unit).strip().upper()
        scoped = df[name_mask & unit_mask]
        if not scoped.empty:
            sub = scoped
    return sub


def _executive_row(df: pd.DataFrame | None, name, unit=None) -> dict | None:
    """One executive's scorecard row for a name (+ optional unit) resolved
    via _scope_executive_rows. Falls back to compute_executive_scorecard's
    own best-Collection%-first row when more than one distinct person still
    matches, purely as a last-resort guard against a crash -- this should
    not normally happen, since investigator/guardrails.py's
    validate_and_correct is expected to have already caught an ambiguous
    name (via distinct_executive_matches below) and asked for clarification
    BEFORE this ever runs. See guardrails.py's own docstring for the real
    bug this two-layer setup exists to prevent: a struggling "Rahul" being
    silently swapped for a better-performing "Rahul" with no signal at all."""
    sub = _scope_executive_rows(df, name, unit)
    if sub.empty:
        return None
    sc = compute_executive_scorecard(sub, min_accounts=1)
    if sc.empty:
        return None
    return sc.iloc[0].to_dict()


def distinct_executive_matches(df: pd.DataFrame | None, name, unit=None) -> list[tuple[str, str]]:
    """Distinct (MNT NAME, Unit) pairs that _executive_row would consider a
    match for this name/unit, via the SAME _scope_executive_rows narrowing
    _executive_row itself uses. investigator/guardrails.py checks
    len(...) > 1 on this BEFORE entity_summary ever runs, to ask which
    person was meant instead of letting _executive_row's scorecard-sort-order
    tiebreak silently pick one."""
    sub = _scope_executive_rows(df, name, unit)
    if sub.empty:
        return []
    unit_col = (
        sub["Unit"].astype(str).str.strip() if "Unit" in sub.columns
        else pd.Series([""] * len(sub), index=sub.index)
    )
    return sorted(set(zip(sub["MNT NAME"].astype(str).str.strip(), unit_col)))


def resolve_executive_identity(df: pd.DataFrame | None, name, unit=None) -> tuple[str, str] | None:
    """The canonical (MNT NAME, Unit) pair a typed name resolves to in the
    ACTUAL data -- e.g. "Taushif Khan" -> ("TAUSIF KHAN NAZIR KH", "BHSWL").

    entity_summary's executive path already does this resolution
    internally (_executive_row's contains-match), but used to throw it
    away -- only the Metric/Value table reached the caller, never the
    canonical identity. Confirmed live-session consequence: EntityMemory
    kept storing the raw TYPED name, which broke any later step needing an
    EXACT match against real data (non_paying_customers's source-table
    merge requires exact MNT NAME/Unit, and a typed "Taushif Khan" never
    exactly equals "TAUSIF KHAN NAZIR KH") -- silently returning empty in
    one observed case, and pulling a stale, unrelated executive's data in
    another (the LLM substituting whatever OTHER executive-shaped table
    happened to be in memory once the typed name failed to resolve
    anywhere useful). ui/tabs/investigator.py calls this right after
    entity_summary to CANONICALIZE what gets stored in EntityMemory, so
    every later reference to "this executive" in the conversation -- by
    any step, however typed -- resolves against the same real identity.

    Returns None (never a guess) when the name matches zero or more than
    one distinct person -- guardrails.py's ambiguous-name check is
    expected to have already asked for clarification before this is ever
    called on an ambiguous name; this is a second, independent guard
    against the same failure mode, not a replacement for it."""
    matches = distinct_executive_matches(df, name, unit)
    if len(matches) != 1:
        return None
    return matches[0]


# ── underperformer_quartile ───────────────────────────────────────────────────

def underperformer_quartile(
    df: pd.DataFrame,
    metric: str,
    higher_is_better: bool | None = None,
    scope_col: str | None = None,
    scope_value: str | None = None,
) -> pd.DataFrame:
    """Executives in the worst quartile of `metric`, portfolio-wide or
    narrowed to one branch/region/zone/BU. scope_col is a dimension alias
    resolved via registry/semantic_model.py::resolve_dimension (the SAME
    org-hierarchy vocabulary graph.py's compiler uses -- BU > Zone > Region >
    Branch) so a BU or Zone head can scope this, not only a region_filter.
    Reuses compute_executive_scorecard + rank_by_metric for the scorecard/
    tier computation -- this function only decides which tier ("top" or
    "bottom" of the raw sorted values) counts as "bad" for the given
    metric's direction, since rank_by_metric's own tiering has no concept of
    which direction is good.
    """
    hib = _higher_is_better(metric, higher_is_better)
    sub = df
    if scope_col and scope_value:
        col = resolve_dimension(scope_col)[0]
        if col in df.columns:
            sub = df[df[col].astype(str).str.strip().str.upper() == str(scope_value).strip().upper()]
    scorecard = compute_executive_scorecard(sub)
    if scorecard.empty or metric not in scorecard.columns:
        return pd.DataFrame()
    ranked = rank_by_metric(scorecard, metric)
    bad_tier = "bottom" if hib else "top"
    flagged = ranked[ranked["Tier"] == bad_tier]
    # rank_by_metric always sorts descending regardless of which direction
    # is "bad" for this metric -- a user asking "why is collection% so
    # less" is focused on badness and expects the WORST offender first, not
    # whatever order the metric's raw descending sort happened to leave the
    # flagged rows in. ascending=hib: worst = lowest value first when higher
    # is better, worst = highest value first otherwise.
    return flagged.sort_values(metric, ascending=hib).reset_index(drop=True)


# ── bottom_n_per_group ────────────────────────────────────────────────────────

def bottom_n_per_group(
    df: pd.DataFrame,
    metric: str,
    n: int,
    group_col: str,
    higher_is_better: bool | None = None,
) -> pd.DataFrame:
    """Worst N executives PER GROUP (e.g. bottom 2 by Collection% per branch)
    -- distinct from underperformer_quartile's portfolio/region-wide cut.
    One vectorized groupby+sort+head pass across ALL groups at once, never a
    Python loop calling this once per group (see this module's own docstring).
    group_col is resolved through the same alias mechanism graph.py already
    uses (registry/semantic_model.py::resolve_dimension) so "branch"->"Unit"
    isn't reinvented here. A group with fewer than n rows returns all of its
    rows; a tie exactly at the n-th cutoff is broken by Accounts descending,
    so the result is deterministic and reproducible on rerun.
    """
    hib = _higher_is_better(metric, higher_is_better)
    key = resolve_dimension(group_col)[0]
    scorecard = compute_executive_scorecard(df)
    if scorecard.empty or metric not in scorecard.columns or key not in scorecard.columns:
        return pd.DataFrame()
    ordered = scorecard.sort_values([metric, "Accounts"], ascending=[hib, False])
    return ordered.groupby(key, sort=False, group_keys=False).head(n).reset_index(drop=True)


# ── executive_roster ─────────────────────────────────────────────────────────

def executive_roster(
    df: pd.DataFrame,
    scope_col: str | None = None,
    scope_value: str | None = None,
    n: int | None = None,
    metric: str | None = None,
    higher_is_better: bool | None = None,
    worst_first: bool = False,
) -> pd.DataFrame:
    """Every executive at (optionally) one branch/region/zone/BU -- a plain
    "list/show me all executives of X" question, distinct from
    underperformer_quartile (only the worst quartile survives) and
    bottom_n_per_group (caps each group at N): with no n/metric given, this
    returns the FULL roster, no rows dropped. Reuses
    compute_executive_scorecard directly, same as underperformer_quartile's
    own scope-filter-then-scorecard pattern, so this is a thin wrapper, not
    a second scorecard implementation.

    n + metric additionally cover "top/bottom N executives [in scope]" --
    the colloquial "top N" (best performers first) is the DEFAULT direction
    here, the opposite default from underperformer_quartile/
    bottom_n_per_group (which are always worst-first, since those exist
    specifically to surface problems). worst_first=True flips it to match
    that same worst-first convention when the question says "bottom"/
    "worst" instead. ascending is derived from METRIC_DIRECTION the same
    way underperformer_quartile does (ascending=hib for worst-first), just
    inverted for the best-first default so a lower-is-better metric like
    NPA% still sorts its lowest (best) values first, not its highest raw
    number. n is silently ignored if metric isn't also given -- ranking
    without knowing what to rank by isn't a real request, and the full
    roster is still a reasonable answer rather than an error."""
    if df is None or df.empty:
        return pd.DataFrame()
    sub = df
    if scope_col and scope_value:
        col = resolve_dimension(scope_col)[0]
        if col in df.columns:
            sub = df[df[col].astype(str).str.strip().str.upper() == str(scope_value).strip().upper()]
    roster = compute_executive_scorecard(sub)
    if n and metric and not roster.empty and metric in roster.columns:
        hib = _higher_is_better(metric, higher_is_better)
        ascending = hib if worst_first else not hib
        roster = roster.sort_values(metric, ascending=ascending).head(int(n)).reset_index(drop=True)
    return roster


# ── intersect_entities ────────────────────────────────────────────────────────

def intersect_entities(
    tables: list[pd.DataFrame],
    key_cols: list[str] | None = None,
    min_appearances: int | None = None,
) -> pd.DataFrame:
    """Entities appearing in at least `min_appearances` (default: ALL) of the
    given tables, keyed on key_cols (default ["MNT NAME", "Unit"]). A plain
    vectorized key-column merge/count -- same idiom as the in-flight
    agents/plan_executor.py::_op_entity_semi_join's key merge, not row-wise
    tuple checks. Returns full rows (all metric columns), sourced from
    whichever input table each qualifying entity first appears in -- an
    entity qualifying via min_appearances doesn't necessarily appear in
    tables[0], so rows are pooled across every table, not read from one only.
    """
    key_cols = key_cols or ["MNT NAME", "Unit"]
    tables = [t for t in (tables or []) if t is not None and not t.empty and all(c in t.columns for c in key_cols)]
    if not tables:
        return pd.DataFrame()
    min_appearances = min_appearances if min_appearances is not None else len(tables)

    combined = pd.concat([t[key_cols].drop_duplicates() for t in tables], ignore_index=True)
    counts = combined.value_counts(subset=key_cols).reset_index(name="_appearances")
    qualifying = counts[counts["_appearances"] >= min_appearances][key_cols]
    if qualifying.empty:
        return pd.DataFrame()
    pool = pd.concat(tables, ignore_index=True).drop_duplicates(subset=key_cols)
    return pool.merge(qualifying, on=key_cols, how="inner").reset_index(drop=True)


# ── non_paying_customers ──────────────────────────────────────────────────────

def non_paying_customers(
    df: pd.DataFrame,
    entities: pd.DataFrame,
    key_cols: list[str] | None = None,
    all_columns: bool = False,
) -> pd.DataFrame:
    """Customers of the given executives (entities: a table with MNT NAME/Unit
    columns, e.g. intersect_entities' output) who are NOT currently paying --
    Arrears / EMI at or above HARD_BUCKET_ARREARS_EMI_MIN, the same threshold
    utils.py::compute_hard_bucket_pct already uses as the single source of
    truth for "in hard bucket" (never a locally reimplemented cutoff).
    Returns the curated audit columns an RBH usually needs (Ag_Date, POS,
    Closing Arrears, LCC%, Arrears / EMI, Last Receipt Date/Amount) unless
    all_columns=True, which returns every source column instead -- a real
    gap this fixes: "give me all columns for these customers" / "show loan
    number too" used to silently re-run with the SAME curated subset, since
    there was no parameter for the LLM to request more even though it
    understood the intent correctly.
    """
    key_cols = key_cols or ["MNT NAME", "Unit"]
    if df is None or df.empty or entities is None or entities.empty:
        return pd.DataFrame()
    if not all(c in df.columns for c in key_cols) or not all(c in entities.columns for c in key_cols):
        return pd.DataFrame()

    scoped = df.merge(entities[key_cols].drop_duplicates(), on=key_cols, how="inner")
    if scoped.empty:
        return pd.DataFrame()

    not_paying = to_num(scoped, "Arrears / EMI") >= HARD_BUCKET_ARREARS_EMI_MIN
    scoped = scoped[not_paying]
    if all_columns:
        return scoped.reset_index(drop=True)
    cols = [c for c in _CUSTOMER_DISPLAY_COLS if c in scoped.columns]
    return scoped[cols].reset_index(drop=True)


# ── customer_loan_book / customer_loan_summary ───────────────────────────────

def customer_loan_book(df: pd.DataFrame, cust_mob_no: str, all_columns: bool = False) -> pd.DataFrame:
    """Every loan for one customer (by Cust Mob No). A blank/missing mobile
    number never matches -- same exclusion rule
    analysis/portfolio_intelligence.py::compute_fleet_exposure already
    applies, for the same reason (a blank key would otherwise group unrelated
    customers together). all_columns=True returns every source column
    instead of the curated audit subset -- see non_paying_customers'
    docstring for the real gap this fixes."""
    if df is None or df.empty or "Cust Mob No" not in df.columns:
        return pd.DataFrame()
    target = str(cust_mob_no).strip()
    if not target:
        return pd.DataFrame()
    sub = df[df["Cust Mob No"].astype(str).str.strip() == target]
    if all_columns:
        return sub.reset_index(drop=True)
    cols = [c for c in _CUSTOMER_DISPLAY_COLS if c in sub.columns]
    return sub[cols].reset_index(drop=True)


def customer_loan_summary(df: pd.DataFrame, cust_mob_no: str) -> dict:
    """Delinquency/exposure rollup across one customer's full loan book."""
    book = customer_loan_book(df, cust_mob_no)
    if book.empty:
        return {"loan_count": 0, "delinquent_count": 0, "total_soh": 0.0}
    delinquent = (
        book["curr_bucket"].isin(["SMA-1", "SMA-2", "NPA"])
        if "curr_bucket" in book.columns else pd.Series(False, index=book.index)
    )
    return {
        "loan_count":       int(len(book)),
        "delinquent_count": int(delinquent.sum()),
        "total_soh":        float(to_num(book, "SOH", fill=0.0).sum()) if "SOH" in book.columns else 0.0,
    }


# ── concept_filter / concept_summary ─────────────────────────────────────────

def concept_filter(
    df: pd.DataFrame,
    concept: str,
    scope_col: str | None = None,
    scope_value: str | None = None,
) -> pd.DataFrame:
    """Rows matching a registered business CONCEPT (registry/ontology.py --
    "non_starter", "easy_settlement", "colending_at_risk", etc, the SAME
    vocabulary graph.py's compiler uses), optionally narrowed to one
    branch/region/zone/BU. Reuses compiler.core._expand_filters (concept ->
    conditions) and agents.data_executor._build_mask (conditions -> a
    boolean mask) -- the identical expansion/application the main AI Query
    pipeline runs, so a concept's definition can never drift between the
    two pipelines by being redefined here. Returns every matching loan-level
    row, all columns -- a "which accounts" answer, not an aggregate (see
    concept_summary for the small aggregate this app hands to the LLM).
    """
    if df is None or df.empty:
        return pd.DataFrame()
    if scope_col and scope_value:
        col = resolve_dimension(scope_col)[0]
        if col in df.columns:
            df = df[df[col].astype(str).str.strip().str.upper() == str(scope_value).strip().upper()]
    if df.empty:
        return pd.DataFrame()

    errs: list[str] = []
    conditions = _expand_filters([{"concept": concept}], errs)
    if errs:
        raise ValueError(f"Unknown concept '{concept}'")
    missing = _missing_condition_columns(df, conditions)
    if missing:
        raise ValueError(
            f"Can't answer -- this upload is missing the column(s) '{concept}' needs: "
            f"{', '.join(missing)}. Not the same as a real zero-match answer."
        )
    mask = _build_mask(df, conditions)
    return df[mask].reset_index(drop=True)


def concept_summary(
    df: pd.DataFrame,
    concept: str,
    scope_col: str | None = None,
    scope_value: str | None = None,
) -> dict:
    """Count + total SOH for a concept filter -- the SMALL aggregate handed
    to narrate_step for a "how many/are there any" question, never the raw
    filtered rows (which can carry customer names/mobile numbers with no
    reason to expose them to the LLM just to answer a count)."""
    matched = concept_filter(df, concept, scope_col, scope_value)
    if matched.empty:
        return {"count": 0, "total_soh": 0.0}
    return {
        "count": int(len(matched)),
        "total_soh": float(to_num(matched, "SOH", fill=0.0).sum()) if "SOH" in matched.columns else 0.0,
    }


def concept_breakdown(df: pd.DataFrame, dimension: str) -> pd.DataFrame:
    """Which branch/region/zone/executive has the most rows in an ALREADY
    CONCEPT-FILTERED loan-level table (concept_filter's own output) --
    "which branch has the most non-starters" -- the drill concept_filter
    otherwise has none of. Distinct from dimension_breakdown, which
    recomputes each entity's OVERALL Collection%/NPA%/etc from the FULL
    portfolio -- a different question from "how many of THIS filtered
    subset's rows belong to it."

    Groups by EVERY column resolve_dimension returns for `dimension`, not
    just the first -- "executive" resolves to ["MNT NAME", "Unit"]
    together, so two different people sharing a first name in different
    branches are never merged into one row. That collision risk is the
    same class investigator/guardrails.py's ambiguous-executive-name check
    exists to prevent; here it's avoided by construction, since this groups
    real rows directly rather than fuzzy-matching a typed name.

    Sorted by Accounts descending -- worst offender first, the natural read
    for a concept breakdown."""
    if df is None or df.empty:
        return pd.DataFrame()
    cols = [c for c in resolve_dimension(dimension) if c in df.columns]
    if not cols:
        return pd.DataFrame()
    d = df.copy()
    d["_soh"] = to_num(d, "SOH", fill=0.0) if "SOH" in d.columns else 0.0
    grouped = d.groupby(cols, dropna=False).agg(
        Accounts=("_soh", "size"), _soh_sum=("_soh", "sum"),
    ).reset_index()
    grouped["SOH (Cr)"] = (grouped["_soh_sum"] / 1_00_00_000).round(2)
    grouped = grouped.drop(columns=["_soh_sum"])
    return grouped.sort_values("Accounts", ascending=False).reset_index(drop=True)


# ── worst_loans_by_metric ─────────────────────────────────────────────────────

# Which metric is in focus -> the registered CONCEPT it maps to (reusing
# concept_filter's own compiler.core._expand_filters/
# agents.data_executor._build_mask machinery directly -- never a second,
# independently-drifting filter definition of "what counts as NPA" etc).
# Strike Rate % has no registered CONCEPT (Strike is a plain Y/N column,
# never expressed as a business-rule concept elsewhere in the registry) --
# a raw {column, op, value} condition is used instead, which
# compiler.core._expand_filters already supports natively alongside concept
# refs (see its own docstring: "concept refs expand to their full condition
# list... raw conditions pass through"). Both spellings (with/without a
# space) are registered, matching the convention every other metric-name
# dict in this module already follows, since callers use either depending
# on which grain they're thinking in.
METRIC_DRILL_FILTER: dict[str, dict] = {
    "NPA %":          {"concept": "npa"},
    "NPA%":           {"concept": "npa"},
    "Hard Bucket %":  {"concept": "hard_bucket"},
    "Hard Bucket%":   {"concept": "hard_bucket"},
    "Collection %":   {"concept": "short_collection"},
    "Collection%":    {"concept": "short_collection"},
    "Strike Rate %":  {"column": "Strike", "op": "in", "value": ["N", "NO"]},
    "Strike%":        {"column": "Strike", "op": "in", "value": ["N", "NO"]},
}


def worst_loans_by_metric(
    df: pd.DataFrame,
    metric: str,
    scope_col: str | None = None,
    scope_value=None,
    all_columns: bool = False,
) -> pd.DataFrame:
    """The LAST hop of the region -> branch -> executive -> loan drill
    chain, made METRIC-AWARE -- "why does Rahul have so much NPA%" and "why
    is Rahul's strike rate low" used to drill into the SAME generic
    "not paying" list (non_paying_customers's fixed Arrears/EMI threshold),
    regardless of which metric was actually in focus. This maps the metric
    already in focus (typically an executive-grain result's own
    metric_focus) to its matching registered CONCEPT via
    METRIC_DRILL_FILTER above, filters to it, and sorts by Arrears / EMI
    descending -- the worst offenders surface first, with Cust Name (not
    just Loan No, which is alphanumeric and not memorable) front and
    center via the same curated _CUSTOMER_DISPLAY_COLS every other
    customer-level step here already uses. all_columns=True (same lever
    non_paying_customers/customer_loan_book/priority_accounts already have)
    returns every raw column instead of that curated subset.

    scope_col/scope_value narrow FIRST, same convention every other step
    in this module uses -- but generalized to match on EVERY column
    resolve_dimension returns, not just the first, so scope_col="executive"
    (2 columns: MNT NAME, Unit) keys on the name+branch pair TOGETHER when
    scope_value is given as a matching-length tuple/list, never on name
    alone (the same collision class investigator/guardrails.py's
    ambiguous-executive-name check and concept_breakdown/
    roll_rate_by_dimension already guard against). A bare scalar scope_value
    against a multi-column dimension falls back to matching only the first
    column, same as this module's pre-existing single-column scope helpers.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    filter_spec = METRIC_DRILL_FILTER.get(metric)
    if filter_spec is None:
        return pd.DataFrame()

    if scope_col and scope_value is not None:
        cols = [c for c in resolve_dimension(scope_col) if c in df.columns]
        if cols:
            if len(cols) > 1 and isinstance(scope_value, (list, tuple)) and len(scope_value) == len(cols):
                mask = pd.Series(True, index=df.index)
                for c, v in zip(cols, scope_value):
                    mask &= df[c].astype(str).str.strip().str.upper() == str(v).strip().upper()
                df = df[mask]
            else:
                df = df[df[cols[0]].astype(str).str.strip().str.upper() == str(scope_value).strip().upper()]
    if df.empty:
        return pd.DataFrame()

    errs: list[str] = []
    conditions = _expand_filters([filter_spec], errs)
    if errs:
        return pd.DataFrame()
    missing = _missing_condition_columns(df, conditions)
    if missing:
        raise ValueError(
            f"Can't answer -- this upload is missing the column(s) needed for '{metric}': "
            f"{', '.join(missing)}. Not the same as a real zero-match answer."
        )
    mask = _build_mask(df, conditions)
    result = df[mask]
    if result.empty:
        return pd.DataFrame()

    if "Arrears / EMI" in result.columns:
        result = result.sort_values("Arrears / EMI", ascending=False)
    if all_columns:
        return result.reset_index(drop=True)
    cols = [c for c in _CUSTOMER_DISPLAY_COLS if c in result.columns]
    return (result[cols] if cols else result).reset_index(drop=True)


def summarize_worst_loans(result_df: pd.DataFrame, top_n: int = 3) -> str:
    """Deterministic (NEVER LLM-improvised) top-offender summary -- same
    reasoning as investigator/suggestions.py's mechanism suggestions: a
    factual claim about which specific rows are worst must never carry LLM
    variance. Independent of the caller's own sort order (uses nlargest,
    not head) -- correct regardless of how worst_loans_by_metric's result
    happens to already be sorted. Returns "" (never a misleading summary)
    when either required column is missing."""
    if result_df is None or result_df.empty:
        return ""
    if "Cust Name" not in result_df.columns or "Arrears / EMI" not in result_df.columns:
        return ""
    top = result_df.nlargest(top_n, "Arrears / EMI")
    parts = [f"{row['Cust Name']} ({row['Arrears / EMI']:.1f} arrears/EMI)" for _, row in top.iterrows()]
    return "Loans driving this the most: " + ", ".join(parts) + "."


# ── high_arrears_at_risk ──────────────────────────────────────────────────────

def high_arrears_at_risk(
    df: pd.DataFrame,
    scope_col: str | None = None,
    scope_value: str | None = None,
) -> pd.DataFrame:
    """A named, actively-tracked leader priority (confirmed against a real
    operational priority email, not just the smart-alerts card) with NO
    registered CONCEPT: accounts where Inst+Exp+BC arrears exceed
    HIGH_ARREARS_LOAN_RATIO (config.py) of the original loan amount --
    "highly critical, potential write-off" cases regardless of DPD bucket.

    This can't be expressed as a simple registry CONCEPT (registry/
    ontology.py's declarative {column, op, value} condition schema has no
    way to compare a SUM of three columns against a RATIO of a fourth, and
    no such derived column exists on the raw upload) -- so this reuses
    smart_alerts.py::alert_high_arrears_ratio directly instead, the SAME
    computation the dashboard's alert card and report_agent's risk_flags
    section already use, rather than forcing a second, differently-shaped
    implementation into the compiler's condition vocabulary.

    Returns every column (alert_high_arrears_ratio's own "df_full"),
    already sorted worst-ratio-first by that function itself. scope_col/
    scope_value optionally narrow to one branch/region/zone/BU first, same
    convention every other step in this module uses.
    """
    if df is None or df.empty:
        return pd.DataFrame()
    if scope_col and scope_value:
        col = resolve_dimension(scope_col)[0]
        if col in df.columns:
            df = df[df[col].astype(str).str.strip().str.upper() == str(scope_value).strip().upper()]
    if df.empty:
        return pd.DataFrame()
    result = alert_high_arrears_ratio(df)
    return result.get("df_full", pd.DataFrame())


# ── fleet_defaulters ──────────────────────────────────────────────────────────

def fleet_defaulters(
    df: pd.DataFrame,
    scope_col: str | None = None,
    scope_value: str | None = None,
) -> pd.DataFrame:
    """Fleet operators (customers holding >= FLEET_MIN_LOANS distinct
    loans, config.py) who are CURRENTLY delinquent -- "fleet owners must be
    in our priority as they carry a huge SOH and must pay their EMI."
    compute_fleet_exposure identifies fleet operators but returns an
    AGGREGATE (one row per operator, portfolio-wide), never the delinquent
    LOAN rows themselves narrowed to just the defaulting ones -- this fills
    that gap as a loan-level, downloadable, auditable table instead.

    Reuses FLEET_MIN_LOANS directly (the SAME threshold
    compute_fleet_exposure uses) and the SAME blank-mobile exclusion that
    function's own docstring documents as a real, confirmed production bug
    otherwise: utils.py::clean_mobile normalizes a missing mobile number to
    "" (not NaN), and without excluding those rows first, every loan with
    no mobile on file groups under the single key "" and gets reported as
    one fictitious mega "fleet operator" combining unrelated customers.

    Sorted worst-first by SOH -- the exposure that makes a defaulting fleet
    operator a priority in the first place, same framing
    compute_fleet_exposure's own top_df already uses.
    """
    if df is None or df.empty or "Cust Mob No" not in df.columns or "Loan No" not in df.columns:
        return pd.DataFrame()
    if scope_col and scope_value:
        col = resolve_dimension(scope_col)[0]
        if col in df.columns:
            df = df[df[col].astype(str).str.strip().str.upper() == str(scope_value).strip().upper()]
    if df.empty:
        return pd.DataFrame()

    has_mobile = df["Cust Mob No"].astype(str).str.strip() != ""
    df = df[has_mobile]
    if df.empty:
        return pd.DataFrame()

    cust_loan_counts = df.groupby("Cust Mob No")["Loan No"].nunique()
    fleet_customers = cust_loan_counts[cust_loan_counts >= FLEET_MIN_LOANS].index
    fleet_df = df[df["Cust Mob No"].isin(fleet_customers)]
    if fleet_df.empty:
        return pd.DataFrame()

    delinquent = to_num(fleet_df, "Arrears / EMI") > 0
    result = fleet_df[delinquent]
    if result.empty:
        return pd.DataFrame()

    soh = to_num(result, "SOH") if "SOH" in result.columns else pd.Series(0.0, index=result.index)
    return result.assign(_soh=soh).sort_values("_soh", ascending=False).drop(columns=["_soh"]).reset_index(drop=True)


# ── top_closing_arrears ───────────────────────────────────────────────────────

def top_closing_arrears(
    df: pd.DataFrame,
    top_n: int | None = 20,
    scope_col: str | None = None,
    scope_value: str | None = None,
) -> pd.DataFrame:
    """Top N loans by RAW Closing Arrears value, descending -- matches your
    team's own real operational priority email's "TOP N LOANS WITH HIGH
    CLOSING ARREARS" section exactly (confirmed against that actual email,
    not guessed): a straight top-N by absolute rupee value. Distinct from
    high_arrears_at_risk, a RATIO (Inst+Exp+BC arrears exceeding a % of
    loan amount -- "potential write-off"), not "top N by rupee value" --
    these are two genuinely different business questions that happen to
    both involve "high arrears," not one duplicated as two steps.

    Restricted to accounts that actually HAVE arrears (Closing Arrears > 0)
    -- a "top closing arrears" list showing clean, zero-arrears accounts
    would be meaningless. top_n=None returns every such account, unranked
    by count (used by priority_menu below to get an exact count without
    also imposing a display cap)."""
    if df is None or df.empty or "Closing Arrears" not in df.columns:
        return pd.DataFrame()
    if scope_col and scope_value:
        col = resolve_dimension(scope_col)[0]
        if col in df.columns:
            df = df[df[col].astype(str).str.strip().str.upper() == str(scope_value).strip().upper()]
    if df.empty:
        return pd.DataFrame()

    candidates = df[to_num(df, "Closing Arrears", fill=0) > 0].copy()
    if candidates.empty:
        return pd.DataFrame()
    candidates = candidates.sort_values("Closing Arrears", ascending=False)
    if top_n is not None:
        candidates = candidates.head(top_n)
    return candidates.reset_index(drop=True)


# ── priority_menu ─────────────────────────────────────────────────────────────

# Ordered categories for the "which loans should I focus on" menu -- the
# user's own stated priority order, cross-checked against their real
# operational priority email (AKOLA_email.eml / config/templates/
# priority_focus.yaml in the sibling "Excel Automation" project). Each
# entry's "kind" says which ALREADY-BUILT step answers it -- concept_filter
# for every registered CONCEPT, or the dedicated step for the two that
# aren't expressible as a simple concept. default_n mirrors the user's own
# stated cap for closing arrears (20); other categories default to 10, a
# reasonable middle ground absent a stated preference for those.
PRIORITY_MENU_CATEGORIES: list[dict] = [
    {"label": "Non Starter",                    "kind": "concept", "concept": "non_starter",                  "default_n": 10},
    {"label": "NPA Accounts",                   "kind": "concept", "concept": "npa",                          "default_n": 10},
    {"label": "High Closing Arrears",           "kind": "top_closing_arrears",                                 "default_n": 20},
    {"label": "Fleet Owners with High POS",     "kind": "fleet_defaulters",                                    "default_n": 10},
    {"label": "Co-lending at Risk",             "kind": "concept", "concept": "colending_at_risk",             "default_n": 10},
    {"label": "Insurance-Only Delinquency",     "kind": "concept", "concept": "insurance_driven_delinquency",  "default_n": 10},
    {"label": "Easy Settlement",                "kind": "concept", "concept": "easy_settlement",               "default_n": 10},
    {"label": "Recent Advances - High Bucket",  "kind": "concept", "concept": "recent_advance_high_bucket",    "default_n": 10},
    {"label": "No Collection 3 Months",         "kind": "concept", "concept": "no_collection_3m",              "default_n": 10},
]


def _priority_menu_category_result(df: pd.DataFrame, category: dict) -> pd.DataFrame:
    """The full (uncapped) matching table for ONE menu category -- shared by
    priority_menu (which only needs the COUNT) and the UI's own per-category
    drill (which needs the actual rows, capped at that category's Shown).
    Reuses concept_filter/top_closing_arrears/fleet_defaulters directly --
    never a second, independently-derived filter for any category."""
    kind = category["kind"]
    if kind == "concept":
        try:
            return concept_filter(df, category["concept"])
        except ValueError:
            # A registered concept whose column(s) this particular upload
            # is missing -- skip this ONE category rather than failing the
            # whole menu (same "don't let one gap break everything" reasoning
            # as every other multi-part step in this module).
            return pd.DataFrame()
    if kind == "fleet_defaulters":
        return fleet_defaulters(df)
    if kind == "top_closing_arrears":
        return top_closing_arrears(df, top_n=None)
    return pd.DataFrame()


def priority_menu(
    df: pd.DataFrame,
    scope_col: str | None = None,
    scope_value: str | None = None,
) -> pd.DataFrame:
    """"Ask me which category to focus on first, don't just dump every
    priority loan at once" -- the menu-first alternative to priority_accounts'
    combined dump. Returns one row per NON-ZERO category (Category, Count,
    Shown), in PRIORITY_MENU_CATEGORIES' own order, so a human picks ONE to
    drill into next instead of scrolling a single giant combined table.

    Each category's count is computed INDEPENDENTLY -- deliberately NOT
    deduplicated to "each loan counted under its single highest-priority
    tier only" the way priority_accounts/execute_priority_mode's dedup rule
    works. This matches the REAL operational priority email this mirrors,
    whose own per-category tables are each an independent filter with no
    cross-table dedup -- a loan can genuinely be both a Non Starter AND
    have high Closing Arrears, and a leader reviewing this menu should see
    it counted in both, not silently claimed by only one.

    Shown is capped at each category's own default_n, but never exceeds
    that category's actual Count (min of the two) -- "show top 20" on a
    branch with only 3 matching loans shows 3, not a misleading 20."""
    if df is None or df.empty:
        return pd.DataFrame()
    if scope_col and scope_value:
        col = resolve_dimension(scope_col)[0]
        if col in df.columns:
            df = df[df[col].astype(str).str.strip().str.upper() == str(scope_value).strip().upper()]
    if df.empty:
        return pd.DataFrame()

    rows = []
    for category in PRIORITY_MENU_CATEGORIES:
        count = len(_priority_menu_category_result(df, category))
        if count == 0:
            continue
        rows.append({
            "Category": category["label"],
            "Count": count,
            "Shown": min(category["default_n"], count),
        })
    return pd.DataFrame(rows)


# ── priority_accounts / priority_by_executive ────────────────────────────────

def priority_accounts(
    df: pd.DataFrame,
    scope_col: str | None = None,
    scope_value: str | None = None,
    all_columns: bool = False,
) -> pd.DataFrame:
    """The 7-tier business priority framework ("what should my team work on
    today"), reusing agents/data_executor.py::execute_priority_mode directly
    -- the SAME framework the main AI Query pipeline's priority-mode routing
    already uses, never a reimplementation. Loan-level output (Priority,
    Loan No, Cust Name, Cust Mob No, RegionName, Unit, MNT NAME, ...), the
    SAME shape non_paying_customers already has -- which is exactly what
    lets customer_loan_book's existing mobile-number drill work on this
    completely unchanged.

    all_columns=True (same lever non_paying_customers/customer_loan_book
    already have) swaps execute_priority_mode's own curated ~13-column
    display_cols for every raw column the upload actually has -- a real
    gap this fixes: a user asking for "all columns" on THIS step
    specifically had no way to get them, unlike the other two loan-level
    steps. Recomputed from the ORIGINAL df (not execute_priority_mode's
    already-curated output), then the SAME Priority tags it computed are
    mapped back on by Loan No -- never a second, independently-derived
    priority assignment.
    """
    from agents.data_executor import execute_priority_mode

    if df is None or df.empty:
        return pd.DataFrame()
    if scope_col and scope_value:
        col = resolve_dimension(scope_col)[0]
        if col in df.columns:
            df = df[df[col].astype(str).str.strip().str.upper() == str(scope_value).strip().upper()]
    if df.empty:
        return pd.DataFrame()
    result, _message = execute_priority_mode(df)
    if result.empty or not all_columns or "Loan No" not in result.columns or "Loan No" not in df.columns:
        return result

    priority_by_loan = result.set_index("Loan No")["Priority"]
    full = df[df["Loan No"].isin(result["Loan No"])].copy()
    full.insert(0, "Priority", full["Loan No"].map(priority_by_loan))
    return full.reset_index(drop=True)


def priority_by_executive(df_curr: pd.DataFrame, priority_df: pd.DataFrame) -> pd.DataFrame:
    """Which executives have the most priority-flagged accounts, most first
    -- reuses compute_executive_scorecard + rank_by_metric (the SAME
    primitives bottom_n_per_group is itself built on), scoped to just the
    priority-flagged loans, so the resulting "Accounts" column IS each
    executive's priority-case count, not their whole book."""
    if priority_df is None or priority_df.empty or "Loan No" not in priority_df.columns:
        return pd.DataFrame()
    if df_curr is None or df_curr.empty or "Loan No" not in df_curr.columns:
        return pd.DataFrame()
    scoped = df_curr[df_curr["Loan No"].isin(priority_df["Loan No"])]
    scorecard = compute_executive_scorecard(scoped, min_accounts=1)
    if scorecard.empty:
        return pd.DataFrame()
    return rank_by_metric(scorecard, "Accounts")


# ── dispatch registry ────────────────────────────────────────────────────────

STEP_REGISTRY = {
    "priority_accounts":       priority_accounts,
    "priority_menu":           priority_menu,
    "top_closing_arrears":     top_closing_arrears,
    "priority_by_executive":   priority_by_executive,
    "concept_filter":          concept_filter,
    "concept_breakdown":       concept_breakdown,
    "high_arrears_at_risk":    high_arrears_at_risk,
    "fleet_defaulters":        fleet_defaulters,
    "worst_loans_by_metric":   worst_loans_by_metric,
    "dimension_breakdown":     dimension_breakdown,
    "roll_rate_summary":       roll_rate_summary,
    "roll_rate_by_dimension":  roll_rate_by_dimension,
    "vintage_summary":         vintage_summary,
    "demand_summary":          demand_summary,
    "new_advances_summary":    new_advances_summary,
    "new_advances_by_dimension": new_advances_by_dimension,
    "new_advances_trend":      new_advances_trend,
    "product_analysis":        product_analysis,
    "top_accounts":            top_accounts,
    "fleet_exposure":          fleet_exposure,
    "repossession_list":       repossession_list,
    "good_customers":          good_customers,
    "entity_summary":          entity_summary,
    "underperformer_quartile": underperformer_quartile,
    "bottom_n_per_group":      bottom_n_per_group,
    "executive_roster":        executive_roster,
    "intersect_entities":      intersect_entities,
    "non_paying_customers":    non_paying_customers,
    "customer_loan_book":      customer_loan_book,
}
