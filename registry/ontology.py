"""Ontology  -  the VOCABULARY layer of the registry.

Three things live here:

- PRIORITY_RULES  -  the 7-tier business priority framework. MIGRATED VERBATIM from
  agents.domain_expert (which now re-exports it for back-compat). This is the
  canonical home; agents.data_executor.execute_priority_mode and the prompt's
  generated priority section both read it. Kept byte-identical so behavior is
  unchanged in Phase 0.

- CONCEPTS  -  named, deterministic business rules the v2 compiler will expand. Each
  concept is referenced by name in the logical IR; the compiler expands it to its
  full condition set, so a multi-condition concept (e.g. colending_at_risk) can
  NEVER silently lose a condition. Schema, per concept:
      {
        "label":       human-readable name,
        "description": one-line meaning (drives the generated prompt + clarify),
        "conditions":  [ {"column", "op", "value"}, ... ]   # ANDed together
      }
  conditions use the SAME shape that agents.data_executor._apply_condition /
  _build_mask already execute, so the compiler lowers a concept to existing
  engine primitives with no new execution code.

  Column-vs-column comparisons ARE supported (op in col_lt/col_lte/col_gt/col_gte/
  col_eq/col_ne, where "value" is another column NAME -- see _COL_COMPARE_OPS in
  agents/data_executor.py). short_collection uses this (Month Receipt Amount
  col_lte Net Collection Demand Inst+Exp+BC). under_collected would be a trivial
  addition on the same primitive if ever needed -- not added since nobody has
  asked for the "partial payers only, zero-payers excluded" distinction yet.

  "strike" (Strike=Y) is STILL NOT encoded as a CONCEPT, for a DIFFERENT reason
  than before: its real definition is an OR of three legs (Month Collection >=
  Month Due-Inst OR LCC%==100 OR ARREARS AGAINST INST<=0), and this schema's
  "conditions" list is ANDed only -- there is no OR'd condition group primitive.
  Encoding it as an AND of the two easier legs would silently compute a
  narrower, WRONG criterion -- worse than not having the concept at all. Defer
  until the compiler supports OR'd condition groups; until then the "Strike"
  column itself is documented correctly in agents/logical_planner.py's glossary
  and usable directly as a raw column filter (Strike == "Y" / "N").

  __CUTOFF_1Y__ is a dynamic placeholder (loan agreement date within last 12
  months); execute_priority_mode resolves it today, and the v2 compiler resolves
  it at lowering time. Same convention as PRIORITY_RULES.

- METRICS  -  named numeric measures with their default aggregation AND grain. The
  grain is part of the definition (a measure is "sum of SOH AT loan grain"), so
  the compiler can prevent fan-out double-counting when grouping.

Nothing here except PRIORITY_RULES is consumed yet  -  CONCEPTS/METRICS are additive
in Phase 0, so they cannot change behavior.
"""

# Thresholds shared with the dashboard (analysis/, smart_alerts.py) -- imported
# rather than re-hardcoded, so a business-rule change in config.py can't silently
# desync the AI Query pipeline's definition of the same concept from what the
# dashboard tabs already show.
from config import (
    EASY_SETTLEMENT_MAX_ARREARS,
    INSURANCE_EXP_ARREARS_MIN,
    FLEET_MIN_LOANS,
    RECENT_ADVANCES_MONTHS,
    HARD_BUCKET_ARREARS_EMI_MIN,
)

# ── Business Priority Framework (migrated verbatim  -  single source of truth) ───
# Used by the system prompt's generated priority section AND by the data executor
# when priority_mode is active.
PRIORITY_RULES = [
    {
        "rank": 1,
        "label": "Non Starters",
        "why": "Never paid even 1st EMI - highest credit risk, possible fraud or disbursement issue",
        "conditions": [{"column": "Non Starter", "op": "==", "value": "Y"}],
    },
    {
        "rank": 2,
        "label": "Easy Settlements",
        "why": f"Closing arrears < ₹{EASY_SETTLEMENT_MAX_ARREARS:,} - one call can clear these, quick wins for collection team",
        "conditions": [
            {"column": "Closing Arrears", "op": ">",  "value": 0},
            {"column": "Closing Arrears", "op": "<",  "value": EASY_SETTLEMENT_MAX_ARREARS},
        ],
    },
    {
        "rank": 3,
        "label": "Recent Advances - High Bucket",
        "why": f"Loans sanctioned within last {RECENT_ADVANCES_MONTHS} months already in SMA-1 or worse  -  early warning of sourcing quality issues",
        "conditions": [
            {"column": "Ag_Date",       "op": ">=", "value": "__CUTOFF_1Y__"},
            {"column": "Arrears / EMI", "op": ">=", "value": 1},
        ],
    },
    {
        "rank": 4,
        "label": "Insurance-Driven Delinquency",
        "why": "Customer paid EMI (no arrears against installment) but unpaid insurance/expense charge is creating artificial arrears - fixable via cash or child loan",
        "conditions": [
            {"column": "ARREARS AGAINST INST", "op": "<=", "value": 0},
            {"column": "ARREARS AGAINST EXP",  "op": ">",  "value": INSURANCE_EXP_ARREARS_MIN},
            {"column": "Arrears / EMI",         "op": ">",  "value": 0},
        ],
    },
    {
        "rank": 5,
        "label": "Co-lending at Risk",
        "why": "Partner bank co-lending loans with any delinquency - SLA breach risk",
        "conditions": [
            {"column": "CoLending_Loans", "op": "==", "value": "Y"},
            {"column": "Arrears / EMI",   "op": ">",  "value": 0},
        ],
    },
    {
        "rank": 6,
        "label": "No Collection 3 Months",
        "why": "No payment for 3+ months AND >6 EMI arrears - pre-NPA deterioration signal",
        "conditions": [{"column": "No Coll 3 Months and >6 EMI", "op": "==", "value": "Y"}],
    },
    {
        "rank": 7,
        "label": "NPA Accounts",
        "why": "Fully non-performing - requires legal/recovery escalation",
        "conditions": [{"column": "curr_bucket", "op": "==", "value": "NPA"}],
    },
]


# ── Named business concepts (referenced by the logical IR, expanded by compiler) ─
CONCEPTS: dict[str, dict] = {
    "delinquent": {
        "label": "Delinquent",
        "description": "Any account with arrears - Arrears/EMI > 0.",
        "conditions": [{"column": "Arrears / EMI", "op": ">", "value": 0}],
    },
    "non_starter": {
        "label": "Non Starter",
        "description": "Customer has not paid even the 1st EMI - highest credit risk.",
        "conditions": [{"column": "Non Starter", "op": "==", "value": "Y"}],
    },
    "npa": {
        "label": "NPA",
        "description": "Non-performing asset - current bucket is NPA.",
        "conditions": [{"column": "curr_bucket", "op": "==", "value": "NPA"}],
    },
    "easy_settlement": {
        "label": "Easy Settlement",
        "description": f"Closing arrears between 0 and ₹{EASY_SETTLEMENT_MAX_ARREARS:,} - one call can clear these.",
        "conditions": [
            {"column": "Closing Arrears", "op": ">", "value": 0},
            {"column": "Closing Arrears", "op": "<", "value": EASY_SETTLEMENT_MAX_ARREARS},
        ],
    },
    "colending_at_risk": {
        "label": "Co-lending at Risk",
        "description": "Partner-bank co-lending loan with any delinquency - SLA breach risk.",
        "conditions": [
            {"column": "CoLending_Loans", "op": "==", "value": "Y"},
            {"column": "Arrears / EMI", "op": ">", "value": 0},
        ],
    },
    "insurance_driven_delinquency": {
        "label": "Insurance-Driven Delinquency",
        "description": f"EMI paid but unpaid insurance/expense charge (> ₹{INSURANCE_EXP_ARREARS_MIN:,}) creates artificial arrears.",
        "conditions": [
            {"column": "ARREARS AGAINST INST", "op": "<=", "value": 0},
            {"column": "ARREARS AGAINST EXP", "op": ">", "value": INSURANCE_EXP_ARREARS_MIN},
            {"column": "Arrears / EMI", "op": ">", "value": 0},
        ],
    },
    "recent_advance_high_bucket": {
        "label": "Recent Advance - High Bucket",
        "description": f"Loan sanctioned within last {RECENT_ADVANCES_MONTHS} months already in SMA-1 or worse.",
        "conditions": [
            {"column": "Ag_Date", "op": ">=", "value": "__CUTOFF_1Y__"},
            {"column": "Arrears / EMI", "op": ">=", "value": 1},
        ],
    },
    "no_collection_3m": {
        "label": "No Collection 3 Months",
        "description": "No payment for 3+ months AND arrears exceed 6 EMIs - pre-NPA signal.",
        "conditions": [{"column": "No Coll 3 Months and >6 EMI", "op": "==", "value": "Y"}],
    },
    "no_collection": {
        "label": "No Collection",
        "description": "No cash received this month - Month Receipt Amount <= 0.",
        "conditions": [
            {"column": "Month Receipt Amount", "op": "<=", "value": 0},
        ],
    },
    "short_collection": {
        "label": "Short Collection",
        "description": (
            "This month's cash received is at or below what was due - Month Receipt "
            "Amount <= Net Collection Demand Inst+Exp+BC. Includes zero-payers (also "
            "covered by no_collection) as well as partial payers."
        ),
        "conditions": [
            {"column": "Month Receipt Amount", "op": "col_lte", "value": "Net Collection Demand Inst+Exp+BC"},
        ],
    },
}


# ── Entity concepts (reusable per-entity predicates for nested aggregation) ────
# A named entity-level predicate: "keep entities of this kind satisfying this
# per-entity HAVING." Referenced from an IR-2 entity_filter as {"concept": <name>},
# so the LLM declares "fleet operators per region" without authoring the rollup.
# `having` predicates use the same agg/column/distinct/concept + op + value shape
# the compiler evaluates at the (output-dimension, entity) grain.
ENTITY_CONCEPTS: dict[str, dict] = {
    "fleet_operator": {
        "entity": "customer",
        "label": "Fleet Operator",
        "description": f"A customer holding {FLEET_MIN_LOANS} or more loans/vehicles.",
        "having": [{"agg": "nunique", "column": "Loan No", "op": ">=", "value": FLEET_MIN_LOANS}],
    },
}


# ── Metrics (numeric measures) ────────────────────────────────────────────────
# Each metric declares a `kind` (its additivity class) so the compiler lowers it
# correctly without per-metric special-casing:
#   additive        -  safe to sum across every dimension (default; column + default_agg)
#   semi_additive   -  additive across some dimensions but not others (non_additive_over)
#   ratio           -  non-additive; defined as numerator/denominator (+ optional
#                    scale/cap), recomputed at the target grain, NEVER summed/averaged
# Adding a new kind only needs a handler in compiler.measures  -  not a schema change.
METRICS: dict[str, dict] = {
    "exposure": {
        "label": "Exposure (SOH)",
        "kind": "additive",
        "column": "SOH",
        "default_agg": "sum",
        "grain": "loan",
        "description": "Sum of Hire = POS + Closing Arrears = total exposure if customer defaults.",
    },
    "pos": {
        "label": "POS",
        "kind": "additive",
        "column": "POS",
        "default_agg": "sum",
        "grain": "loan",
        "description": "Principal outstanding - future principal balance remaining.",
    },
    "closing_arrears": {
        "label": "Closing Arrears",
        "kind": "additive",
        "column": "Closing Arrears",
        "default_agg": "sum",
        "grain": "loan",
        "description": "Total overdue amount at month close.",
    },
    "closing_penal": {
        "label": "Closing Penal Charges",
        "kind": "additive",
        "column": "ClosingPC",
        "default_agg": "sum",
        "grain": "loan",
        "description": "Accumulated penalty charges on the loan.",
    },
    "lcc_pct": {
        "label": "LCC %",
        "kind": "ratio",
        # Cumulative collection efficiency: cum collection / cum dues, as a % capped
        # at 100. Computed as sum(num)/sum(den)*100 at the target grain  -  averaging
        # per-loan LCC% would be wrong.
        "numerator": ["Cum Coll (Inst+Exp)"],
        "denominator": ["Cum Due-Inst", "Cum Due-Exp"],
        "scale": 100,
        "cap": 100,
        "grain": "loan",
        "description": "Cumulative collection efficiency = cum collection / cum dues, as a percent (capped at 100).",
    },
    "collection_pct": {
        "label": "Collection %",
        "kind": "ratio",
        # This-month collection efficiency: sum(collected) / sum(demand) * 100.
        # Same shape as lcc_pct -- computed at the target grain, never averaged
        # per-loan. Registering this (rather than leaving it for the planner to
        # invent as a bare name) also means time.compare auto-derives a correct
        # prev_collection_pct for free, the same way it already does for lcc_pct.
        "numerator": ["Month Collection (Excluding Reserve Collection)"],
        "denominator": ["Net Collection Demand Inst+Exp+BC"],
        "scale": 100,
        "grain": "loan",
        "description": "This month's collection efficiency = month collection / month demand, as a percent.",
    },
    "overdue_collection_pct": {
        "label": "Overdue Collection %",
        "kind": "ratio",
        # Overdue-first collection waterfall: a payment clears last month's
        # carried-over overdue (Arrear Opening) BEFORE any of it counts against
        # this month's own EMI demand. "Overdue"/"OverdueCollected" are derived
        # columns utils.py::assign_buckets computes once per loan (same pattern
        # as SOH) -- utils.compute_overdue_demand_pct is the dashboard/report's
        # shared implementation of this same waterfall.
        #
        # Known, deliberate divergence from utils.compute_overdue_demand_pct:
        # that function returns 100% (not 0/0) when a group's total Overdue is
        # zero -- nothing was outstanding, treated as fully clear. This generic
        # "ratio" measure kind has no concept of that override; a query whose
        # group has zero total Overdue gets a plain 0/0 division here (NaN),
        # not 100. Giving the compiler a conditional-on-zero-denominator ratio
        # kind for one metric would be speculative infrastructure for a case
        # that's rare in practice (a branch/region with literally zero overdue
        # across every loan) -- documented here rather than silently accepted.
        "numerator": ["OverdueCollected"],
        "denominator": ["Overdue"],
        "scale": 100,
        "grain": "loan",
        "description": "Overdue-first collection %: amount collected against last month's carried-over overdue / total overdue, as a percent.",
    },
    "month_demand_collection_pct": {
        "label": "Month Demand Collection %",
        "kind": "ratio",
        # The other half of the same waterfall: whatever's left over AFTER
        # clearing Overdue counts against this month's own EMI demand
        # (Inst+Exp+BC -- deliberately excludes MONTH DUE PC, a penalty, not
        # core EMI demand). Same zero-denominator caveat as overdue_collection_pct
        # above applies here too.
        "numerator": ["DemandCollected"],
        "denominator": ["MonthDemandExclPC"],
        "scale": 100,
        "grain": "loan",
        "description": "This month's own EMI demand collection % (after overdue is cleared first) = demand collected / month demand (Inst+Exp+BC), as a percent.",
    },
    "hard_bucket_pct": {
        "label": "Hard Bucket %",
        "kind": "count_ratio",
        # % of accounts >= HARD_BUCKET_ARREARS_EMI_MIN EMIs overdue -- a COUNT ratio
        # (accounts matching a condition / all accounts), not a column-sum ratio
        # like collection_pct, so it needs the count_ratio kind (compiler/measures.py).
        # Must match utils.compute_hard_bucket_pct (the shared helper used everywhere
        # else -- dashboard + Portfolio Intelligence). This declarative definition is
        # consumed by the general compiler rather than calling that function directly,
        # so the two can't share code, but tests/test_metric_consistency.py checks they
        # stay numerically identical.
        "numerator_where": [{"column": "Arrears / EMI", "op": ">=", "value": HARD_BUCKET_ARREARS_EMI_MIN}],
        "denominator_where": [],  # empty = count all rows in the group
        "scale": 100,
        "grain": "loan",
        "description": f"% of accounts >= {HARD_BUCKET_ARREARS_EMI_MIN} EMIs overdue - a narrower, more severe signal than NPA.",
    },
    "strike_pct": {
        "label": "Strike %",
        "kind": "count_ratio",
        # % of accounts current on their installment obligation this month, among
        # accounts with a valid (Y/N, or spelled-out Yes/No -- some monthly LCC
        # extracts spell the flag out instead of abbreviating it) Strike value.
        # Must match utils.compute_strike_pct (the shared helper used everywhere
        # else -- dashboard + Portfolio Intelligence). This declarative definition
        # is consumed by the general compiler rather than calling that function
        # directly, so the two can't share code, but tests/test_metric_consistency.py
        # checks they stay numerically identical.
        "numerator_where": [{"column": "Strike", "op": "in", "value": ["Y", "YES"]}],
        "denominator_where": [{"column": "Strike", "op": "in", "value": ["Y", "N", "YES", "NO"]}],
        "scale": 100,
        "grain": "loan",
        "description": "% of accounts current on their installment (Strike=Y) among accounts with a valid Strike value.",
    },
}

# Domain-specific terms confirmed (live, on real data) to have more than one
# materially different reading in THIS business -- unlike CONCEPTS/METRICS/VIEWS
# above (a fixed vocabulary the planner picks FROM), this is a fixed list of
# vocabulary the planner must actively watch OUT for in the user's own wording
# and ask about, rather than silently pick a default reading. Seeded from two
# real gaps observed live: "which branch had best business" and "show me the
# risky accounts" both resolved to ONE interpretation without asking, despite
# each having a second, equally plausible, materially different reading that a
# generic "is this ambiguous" LLM judgment call didn't catch on its own --
# these terms are ambiguous because of NBFC-domain meaning specifically (e.g.
# "business" meaning new originations, not portfolio health), not because of
# generic sentence-level vagueness the model already handles well (it correctly
# asks for "best branch" alone, since ranking-by-what is a generic ambiguity).
# Extend this list the same way CLAUDE.md describes growing the rest of the
# registry vocabulary: from real query-log evidence, not speculation.
AMBIGUOUS_TERMS = [
    {
        "term": "business",
        "note": (
            '"business" is ambiguous in this NBFC domain, between TWO REAL, '
            "SEPARATELY-ANSWERABLE views -- not just two ways of describing the same "
            "answer:\n"
            "    (1) NEW LOANS ORIGINATED this period -- the new_advances_by_region / "
            "new_advances_by_branch / new_advances_by_executive views (disbursement "
            "volume, accounts funded this month).\n"
            "    (2) PORTFOLIO/COLLECTION PERFORMANCE -- region_scorecard / "
            "branch_quadrant / executive_recovery views (Collection%, NPA%, Strike%, "
            "Concern Score, etc).\n"
            "  These are computed from DIFFERENT rows (Ag_Date-filtered originations vs. "
            "the whole current book) and give completely different rankings -- a branch "
            "can lead on one and trail on the other. If the query does not already "
            'specify which reading (e.g. "new business", "new advances", "originated", '
            '"disbursed", "funded" clearly means (1); "collection performance", "NPA%", '
            '"concern score", "delinquency" clearly means (2)), clarification_options '
            "MUST include AT LEAST ONE option from EACH reading -- never 2-4 options that "
            "are all flavors of the same one (e.g. offering only Collection Efficiency / "
            "Strike Rate / NPA% is WRONG here, since all three are reading (2) only, "
            "reading (1) is completely missing). Minimum valid example:\n"
            '    ["New advances (loans originated this period)", '
            '"Portfolio/collection performance (Collection%, NPA%, Concern Score)"]'
        ),
    },
    {
        "term": "risk / risky",
        "note": (
            '"risk"/"risky" is ambiguous: could mean NPA% (90+ DPD), SMA-2% '
            "(early-stage delinquency), Hard Bucket% (deep arrears), Co-lending "
            "exposure (partner-bank risk), or a composite Concern Score -- each is a "
            "different metric with a different ranking. If the query does not "
            "already name one of these specifically, ask for clarification listing "
            "these as options rather than defaulting to one."
        ),
    },
]
