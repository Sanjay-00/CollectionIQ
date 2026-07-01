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

  NOTE: Phase 0 includes only concepts expressible as an AND of column/op/value
  conditions (which the current engine runs directly). Concepts that need a
  column-vs-column comparison (short_collection, under_collected) are intentionally
  deferred until the compiler grows that primitive  -  they are not encoded here yet
  rather than encoded in a form the engine cannot run.

  __CUTOFF_1Y__ is a dynamic placeholder (loan agreement date within last 12
  months); execute_priority_mode resolves it today, and the v2 compiler resolves
  it at lowering time. Same convention as PRIORITY_RULES.

- METRICS  -  named numeric measures with their default aggregation AND grain. The
  grain is part of the definition (a measure is "sum of SOH AT loan grain"), so
  the compiler can prevent fan-out double-counting when grouping.

Nothing here except PRIORITY_RULES is consumed yet  -  CONCEPTS/METRICS are additive
in Phase 0, so they cannot change behavior.
"""

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
        "why": "Closing arrears < ₹1000 - one call can clear these, quick wins for collection team",
        "conditions": [
            {"column": "Closing Arrears", "op": ">",  "value": 0},
            {"column": "Closing Arrears", "op": "<",  "value": 1000},
        ],
    },
    {
        "rank": 3,
        "label": "Recent Advances - High Bucket",
        "why": "Loans sanctioned within last 12 months already in SMA-1 or worse  -  early warning of sourcing quality issues",
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
            {"column": "ARREARS AGAINST EXP",  "op": ">",  "value": 5000},
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
        "description": "Closing arrears between 0 and ₹1000 - one call can clear these.",
        "conditions": [
            {"column": "Closing Arrears", "op": ">", "value": 0},
            {"column": "Closing Arrears", "op": "<", "value": 1000},
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
        "description": "EMI paid but unpaid insurance/expense charge (> ₹5000) creates artificial arrears.",
        "conditions": [
            {"column": "ARREARS AGAINST INST", "op": "<=", "value": 0},
            {"column": "ARREARS AGAINST EXP", "op": ">", "value": 5000},
            {"column": "Arrears / EMI", "op": ">", "value": 0},
        ],
    },
    "recent_advance_high_bucket": {
        "label": "Recent Advance - High Bucket",
        "description": "Loan sanctioned within last 12 months already in SMA-1 or worse.",
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
        "description": "Zero payment this month despite a demand due.",
        "conditions": [
            {"column": "Month Collection (Excluding Reserve Collection)", "op": "==", "value": 0},
            {"column": "Net Collection Demand Inst+Exp+BC", "op": ">", "value": 0},
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
        "description": "A customer holding 3 or more loans/vehicles.",
        "having": [{"agg": "nunique", "column": "Loan No", "op": ">=", "value": 3}],
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
}
