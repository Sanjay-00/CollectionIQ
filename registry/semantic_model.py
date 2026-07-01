"""Semantic Data Model  -  the STRUCTURE layer of the registry.

CollectionIQ operates over a single denormalised in-memory DataFrame: Loan,
Customer, Executive, Branch and Region are all columns on the SAME table. So the
"relationships" here are NOT foreign-key joins  -  they are a grain lattice plus an
entity->key-column map, expressing functional dependencies between columns.

Why this exists: the v2 compiler derives the execution DAG deterministically from
grain (e.g. "customers per branch with >3 loans" needs a per-customer rollup then
a per-branch count  -  derivable purely from the fact that customer is coarser than
loan). The LLM must NEVER author that nesting; the compiler does, from this model.

CRITICAL  -  the hierarchy is NOT a strict tree. Executive, Branch and Region are
attributes of the LOAN, not of the customer: one customer (one mobile number) can
hold loans across multiple branches. So "customer -> branch" is not a functional
dependency. This model records, per entity, whether it is an atomic grain, a true
rollup of loans, or a loan-level dimension  -  so the compiler can DETECT grain
ambiguity (and route to clarification) instead of silently guessing.
"""

# ── Entities ────────────────────────────────────────────────────────────────
# kind:
#   "atomic"     -  the base grain; every row is one of these (loan)
#   "rollup"     -  a genuinely coarser grain reachable by grouping the atomic rows
#                 on its key (customer = group loans by mobile number)
#   "dimension"  -  a loan-level attribute used for grouping/slicing; does NOT
#                 functionally depend on the customer rollup (a customer may span
#                 several values), so grouping a customer-grain measure by one of
#                 these is ambiguous and must be flagged by the compiler.
# grain: smaller = finer. Used only to compare coarseness between atomic/rollup
#        entities; dimensions share the loan grain (they live on the loan row).
ENTITIES: dict[str, dict] = {
    "loan": {
        "key": ["Loan No"],
        "kind": "atomic",
        "grain": 0,
    },
    "customer": {
        "key": ["Cust Mob No"],
        "kind": "rollup",
        "grain": 1,
        "rolls_up_from": "loan",
    },
    "executive": {
        # The same executive name recurs across branches, so an executive is only
        # uniquely identified together with the branch (see GROUP_BY RULES prose).
        "key": ["MNT NAME", "Unit"],
        "kind": "dimension",
        "grain": 0,
    },
    "branch": {
        "key": ["Unit"],
        "kind": "dimension",
        "grain": 0,
    },
    "region": {
        "key": ["RegionName"],
        "kind": "dimension",
        "grain": 0,
    },
}

# ── Dimension aliases ─────────────────────────────────────────────────────────
# Plain-English grouping words the user/LLM may reference -> the actual group_by
# column(s). Mirrors the GROUP_BY RULES + CUSTOMER IDENTITY sections of the prompt
# so the compiler can lower a logical "dimensions: [branch]" to real columns.
DIMENSIONS: dict[str, list[str]] = {
    "branch":     ["Unit"],
    "region":     ["RegionName"],
    "executive":  ["MNT NAME", "Unit"],
    "customer":   ["Cust Mob No"],
}


# ── Helpers (used by the compiler; kept tiny and pure) ────────────────────────

def entity_key(entity: str) -> list[str]:
    """Key column(s) that uniquely identify one instance of an entity."""
    return list(ENTITIES[entity]["key"])


def grain_level(entity: str) -> int:
    """Coarseness score; larger = coarser. loan=0, customer=1."""
    return int(ENTITIES[entity]["grain"])


def is_coarser(a: str, b: str) -> bool:
    """True if entity `a` is a strictly coarser grain than `b` (a rolls up b).
    e.g. is_coarser('customer', 'loan') == True. Used by the compiler to decide
    when a query needs a nested (multi-pass) rollup."""
    return grain_level(a) > grain_level(b)


def resolve_dimension(name: str) -> list[str]:
    """Map a dimension alias ('branch') to its group_by column(s) (['Unit']).
    Returns the name as-is (single-element list) if it is already a real column
    name rather than a known alias."""
    return list(DIMENSIONS.get(name, [name]))


# ── Temporal model ────────────────────────────────────────────────────────────
# Describes the time semantics of the data layer. Currently the system loads
# 1 - 2 monthly snapshots; prev_* columns are merged onto the current DataFrame
# (see utils.PREV_CARRYOVER_COLS). The temporal MODEL is forward-compatible with
# multi-period storage, but genuine windows/cohorts/trends require a data-layer
# change (persisting many periods) and are intentionally out of scope here.
#
# Step 3 of IR-2: the compiler maps a logical `time.compare` block to the
# existing prev_* column references deterministically, so the LLM never authors
# raw prev_* column names  -  it only declares {"time": {"compare": {"from":"prev",
# "to":"curr"}}}. The hack becomes a modeled special case.
TIME_MODEL: dict = {
    # The column that marks when each row's data was captured.
    "axis_column":  "Ag_Date",
    "period_grain": "month",

    # snapshot = data is a point-in-time stock (SOH, arrears, bucket).
    # NOT additive over time: summing SOH across months double-counts principal.
    # Flows (Month Receipt Amount) ARE additive over time but are a minority.
    "row_semantics": "snapshot",

    # Named periods the compiler resolves. "prev" maps to PREV_CARRYOVER_COLS
    # prefixes; "curr" maps to the current (non-prefixed) columns directly.
    "supported_periods": {
        "prev": "Previous month snapshot (prev_* columns; only present if a second file was uploaded).",
        "curr": "Current month snapshot (the main uploaded file).",
    },

    # Columns whose semantics are 'flow' (additive over time). All others are
    # treated as 'snapshot' stocks and must NOT be summed across periods.
    "flow_columns": frozenset({
        "Month Receipt Amount",
        "Month Collection (Excluding Reserve Collection)",
        "Month Due-Inst",
        "Month Due-Exp",
    }),
}
