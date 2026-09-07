"""
Gemini-facing routing/narration layer for the Investigator drill-down
assistant. Mirrors agents/logical_planner.py's client/retry pattern.

parse_turn's ONLY job is picking a step_type + params from
investigator.steps.STEP_REGISTRY's fixed vocabulary and resolving any
reference in the user's message ("him," "that branch") against the current
EntityMemory -- it never computes the numbers itself, and it is never handed
a full result table to aggregate over (see investigator/steps.py's module
docstring for why: that would reintroduce the "LLM does execution logic"
failure mode CLAUDE.md documents graph.py's own IR-1/compiler split was
built to avoid).

narrate_step writes a short plain-English observation about one already-
computed step result (same narrow role agents/insight_generator.py plays
for the main AI Query pipeline).

Both are the only 2 Gemini call sites in this module -- one per user turn,
one per step the user actually triggers -- matching this codebase's
existing Flash-Lite cost discipline (CLAUDE.md's "Model Choice" section).
"""
from __future__ import annotations

import json
import os
import re
from typing import Iterator

from google import genai

from agents.domain_expert import _add_token_usage, _call_gemini_with_retry
from config import GEMINI_MODEL, PLANNER_TEMPERATURE
from investigator.state import EntityMemory
from investigator.steps import METRIC_DIRECTION, STEP_REGISTRY
from registry.ontology import CONCEPTS
from registry.semantic_model import resolve_dimension

STEP_TYPES = sorted(STEP_REGISTRY.keys())


def _strip_code_fence(raw: str) -> str:
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    return raw


_MAX_NAMES_LISTED = 20

# underperformer_quartile/bottom_n_per_group both produce an
# executive-scorecard-shaped table under this entity_type.
_EXECUTIVE_GRAIN_ENTITY_TYPES = {"executive_group", "branch_executives"}


def _entity_memory_context(entity_memory: EntityMemory) -> str:
    """Plain-text summary of currently-remembered entities, most recent
    first -- gives the model what it needs to resolve "him"/"that branch"
    without ever handing it a full result table to compute over.

    For a dimension_breakdown entity, this also lists the actual entity
    NAMES that appeared as rows (e.g. "CS NAGAR, NAGPUR, ...") -- names
    only, never the metric values next to them. Without this, a live bug
    was caught: a user asked "why is CS Nagar not performing" right after
    seeing a region-level dimension_breakdown, and the model had no way to
    know "CS Nagar" was one of the regions just shown, so it guessed
    entity_type="branch" -- which silently returned an all-None table
    (Unit=="CS Nagar" matches nothing), no error at all.

    For an executive_group/branch_executives entity (underperformer_quartile
    or bottom_n_per_group's output), this similarly lists the executives
    shown, the branch they're scoped to, and which OTHER metrics are
    available on that same table. Without this, a second live bug was
    caught: a follow-up naming a different metric ("why strike rate is
    less") while an executive-grain table was already in focus reverted to
    an unrelated, coarser dimension_breakdown(region) -- the model had no
    explicit signal that Strike Rate % was already a real column on the
    executive-grain table right in front of it, scoped to the same branch.
    """
    entities = entity_memory.all()
    if not entities:
        return "No entities discussed yet this session."
    lines = ["Entities discussed this session, most recent first:"]
    for e in entities:
        is_dimension_breakdown = e.entity_type == "dimension_breakdown"
        is_executive_grain = e.entity_type in _EXECUTIVE_GRAIN_ENTITY_TYPES
        focus = f", currently focused on {e.metric_focus}" if e.metric_focus and not is_dimension_breakdown else ""
        lines.append(f"  - {e.entity_type}: {e.entity_value}{focus}")

        if is_dimension_breakdown and e.metric_focus and e.result_df is not None and not e.result_df.empty:
            col = resolve_dimension(e.metric_focus)[0]
            if col in e.result_df.columns:
                names = e.result_df[col].astype(str).unique().tolist()[:_MAX_NAMES_LISTED]
                lines.append(f"    (these {e.metric_focus}s were shown: {', '.join(names)})")

        elif is_executive_grain and e.result_df is not None and not e.result_df.empty and "MNT NAME" in e.result_df.columns:
            names = e.result_df["MNT NAME"].astype(str).unique().tolist()[:_MAX_NAMES_LISTED]
            units = e.result_df["Unit"].astype(str).unique().tolist() if "Unit" in e.result_df.columns else []
            scope_note = f", scoped to branch {units[0]}" if len(units) == 1 else ""
            available = [c for c in e.result_df.columns if c in METRIC_DIRECTION]
            lines.append(
                f"    (this is EXECUTIVE-grain data{scope_note} -- executives shown: "
                f"{', '.join(names)}; other metrics available on this SAME table: {', '.join(available)}. "
                f"A follow-up naming one of these metrics should STAY at executive grain with the "
                f"SAME scope, not jump to a coarser dimension_breakdown.)"
            )
    context = "\n".join(lines)
    context += _current_scope_hint(entities)
    context += _next_executive_hint(entities)
    return context


def _entity_branch_scope(record) -> str | None:
    """Best-effort branch a record is ultimately scoped to -- used by both
    _current_scope_hint (so priority_menu etc. can inherit the active focus
    instead of silently going portfolio-wide) and _next_executive_hint
    (to find the roster a single executive was drilled from). Returns None
    for a record with no branch-level scope (portfolio-wide, or scoped only
    to a region/zone/BU)."""
    if record.entity_type == "branch":
        return str(record.entity_value)
    if record.entity_type == "executive":
        value = record.entity_value
        if isinstance(value, (list, tuple)) and len(value) == 2:
            return str(value[1])
        return None
    params = record.params or {}
    if params.get("scope_col") == "branch" and params.get("scope_value"):
        return str(params["scope_value"])
    if params.get("scope_col") == "executive":
        value = params.get("scope_value")
        if isinstance(value, (list, tuple)) and len(value) == 2:
            return str(value[1])
    return None


def _current_executive_focus(record) -> tuple[str, str] | None:
    """(name, branch) of the SINGLE executive currently in focus, or None --
    covers both entity_summary(executive) and any step scoped to one
    executive via scope_col="executive" (e.g. worst_loans_by_metric)."""
    if record.entity_type == "executive":
        value = record.entity_value
        if isinstance(value, (list, tuple)) and len(value) == 2:
            return str(value[0]), str(value[1])
        return None
    params = record.params or {}
    if params.get("scope_col") == "executive":
        value = params.get("scope_value")
        if isinstance(value, (list, tuple)) and len(value) == 2:
            return str(value[0]), str(value[1])
    return None


def _current_scope_hint(entities: list) -> str:
    """A live bug: mid-conversation about one branch, "what should I focus
    on" routed priority_menu portfolio-wide (unscoped) instead of staying on
    the branch just discussed -- there was no signal telling the planner a
    scope was even available to inherit. This computes that scope in plain
    Python (never guessed by the LLM) from whichever entity is most recently
    touched, so a step with an optional scope_col/scope_value can default to
    it instead of silently discarding the conversation's current focus."""
    if not entities:
        return ""
    branch = _entity_branch_scope(entities[0])
    if not branch:
        return ""
    return (
        f'\nCURRENT SCOPE: branch "{branch}" (from the entity currently in focus). For any step '
        f"that accepts an optional scope_col/scope_value (e.g. priority_menu, concept_filter, "
        f'top_closing_arrears), prefer scope_col="branch", scope_value="{branch}" over leaving it '
        f"unscoped, UNLESS the question explicitly asks for the whole portfolio or names a "
        f"different branch/region/zone/BU -- an unscoped call silently jumps back to portfolio-wide "
        f"and discards what's currently being discussed."
    )


def _next_executive_hint(entities: list) -> str:
    """Deterministic (no-LLM-guessing) hint for a "next executive"/"another
    one" follow-up. A live bug: there was no vocabulary at all for "move to
    the next executive" -- Gemini had nothing to map it to and fell back to
    re-showing the SAME roster table already in memory. This finds the
    roster the currently-focused single executive was originally drilled
    from (an executive_group/branch_executives record scoped to the SAME
    branch) and names the next executive after them in that roster's own
    row order -- computed here, not left for the LLM to infer from a name
    list, because a wrong "next" would silently point a leader at the wrong
    person's numbers (the same class of risk resolve_executive_identity
    already exists to guard against elsewhere in this module)."""
    if not entities:
        return ""
    focus = _current_executive_focus(entities[0])
    if focus is None:
        return ""
    current_name, current_branch = focus

    for record in entities:
        if record.entity_type not in _EXECUTIVE_GRAIN_ENTITY_TYPES:
            continue
        df = record.result_df
        if df is None or df.empty or "MNT NAME" not in df.columns or "Unit" not in df.columns:
            continue
        units = df["Unit"].astype(str).str.upper().unique().tolist()
        if len(units) != 1 or units[0] != current_branch.upper():
            continue
        names = df["MNT NAME"].astype(str).tolist()
        upper_names = [n.upper() for n in names]
        if current_name.upper() not in upper_names:
            continue
        idx = upper_names.index(current_name.upper())
        if idx + 1 < len(names):
            next_name = names[idx + 1]
            return (
                f'\nNEXT EXECUTIVE HINT: the executive currently in focus ("{current_name}", branch '
                f"{current_branch}) is at position {idx + 1} of {len(names)} in the roster shown "
                f'earlier. If the question asks to move to the next executive, route to entity_summary '
                f'with entity_type="executive", entity_value=["{next_name}", "{current_branch}"].'
            )
        return (
            f'\nNEXT EXECUTIVE HINT: the executive currently in focus ("{current_name}", branch '
            f"{current_branch}) is LAST (position {len(names)} of {len(names)}) in the roster shown "
            f"earlier -- there is no next executive. If asked to move to the next one, set "
            f"needs_clarification true and say there are no more executives left in that list."
        )
    return ""


# Exact param contract per step type -- a live-Gemini smoke check caught the
# earlier, name-list-only prompt producing WRONG param keys (e.g. "group_by"
# instead of "group_col", a bare "branch" key that underperformer_quartile
# doesn't accept at all) and the WRONG step type for an open "why is X
# underperforming" question (it should route to entity_summary first, not
# jump straight to a metric-specific step before the RBH has even picked a
# metric). Each entry below is a worked example the model can copy the SHAPE
# of, not just a name -- this is deliberately as explicit as
# agents/logical_planner.py's own IR-1 schema section, for the same reason:
# a name-only vocabulary list under-specifies structured extraction.
_HIERARCHY_RULE = """
ONE-LEVEL-AT-A-TIME DRILL RULE: the org hierarchy is
BU > Zone > Region > Branch > Executive > Customer. When the entity
currently in focus (see entity memory below) is coarser than Branch (a BU,
Zone, or Region), and the question asks "why is X bad" or drills deeper
WITHOUT explicitly naming executives/officers, move exactly ONE level down
-- e.g. focus is a Region -> use dimension_breakdown(dimension="branch",
scope_col="region", scope_value=<that region>) to show branches within it,
NOT underperformer_quartile/bottom_n_per_group (those are executive-grain
and would skip Branch entirely). Only go to executive-grain steps once
Branch is already the focus, or the question explicitly asks about
executives/officers ("which executive," "who is responsible").

STAY-AT-GRAIN RULE: this cuts the OTHER way too. If the entity currently in
focus is ALREADY executive-grain (entity memory below says "this is
EXECUTIVE-grain data"), and the follow-up just names a DIFFERENT metric
that's already listed as available on that same table (e.g. "why is strike
rate less" after a Collection% executive table), do NOT jump back to a
coarser dimension_breakdown -- re-run underperformer_quartile or
bottom_n_per_group with the SAME scope (same scope_col/scope_value or
group_col) shown in that entity's context, just with the new metric. A
metric-focused follow-up never coarsens the grain on its own.
""".strip()

_CONTEXT_HINT_RULE = """
CONTEXT HINTS: the entity memory below may end with a CURRENT SCOPE line
and/or a NEXT EXECUTIVE HINT line -- both are computed directly from stored
data, not guessed, and MUST be trusted over re-deriving scope or order from
names yourself. Use CURRENT SCOPE to fill scope_col/scope_value on a step
that accepts an optional scope whenever the question doesn't explicitly ask
for the whole portfolio or a different branch/region ("what should I focus
on," "what should my team work on" asked mid-conversation about one branch/
executive should stay scoped to it, not jump back to portfolio-wide). Use
NEXT EXECUTIVE HINT exactly as given for a "move to the next executive,"
"another one," or "next one" follow-up -- route to entity_summary with the
EXACT entity_value it names; if it says there is no next executive, set
needs_clarification true instead of guessing a name.
""".strip()

_STEP_EXAMPLES = """
dimension_breakdown -- use this for a PLURAL/comparative question naming NO
single entity ("how are my regions performing," "show me all branches,"
"which zones are struggling," "why is NPA more" with no single entity named
-- "here are the branches/regions with the worst NPA," NOT a jump straight
to executives). One row per branch/region/zone/BU. NEVER use entity_summary
for this (entity_summary requires exactly ONE named entity_value; a plural
question has none, and forcing one is a crash, not a graceful fallback).
Optionally scope_col/scope_value narrow to ONE entity one level up first
(e.g. "show me branches within CS Nagar region" -- CS Nagar is the region,
branch is what's being broken down -- THIS is also how you drill one level
down the hierarchy, see the ONE-LEVEL-AT-A-TIME rule below). Optionally
metric/higher_is_better sort the result by whichever metric the question is
actually ABOUT, worst-first -- a "why is X bad" question naming a metric
MUST pass that metric here, otherwise the table silently sorts by
Collection% regardless of what was asked, which is wrong. Params:
  {"dimension": "region"}
  {"dimension": "branch"}
  {"dimension": "zone"}
  {"dimension": "bu"}
  {"dimension": "branch", "scope_col": "region", "scope_value": "CS Nagar"}
  {"dimension": "branch", "scope_col": "region", "scope_value": "CS Nagar",
   "metric": "NPA%", "higher_is_better": false}

entity_summary -- use this FIRST for an open "why is X underperforming/doing
well" question that names ONE SPECIFIC entity by name. entity_type is
whichever level of the org hierarchy (BU > Zone > Region > Branch) OR
executive that entity ACTUALLY IS -- branch/region/zone/bu/executive are
ALL valid entity_types here, not just branch. Critically: if that exact
name was already shown in a table this session (check the entity memory
context below, and note which dimension any dimension_breakdown used), use
THAT SAME entity_type -- do not guess "branch" by default for a name you
don't recognize. Getting this wrong silently returns an all-None table with
no error, which is worse than asking for clarification. NEVER use this for
"list/show me all executives of X branch/region" -- that names a branch but
is asking for a ROSTER of the executives inside it, not a metrics summary
of the branch itself; use executive_roster below instead. Params:
  {"entity_type": "branch", "entity_value": "Mahad"}
  {"entity_type": "region", "entity_value": "CS Nagar"}
  {"entity_type": "zone", "entity_value": "North Zone"}
  {"entity_type": "bu", "entity_value": "CV"}
  {"entity_type": "executive", "entity_value": ["Rahul", "Pune Unit"]}

executive_roster -- use this for "list/show me all executives of X branch,"
"who are the executives in Y region," or similar plain rosters -- returns
EVERY executive at the given scope when no n/metric is given, unlike
underperformer_quartile (only the worst quartile survives) or
bottom_n_per_group (caps each group at N, always worst-first). Optionally
scope_col/scope_value narrow to one branch/region/zone/BU; omit both for a
portfolio-wide roster. ALSO use this (with n + metric) for "top N
executives [in X] by METRIC" -- best performers first by default (the
colloquial meaning of "top N"), unlike every other ranked step in this
vocabulary which is worst-first. Set worst_first: true instead for a PLAIN
"bottom N executives [by METRIC]" with NO group phrase (e.g. "bottom 2
executives by collection percent," no "per branch"/"per zone" named) --
this is the correct step for that, NOT bottom_n_per_group (which requires
an explicit group_col and NEVER accepts "executive" as one -- there is no
"per executive" grouping). Params:
  {}
  {"scope_col": "branch", "scope_value": "Mahad"}
  {"scope_col": "region", "scope_value": "WEST"}
  {"scope_col": "branch", "scope_value": "Mahad", "n": 5, "metric": "Collection %"}
  {"n": 10, "metric": "NPA %", "worst_first": true}

roll_rate_summary -- use this for a question about bucket-migration
MECHANISM, not location: "why is NPA up" asked in a way that wants to know
HOW, not WHO -- "is it roll-forward or roll-backward driven," "are accounts
sliding into worse buckets," "what's our roll rate," "how many accounts
newly turned NPA this month." Returns Roll Forward Rate % (existing
accounts worsening), Roll Backward Rate % (existing accounts recovering),
NPA Formation Rate % (non-NPA accounts newly turning NPA), plus matched/new/
exited account counts. Distinct from dimension_breakdown/entity_summary
(those answer WHICH branch/region is worst, never HOW/WHY it's moving) --
requires BOTH a current and a previous period (returns nothing usable with
only one). Optionally scope_col/scope_value narrow to ONE branch/region/
zone/BU. NEVER use this for "WHICH branch had the most roll forward" or any
question asking to RANK/COMPARE roll-rate across multiple entities -- this
only ever computes ONE scope's rates, use roll_rate_by_dimension below for
a ranked breakdown instead. Params:
  {}
  {"scope_col": "branch", "scope_value": "Mahad"}
  {"scope_col": "region", "scope_value": "WEST"}

vintage_summary -- use this for a question about ORIGINATION-COHORT
mechanism, not location: "is this a recent-book problem," "are new loans
defaulting early," "is delinquency concentrated in recent originations,"
"show me NPA% by disbursement month/vintage." One row per DISBURSEMENT
MONTH cohort with NPA%/SMA-2%/Collection% for that cohort alone -- distinct
from roll_rate_summary (this-month bucket migration, not origination
cohort) and dimension_breakdown/entity_summary (org grain, never cohort).
A cohort with too few accounts is silently excluded (not an error).
Optionally scope_col/scope_value narrow to one branch/region/zone/BU.
Params:
  {}
  {"scope_col": "branch", "scope_value": "Mahad"}

demand_summary -- use this for a question about COLLECTION-COMPOSITION
mechanism: is poor collection% driven by failing to recover OLD arrears
(Overdue Collection%) or by failing to collect THIS MONTH's own fresh EMI
demand (Month Demand Collection%) -- "is this an old-arrears problem or a
fresh-collection problem," "break down collection into overdue vs month
demand," "show overdue vs current demand collection by branch/executive."
dimension is "region", "branch", or "executive" (which grain's rows to
return -- required). Distinct from dimension_breakdown/entity_summary
(plain Collection%, not decomposed this way) and from roll_rate_summary/
vintage_summary (different mechanism lenses entirely -- bucket migration
and origination cohort, not collection composition). Optionally scope_col/
scope_value narrow to one branch/region/zone/BU first. Params:
  {"dimension": "branch"}
  {"dimension": "executive", "scope_col": "branch", "scope_value": "Mahad"}

roll_rate_by_dimension -- use this for "WHICH branch/region/zone/executive
had the most roll forward/backward," "rank branches by roll rate," or
similar -- a RANKED BREAKDOWN of bucket-migration mechanism across many
entities, unlike roll_rate_summary (which only ever computes ONE scope's
rates, portfolio-wide or narrowed to a single named entity, never a ranked
list). dimension is "branch", "region", "zone", "bu", or "executive"
(required -- which grain to rank). Needs BOTH a current and a previous
period. Params:
  {"dimension": "branch"}
  {"dimension": "executive"}

underperformer_quartile -- worst quartile of ONE metric, AT EXECUTIVE GRAIN
ONLY (every row is one executive) -- narrowed to one branch/region/zone/BU
via scope_col/scope_value, but the ROWS returned are always executives, never
branches/regions themselves. Do NOT use this to answer "why is region/branch
X bad" -- that question is answered by dimension_breakdown or entity_summary
at the region/branch grain (see the ONE-LEVEL-AT-A-TIME rule above). Use
this only once the question is specifically about WHO (which executives) is
responsible, not before. Params (metric is REQUIRED; only use this once a
metric is actually named):
  {"metric": "Collection %", "higher_is_better": null, "scope_col": null, "scope_value": null}
  {"metric": "NPA %", "higher_is_better": null, "scope_col": "region", "scope_value": "WEST"}
  {"metric": "NPA %", "higher_is_better": null, "scope_col": "zone", "scope_value": "NORTH ZONE"}
  {"metric": "NPA %", "higher_is_better": null, "scope_col": "bu", "scope_value": "CV"}

bottom_n_per_group -- worst N executives PER GROUP -- ONLY when the question
explicitly names what to group by, e.g. "bottom 2 PER branch," "bottom 2 PER
zone" (one flagged sub-list for EACH branch/zone). group_col is "branch",
"region", "zone", or "bu" (never "executive" -- executive is what's being
ranked, not what's being grouped by, and there is no "per executive"
grouping). A PLAIN "bottom N executives [by METRIC]" with NO group phrase
at all (no branch/region/zone/bu named as what to group by) is NOT this
step -- that is one flat ranked list, use executive_roster instead with
n/metric/worst_first: true (optionally scope_col/scope_value if ONE
branch/region/zone/BU is named as where to look, which is scoping, not
grouping). Params:
  {"metric": "Collection %", "n": 2, "group_col": "branch", "higher_is_better": null}

intersect_entities -- entities common to several PREVIOUSLY COMPUTED results
already in entity memory (source_entities names them). Params:
  {"source_entities": [{"entity_type": "branch_executives", "entity_value": "Mahad:Collection %"},
                        {"entity_type": "branch_executives", "entity_value": "Mahad:NPA %"}],
   "min_appearances": null}

non_paying_customers -- customers behind a previously-computed executive
table already in entity memory, OR behind a SINGLE executive currently in
focus (entity_type "executive" -- e.g. right after "why does Tausif Khan
have low collection," a follow-up like "who are the customers behind him"
or "show me customers responsible for his low strike rate" references that
SAME executive entity, not a group table). ALWAYS reuse the exact
entity_type/entity_value already shown for the CURRENTLY FOCUSED
executive/group in the entity memory context below -- never invent one,
and never substitute a DIFFERENT executive-shaped entity from memory just
because the current focus doesn't look like a group table. Both this and
customer_loan_book default to a CURATED subset of columns (Ag_Date, POS,
Closing Arrears, LCC%, Arrears / EMI, Last Receipt Date/Amount, etc) -- if
the question asks for "all columns," "every column," "full details," or
names a SPECIFIC column not in that curated list, set all_columns: true and
RE-RUN this same step (don't try to enumerate individual extra columns,
that's not a real parameter -- all_columns is the only lever). Params:
  {"source_entities": [{"entity_type": "branch_executives", "entity_value": "Mahad:Collection %"}]}
  {"source_entities": [{"entity_type": "executive", "entity_value": ["Tausif Khan Nazir Kh", "BHSWL"]}]}
  {"source_entities": [{"entity_type": "branch_executives", "entity_value": "Mahad:Collection %"}],
   "all_columns": true}

customer_loan_book -- every loan for one customer. Same all_columns lever
as non_paying_customers above. Params:
  {"cust_mob_no": "9876543210"}
  {"cust_mob_no": "9876543210", "all_columns": true}

concept_filter -- use this for a question naming one of the KNOWN CONCEPTS
below ("are there any non starters in my region," "show me easy
settlements," "which accounts are colending at risk") -- these are
predefined business rules, NOT metrics, and NOT something to answer via
dimension_breakdown or underperformer_quartile. Optionally scope_col/
scope_value narrow to one branch/region/zone/BU. Params:
  {"concept": "non_starter"}
  {"concept": "non_starter", "scope_col": "region", "scope_value": "WEST"}

concept_breakdown -- "which branch/executive has the most [of a concept
already shown]," e.g. "which branch has the most non-starters" right after
a concept_filter result. Only use this as a follow-up on an EXISTING
concept_filter result already in entity memory (source_entities names it)
-- never call this as the first step of a question; if no concept_filter
result exists yet for what's being asked, use concept_filter first instead.
dimension is "branch" or "executive" (never a metric -- this counts rows of
the ALREADY-FILTERED table, it does not recompute anything portfolio-wide).
Params:
  {"source_entities": [{"entity_type": "concept_filter", "entity_value": "non_starter"}],
   "dimension": "branch"}

high_arrears_at_risk -- use this for "high arrears cases," "loans at risk of
write-off," "accounts where arrears exceed [a large %] of the loan amount,"
or similar -- a named priority category, NOT a registered CONCEPT (this
ratio can't be expressed as a simple concept condition), so it's its own
step. Optionally scope_col/scope_value narrow to one branch/region/zone/BU.
Params:
  {}
  {"scope_col": "branch", "scope_value": "Mahad"}

fleet_defaulters -- use this for "which fleet owners are defaulting,"
"fleet operators with overdue EMIs," "customers with multiple loans who are
delinquent," or similar -- fleet operators (customers holding several
loans) who are CURRENTLY behind on payment, loan-level, sorted by exposure.
Distinct from a plain concept_filter("delinquent") (that has no fleet-size
filter) and from a plain fleet lookup (that has no delinquency filter) --
this is specifically the INTERSECTION. Optionally scope_col/scope_value
narrow to one branch/region/zone/BU. Params:
  {}
  {"scope_col": "branch", "scope_value": "Mahad"}

worst_loans_by_metric -- use this for "why does [executive/branch] have so
much NPA%/such a low strike rate/collection%," "which loans are dragging
[X] down," or any question wanting the actual LOAN-LEVEL rows responsible
for a SPECIFIC metric already in focus -- the natural last hop after
underperformer_quartile/bottom_n_per_group (executive grain) or
dimension_breakdown (branch grain), when the question goes one level
FURTHER than "which executive/branch," down to "which actual loans/
customers." metric MUST be one of "NPA %", "Hard Bucket %", "Collection %",
"Strike Rate %" (also accepts the no-space spelling) -- this maps that
metric to its matching registered concept and returns sorted-worst-first,
customer-named rows, distinct from non_paying_customers (a fixed generic
"not paying" threshold, not tied to any specific metric) and from
concept_filter (which needs the concept named directly, not derived from a
metric). scope_col/scope_value narrow to ONE executive (["name","branch"]
pair, both required together to avoid matching the wrong person sharing a
first name in a different branch), or one branch/region/zone/BU. Same
all_columns lever as non_paying_customers/customer_loan_book/
priority_accounts -- if the question asks for "all columns," "every
column," "full details," or names a specific column not in the default
curated set, set all_columns: true and RE-RUN this same step. Params:
  {"metric": "NPA %", "scope_col": "executive", "scope_value": ["Rahul", "Pune Unit"]}
  {"metric": "Collection %", "scope_col": "branch", "scope_value": "Mahad"}
  {"metric": "NPA %", "scope_col": "executive", "scope_value": ["Rahul", "Pune Unit"], "all_columns": true}

new_advances_summary -- use this for "how much new business did we do this
month," "how many new advances," "what's our funded amount/avg ticket size
this month," or similar -- BUSINESS VOLUME (originations), a completely
different axis from every other step in this vocabulary (all of which
answer collection performance, not how much new business was written).
Never gated by delinquency bucket -- a fresh advance counts even if already
overdue by upload time. Optionally scope_col/scope_value narrow to one
branch/region/zone/BU. Params:
  {}
  {"scope_col": "branch", "scope_value": "Mahad"}

new_advances_by_dimension -- use this for "which branch/region/executive
originated the most new business," "who has the most new advances," "rank
branches by new business this month," or similar -- the ranked-breakdown
counterpart to new_advances_summary above (which only ever answers ONE
scope's own totals). dimension is "region", "branch", or "executive"
(required -- which grain to rank). Optionally scope_col/scope_value narrow
the INPUT to one branch/region/zone/BU first (e.g. "new advances by
executive within Mahad"). Params:
  {"dimension": "branch"}
  {"dimension": "executive", "scope_col": "branch", "scope_value": "Mahad"}

product_analysis -- use this for "which segment/fuel-type/source has the
worst NPA%," "break down NPA by product segment," "how does DIESEL vs
PETROL perform," or similar -- a PRODUCT/CHANNEL attribute breakdown,
distinct from vintage_summary (that's origination COHORT, i.e. WHEN a loan
was disbursed, never product/channel). axis is "segment", "fuel", or
"source" (required -- which attribute to break down). Optionally scope_col/
scope_value narrow to one branch/region/zone/BU. Params:
  {"axis": "segment"}
  {"axis": "fuel", "scope_col": "branch", "scope_value": "Mahad"}

top_accounts -- use this for "largest exposures," "biggest accounts at
risk," "top N accounts by outstanding," or similar -- the largest single
exposures among DELINQUENT accounts (any non-STD bucket) by SOH. Distinct
from top_closing_arrears (raw Closing Arrears value, ANY status, not
restricted to delinquent accounts) and from high_arrears_at_risk (a RATIO,
not a plain SOH ranking). n defaults to 20. Optionally scope_col/
scope_value narrow to one branch/region/zone/BU. Params:
  {}
  {"n": 10, "scope_col": "branch", "scope_value": "Mahad"}

new_advances_trend -- use this for "show me the last N months of new
business," "new advances trend," "quarterly/monthly new business over
time," or similar -- MONTH-BY-MONTH new-business volume across this SINGLE
upload's own Ag_Date history (not subject to the app's 2-uploaded-file
ceiling other trend-like questions run into, since it never needs a
previous file). Distinct from new_advances_summary (ONE month's own
totals, no history) and new_advances_by_dimension (ranked by entity, not by
time). months is a lookback window (null/omitted means "all history");
granularity is "Monthly" (default), "Quarterly", "Half-Yearly", "Yearly",
or "Financial Year". Optionally scope_col/scope_value narrow to one branch/
region/zone/BU. Params:
  {}
  {"months": 6}
  {"granularity": "Quarterly", "scope_col": "branch", "scope_value": "Mahad"}

fleet_exposure -- use this for "which customers have multiple loans," "who
are our biggest fleet operators," "customers with the most vehicles
financed," or similar -- customer-level, ranked by total exposure, NO
delinquency filter (a relationship/cross-sell/concentration question, not a
collections-risk one). Distinct from fleet_defaulters (loan-level, ONLY
customers who are CURRENTLY delinquent -- use fleet_defaulters instead for
"which fleet owners are defaulting/overdue/behind on payment"). Optionally
scope_col/scope_value narrow to one branch/region/zone/BU. Params:
  {}
  {"scope_col": "branch", "scope_value": "Mahad"}

repossession_list -- use this for "which accounts are eligible for
repossession," "vehicles we can repossess," "SMA-2/NPA accounts still
within the collateral window," or similar -- deep-delinquent (SMA-2/NPA)
accounts on loans recent enough that the vehicle collateral still has
resale value. Optionally scope_col/scope_value narrow to one branch/
region/zone/BU. Params:
  {}
  {"scope_col": "branch", "scope_value": "Mahad"}

good_customers -- use this for "which customers are eligible for refinance,"
"loyal customers we should re-approach," "good customers for relationship
management," or similar -- a BUSINESS-DEVELOPMENT question, distinct from
every other step in this vocabulary (all of which answer collection risk
or origination volume, never "who should we proactively court"). Fixed
criteria (tenure completed >= 70%, LCC% >= 100%, sorted lowest exposure
first). Optionally scope_col/scope_value narrow to one branch/region/
zone/BU. Params:
  {}
  {"scope_col": "branch", "scope_value": "Mahad"}

priority_menu -- use this FIRST for "which loans should I focus on," "what
should my team work on today," "which loans to prioritise," or similar --
this is now the DEFAULT for this kind of question. Returns COUNTS per
priority category (Non Starter, NPA Accounts, High Closing Arrears, Fleet
Owners with High POS, Co-lending at Risk, Insurance-Only Delinquency, Easy
Settlement, Recent Advances - High Bucket, No Collection 3 Months) instead
of dumping every loan at once -- the human picks ONE category to drill
into next (as a typed follow-up naming that category, which routes to
concept_filter/fleet_defaulters/top_closing_arrears directly). Each
category's count is independent -- a loan CAN appear in more than one
category, this is NOT deduplicated to one tier per loan (that's
priority_accounts below, a different, narrower use case). Optionally
scope_col/scope_value narrow to one branch/region/zone/BU. Params:
  {}
  {"scope_col": "branch", "scope_value": "Mahad"}

priority_accounts -- use this ONLY when the question explicitly asks for
the FULL combined priority list across every category AT ONCE, with no
menu ("give me the whole priority list," "everything, all tiers combined")
-- NOT for a plain "which loans should I focus on" (use priority_menu
above for that instead). A fixed 7-tier business framework (non-starters,
easy settlements, high-risk recent advances, insurance-driven delinquency,
co-lending risk, chronic non-collection, then NPA), each loan claimed under
its SINGLE highest-priority tier only (deduplicated -- distinct from
priority_menu's independent-per-category counts). Optionally scope_col/
scope_value narrow to one branch/region/zone/BU. Same all_columns lever as
non_paying_customers/customer_loan_book/worst_loans_by_metric -- if the
question asks for "all columns," "every column," "full details," or names
a SPECIFIC column not in the default curated set, set all_columns: true
and RE-RUN this same step. Params:
  {}
  {"scope_col": "branch", "scope_value": "Mahad"}
  {"all_columns": true}

top_closing_arrears -- use this for "top N loans with high closing
arrears," "which accounts have the biggest outstanding balance," or
similar -- a straight top-N by RAW Closing Arrears value (descending),
restricted to accounts that actually have arrears. Distinct from
high_arrears_at_risk (a RATIO -- Inst+Exp+BC arrears exceeding a % of loan
amount, "potential write-off" -- a different business question, not "top N
by rupee value"). top_n defaults to 20 if not given. Optionally scope_col/
scope_value narrow to one branch/region/zone/BU. Params:
  {}
  {"top_n": 10, "scope_col": "branch", "scope_value": "Mahad"}

priority_by_executive -- which executives have the most priority-flagged
accounts, most first. Only use this as a follow-up on an EXISTING
priority_accounts result already in entity memory (source_entities names
it) -- never call this as the first step of a question. Params:
  {"source_entities": [{"entity_type": "priority_accounts", "entity_value": "portfolio"}]}
""".strip()


def build_turn_prompt(question: str, entity_memory: EntityMemory) -> str:
    """Deterministic prompt text -- no Gemini call, fully unit-testable."""
    metrics = ", ".join(sorted(METRIC_DIRECTION.keys()))
    concepts = ", ".join(sorted(CONCEPTS.keys()))
    return f"""You are a sharp, experienced NBFC collections & credit risk analyst,
acting as a personal assistant to a busy leader -- a Regional Business Head,
Zonal Head, or Business Unit Head. You know this domain fluently (NPA,
SMA-1/SMA-2, hard bucket, strike rate, collection efficiency, the BU > Zone >
Region > Branch > Executive > Customer hierarchy) and use that fluency to
correctly interpret what they're actually asking, even when they phrase it
casually or ambiguously ("why's collection so bad," "which branches are
struggling"). You are routing their question to a fixed set of
portfolio-analysis tools. You NEVER compute numbers yourself -- you only pick
ONE tool name and its parameters EXACTLY matching the shape shown below, and
resolve any reference in the question (a pronoun, "that branch," "focus on
X") against the entity memory given.

KNOWN METRICS (use these exact names): {metrics}

KNOWN CONCEPTS (predefined business rules, use with concept_filter, use
these exact names): {concepts}

{_HIERARCHY_RULE}

{_CONTEXT_HINT_RULE}

STEP TYPES AND THEIR EXACT PARAMETERS:
{_STEP_EXAMPLES}

{_entity_memory_context(entity_memory)}

Return ONLY a JSON object with this shape, no markdown fences, no prose:
{{
  "step_type": "<one of the step types above, or null if you cannot map this to a tool>",
  "params": {{...parameters matching that step type's shape EXACTLY, same key names...}},
  "resolved_entity": {{"entity_type": "...", "entity_value": "..."}} or null,
  "needs_clarification": false,
  "clarification_question": ""
}}

If the question is genuinely ambiguous (e.g. two recently-discussed entities
could both be "him"), set needs_clarification to true and ask, rather than
guessing.

Question: {question}"""


def _fallback_turn_result(reason: str) -> dict:
    return {
        "step_type": None,
        "params": {},
        "resolved_entity": None,
        "needs_clarification": True,
        "clarification_question": reason,
    }


def _normalize_turn_response(raw_text: str) -> dict:
    """Parse and validate a raw Gemini response into a turn-result dict.
    Never raises -- any malformed/unparseable response degrades to a
    clarification request instead of crashing the caller, matching
    agents/logical_planner.py::plan_logical's own JSON-failure handling."""
    try:
        parsed = json.loads(_strip_code_fence(raw_text))
    except (json.JSONDecodeError, ValueError):
        return _fallback_turn_result(
            "I couldn't understand that as a portfolio question. Could you rephrase it?"
        )

    step_type = parsed.get("step_type")
    if step_type is not None and step_type not in STEP_REGISTRY:
        return _fallback_turn_result(
            f"I'm not sure how to answer that yet (unrecognized step '{step_type}')."
        )

    return {
        "step_type":               step_type,
        "params":                  parsed.get("params") or {},
        "resolved_entity":         parsed.get("resolved_entity"),
        "needs_clarification":     bool(parsed.get("needs_clarification", False)),
        "clarification_question":  parsed.get("clarification_question", ""),
    }


def parse_turn(question: str, entity_memory: EntityMemory) -> dict:
    """One Gemini call: route free text (initial question OR any follow-up)
    to a step_type + params from the fixed vocabulary, resolving references
    against entity_memory. Never performs aggregation/grouping itself."""
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        return _fallback_turn_result("GOOGLE_API_KEY not configured.")

    client = genai.Client(api_key=api_key)
    response = _call_gemini_with_retry(
        client, GEMINI_MODEL, build_turn_prompt(question, entity_memory),
        {"temperature": PLANNER_TEMPERATURE},
    )
    _add_token_usage(response)
    return _normalize_turn_response(response.text)


def build_narration_prompt(step_type: str, result_df_preview: str) -> str:
    """Deterministic prompt text for narrate_step -- unit-testable without
    a live call. Only ever receives a small preview (head of the already-
    computed table), never the full table -- narration, not computation."""
    return f"""You are a sharp, experienced NBFC collections & credit risk analyst
briefing a busy leader -- a Regional Business Head, Zonal Head, or
Business Unit Head -- on their own portfolio. Write 1-3 short,
plain-English sentences observing what this table shows, the way a trusted
analyst would: direct, specific, no hedging or filler, and no invented
numbers beyond what is below. Do not recommend actions unless a number
obviously implies one.

Step: {step_type}
Table preview:
{result_df_preview}"""


def _build_preview(step_type: str, result_df) -> str:
    """The text handed to narrate_step's prompt -- a small ROW preview for
    most step types, but a small AGGREGATE summary (count + total SOH) for
    concept_filter, whose result is raw loan-level rows (customer names,
    mobile numbers). A "how many non-starters" question only needs a count
    and an exposure total; there's no reason to send Gemini up to 20 rows of
    customer PII to answer it, and an arbitrary head(20) slice wouldn't even
    represent the true total if the real match count is larger anyway."""
    if step_type == "concept_filter":
        count = int(len(result_df))
        total_soh = float(result_df["SOH"].sum()) if "SOH" in result_df.columns else None
        summary = f"Matching accounts: {count}"
        if total_soh is not None:
            summary += f"\nTotal SOH (exposure) across these accounts: {total_soh:,.2f}"
        return summary
    return result_df.head(20).to_csv(index=False)


def narrate_step(step_type: str, result_df) -> str:
    """One Gemini call per step the user actually triggers. Writes a short
    narrative for an already-computed step result -- never computes
    anything itself, and only ever sees a preview of the table, matching
    the same discipline as agents/insight_generator.py for the main
    AI Query pipeline."""
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key or result_df is None or result_df.empty:
        return ""

    preview = _build_preview(step_type, result_df)
    client = genai.Client(api_key=api_key)
    response = _call_gemini_with_retry(
        client, GEMINI_MODEL, build_narration_prompt(step_type, preview),
        {"temperature": PLANNER_TEMPERATURE},
    )
    _add_token_usage(response)
    return (response.text or "").strip()


def narrate_step_stream(step_type: str, result_df) -> Iterator[str]:
    """Same call/prompt/guardrails as narrate_step, but yields text chunks
    as they arrive -- for st.write_stream() in the chat UI, so the narrative
    appears progressively instead of all at once. Callers that need the
    final full string (e.g. to store in EntityMemory) should use
    st.write_stream's own return value, which concatenates every yielded
    chunk -- never re-request or reconstruct it here."""
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key or result_df is None or result_df.empty:
        return

    preview = _build_preview(step_type, result_df)
    client = genai.Client(api_key=api_key)
    stream = client.models.generate_content_stream(
        model=GEMINI_MODEL,
        contents=build_narration_prompt(step_type, preview),
        config={"temperature": PLANNER_TEMPERATURE},
    )
    for chunk in stream:
        if chunk.text:
            yield chunk.text
