"""
Deterministic post-validation of parse_turn's routing output against facts
we already know for certain -- never trust an LLM claim that contradicts
something the registry/step vocabulary already settles.

This generalizes a principle already applied once in investigator/steps.py
(_higher_is_better ignores a conflicting LLM-supplied override for a
REGISTERED metric direction, after a live call asserted Collection% was
higher_is_better=False, which is simply false). Every bug found in this
layer so far has the same shape: the model re-derives structured parameters
(scope, grain, direction) from a text description each turn, and
occasionally gets it wrong in a way that produces a table that LOOKS
answerable but silently isn't what was asked. A prompt rewrite fixes the
ONE discovered case; this module catches the general CLASS after the fact,
so a new, not-yet-seen phrasing that trips the same failure mode gets
caught here too, not just the exact wording already patched into the prompt.

validate_and_correct is the single entry point: called on parse_turn's
output before executing. It either returns the turn unchanged, returns a
corrected turn (spliced from a suitable entity's OWN stored params -- see
investigator/state.py's EntityRecord.params), or degrades to a
clarification request. It never silently executes something it has
concrete reason to believe is wrong.

df_curr (optional, defaults to None) lets this module check the turn
against the ACTUAL data before executing, not just against static
registry/step-vocabulary facts -- currently used only for
_ambiguous_executive_clarification's name-collision check. None/empty is a
safe no-op for every check here, never a crash, since this validation is a
best-effort safety net layered on top of what steps.py already guards for
on its own (see _executive_row's iloc[0] fallback).
"""
from __future__ import annotations

from investigator.state import EntityMemory
from investigator.steps import distinct_executive_matches
from registry.semantic_model import resolve_dimension

# Which metric columns each step type can actually produce -- known
# statically from each step's own implementation (investigator/steps.py),
# not guessed. dimension_breakdown's generic aggregator never computes
# Strike Rate% (see _GENERIC_DIMENSION_METRICS); underperformer_quartile/
# bottom_n_per_group both run on the executive scorecard, which has all
# four. Both spellings (with/without a space) are listed since callers use
# either depending on which grain they're thinking in.
_STEP_PRODUCIBLE_METRICS: dict[str, set[str]] = {
    "dimension_breakdown": {"Collection%", "NPA%", "Delinquency%", "Hard Bucket%"},
    "underperformer_quartile": {"Collection %", "NPA %", "Delinquency %", "Strike Rate %", "Hard Bucket %"},
    "bottom_n_per_group": {"Collection %", "NPA %", "Delinquency %", "Strike Rate %", "Hard Bucket %"},
    "executive_roster": {"Collection %", "NPA %", "Delinquency %", "Strike Rate %", "Hard Bucket %"},
}

# Entities produced by underperformer_quartile/bottom_n_per_group -- see
# investigator/llm.py's _EXECUTIVE_GRAIN_ENTITY_TYPES for the identical set,
# duplicated here rather than imported to keep this module's only
# dependency on the step layer, not the LLM-facing prompt layer.
_EXECUTIVE_GRAIN_ENTITY_TYPES = {"executive_group", "branch_executives"}


def _normalize(metric: str) -> str:
    return metric.replace(" %", "%")


def _metric_producible(step_type: str, metric: str) -> bool:
    producible = _STEP_PRODUCIBLE_METRICS.get(step_type)
    if producible is None:
        return True  # step type has no known column restriction to check
    return _normalize(metric) in {_normalize(m) for m in producible}


def _fallback_clarification(metric: str) -> dict:
    return {
        "step_type": None,
        "params": {},
        "resolved_entity": None,
        "needs_clarification": True,
        "clarification_question": (
            f"I couldn't find {metric} on the table I'd need to build for that. "
            f"Could you say which branch, region, or executive you mean?"
        ),
    }


def _redirect_invalid_executive_grouping(turn: dict) -> dict | None:
    """bottom_n_per_group with group_col="executive" is never valid --
    resolve_dimension has no "executive" alias, and executive is what's
    being ranked, never what's grouped by. Caught live: a plain "bottom N
    executives [by METRIC]" question with no group named routed here
    anyway. That question is a flat ranked list, the exact shape
    executive_roster(worst_first=True) exists for -- redirect there
    deterministically rather than executing a step known to be broken for
    this group_col, or leaving the fix to a prompt wording that already
    says not to do this and got ignored once."""
    step_type = turn.get("step_type")
    params = turn.get("params") or {}
    if step_type != "bottom_n_per_group" or str(params.get("group_col")).strip().lower() != "executive":
        return None
    return {
        "step_type": "executive_roster",
        "params": {
            "metric": params.get("metric"), "n": params.get("n"),
            "higher_is_better": params.get("higher_is_better"), "worst_first": True,
        },
        "resolved_entity": turn.get("resolved_entity"),
        "needs_clarification": False,
        "clarification_question": "",
    }


def _ambiguous_executive_clarification(turn: dict, df_curr) -> dict | None:
    """entity_summary("executive", name[, unit]) resolves the name by
    `contains` substring match (investigator/steps.py::_scope_executive_rows
    -- names are free-typed, not a controlled vocabulary). Found via audit,
    not a live report: when that match resolves to MORE THAN ONE distinct
    (MNT NAME, Unit) pair, _executive_row used to silently return
    compute_executive_scorecard's iloc[0] -- best Collection% first, per
    that function's own sort -- with zero signal a second person even
    matched. Concretely: two executives both named "Rahul," one genuinely
    struggling and one doing fine -- asking "why is Rahul struggling" would
    silently return the WELL-PERFORMING Rahul's numbers every time, since
    the struggling one always loses that iloc[0] tiebreak. A real
    reputational-risk bug for a tool whose answers can affect how a real
    person's performance is judged -- must ask which Rahul was meant,
    listing every distinct match, rather than guessing.

    df_curr may be None/empty (e.g. validated before data is ready) -- this
    check is a best-effort safety net, not a hard requirement, and silently
    no-ops in that case; _executive_row's own iloc[0] fallback still applies
    if this genuinely couldn't be checked."""
    if df_curr is None or (hasattr(df_curr, "empty") and df_curr.empty):
        return None
    if turn.get("step_type") != "entity_summary":
        return None
    params = turn.get("params") or {}
    if params.get("entity_type") != "executive":
        return None

    entity_value = params.get("entity_value")
    if isinstance(entity_value, (list, tuple)):
        name = entity_value[0] if len(entity_value) > 0 else None
        unit = entity_value[1] if len(entity_value) > 1 else None
    else:
        name, unit = entity_value, None
    if not name:
        return None

    matches = distinct_executive_matches(df_curr, name, unit)
    if len(matches) <= 1:
        return None

    listed = ", ".join(f"{n} ({u})" if u else n for n, u in matches)
    return {
        "step_type": None,
        "params": {},
        "resolved_entity": None,
        "needs_clarification": True,
        "clarification_question": (
            f"There's more than one executive matching \"{name}\": {listed}. "
            f"Which one did you mean?"
        ),
    }


def _source_found(ref: dict, entity_memory: EntityMemory) -> bool:
    etype, evalue = ref.get("entity_type"), ref.get("entity_value")
    if etype == "executive" and isinstance(evalue, (list, tuple)) and len(evalue) == 2:
        return True                                  # looked up in the file itself, not in memory
    if etype == "executive" and isinstance(evalue, list):
        evalue = tuple(evalue)
    return entity_memory.get(etype, evalue) is not None


def _place_named_in(ref: dict, df_curr) -> tuple[str, str] | None:
    """The branch / region / zone a missing source's own name points at
    ("branch_executives" "AKLUJ:Collection %" -> branch AKLUJ), checked
    against the file. None when nothing (or more than one level) matches."""
    if df_curr is None or getattr(df_curr, "empty", True):
        return None
    raw = str(ref.get("entity_value") or "")
    pieces = {p.strip().upper() for part in raw.split(":") for p in part.split(" within ") if p.strip()}
    for level in ("branch", "region", "zone"):
        col = resolve_dimension(level)[0]
        if col not in df_curr.columns:
            continue
        names = {str(v).strip().upper(): str(v).strip() for v in df_curr[col].dropna().unique()}
        hits = [names[p] for p in pieces if p in names]
        if len(hits) == 1:
            return level, hits[0]
    return None


def _redirect_missing_source(turn: dict, entity_memory: EntityMemory, df_curr) -> dict | None:
    """non_paying_customers pointed at an earlier result that isn't in this
    chat (live: "customers not paying in Akluj" as a first question named a
    "branch_executives" table never built). Run on the branch / region that
    reference names instead. When no place can be read from it, the turn is
    left alone and the run refuses with a clear message
    (ui/tabs/investigator.py::_resolve_source_tables), never a false zero."""
    if turn.get("step_type") != "non_paying_customers":
        return None
    params = turn.get("params") or {}
    missing = [r for r in params.get("source_entities") or [] if not _source_found(r, entity_memory)]
    if not missing:
        return None
    place = _place_named_in(missing[0], df_curr)
    if place is None:
        return None
    corrected = {k: v for k, v in params.items() if k != "source_entities"}
    corrected.update(scope_col=place[0], scope_value=place[1])
    return {**turn, "params": corrected, "needs_clarification": False, "clarification_question": ""}


# Words that mean "any EMI overdue" (Delinquency %), not NPA. A model that
# reads "most delinquent loans" as NPA % ranks a different, smaller set.
_DELINQUENCY_WORDS = ("delinquen", "overdue", "behind on")


def _delinquency_not_npa(turn: dict, question: str | None) -> dict | None:
    params = turn.get("params") or {}
    metric = params.get("metric")
    if not question or not metric or _normalize(str(metric)) != "NPA%":
        return None
    q = question.lower()
    if "npa" in q or not any(w in q for w in _DELINQUENCY_WORDS):
        return None
    new_metric = "Delinquency %" if " %" in str(metric) else "Delinquency%"
    return {**turn, "params": {**params, "metric": new_metric}}


def validate_and_correct(turn: dict, entity_memory: EntityMemory, df_curr=None, question: str | None = None) -> dict:
    """Returns turn unchanged if nothing is wrong, a corrected turn spliced
    from a suitable entity's own stored params, or a clarification request.
    Never returns a turn known to be unable to satisfy what was asked.
    question (the user's own words) lets a wording check correct the metric."""
    ambiguous = _ambiguous_executive_clarification(turn, df_curr)
    if ambiguous is not None:
        return ambiguous

    redirected = _redirect_invalid_executive_grouping(turn)
    if redirected is not None:
        return redirected

    turn = _redirect_missing_source(turn, entity_memory, df_curr) or turn
    turn = _delinquency_not_npa(turn, question) or turn

    step_type = turn.get("step_type")
    params = turn.get("params") or {}
    metric = params.get("metric")

    if not metric or step_type is None or _metric_producible(step_type, metric):
        return turn

    # The requested metric can never be a column of this step_type's own
    # output -- try to correct by reusing a recent executive-grain entity's
    # OWN stored params (same scope, new metric) rather than silently
    # running a step that can't answer the question.
    for record in entity_memory.all():
        if record.entity_type not in _EXECUTIVE_GRAIN_ENTITY_TYPES:
            continue
        if record.result_df is None or metric not in record.result_df.columns:
            continue
        if not record.params:
            continue
        corrected_params = {**record.params, "metric": metric}
        corrected_params.pop("dimension", None)
        return {
            "step_type": "underperformer_quartile",
            "params": corrected_params,
            "resolved_entity": {"entity_type": record.entity_type, "entity_value": record.entity_value},
            "needs_clarification": False,
            "clarification_question": "",
        }

    return _fallback_clarification(metric)
