"""
Investigator tab: a bounded, human-gated root-cause drill-down assistant,
presented as a chat thread (st.chat_message/st.chat_input) rather than a
form -- chosen over a React rewrite because the actual complaint it fixes
("feels like a form, not a conversation") is fully addressable inside
Streamlit's own chat primitives, at a fraction of the cost/risk of a new
frontend + backend. See the design discussion for why a full client-server
rewrite is deferred until auth/multi-tenant access control forces one
anyway (that decision would need a real backend either way, so it's the
natural point to bundle both).

investigator.llm.parse_turn resolves free text (initial question AND every
follow-up) to a step_type + params from investigator.steps.STEP_REGISTRY's
fixed vocabulary -> the step runs as one deterministic pandas pass -> the
result is stored in an EntityMemory (NOT a linear queue -- see
investigator/state.py's module docstring for why) so a side-question about
an unrelated entity can't corrupt an in-progress thread -> every step's full
table is downloadable as Excel for audit, via the same _dl_btn/_excel_bytes
helper every other tab in this app already uses.

The chat transcript (investigator_turns) is a thin, replayable log on top
of EntityMemory: each assistant turn stores only an entity_key, and
re-renders by reading that entity's CURRENT stored result on every rerun --
never by re-running parse_turn or re-streaming narration for old turns.
Only the newest question triggers a real Gemini call and a live stream.

"Analyse further" controls render only on the latest assistant turn (a
deliberate simplification, not an oversight): revisiting an OLDER entity to
drill further is done via a plain follow-up in the chat box ("back to the
branch, drill into NPA% instead"), which parse_turn already resolves
correctly against EntityMemory -- confirmed live. This avoids re-rendering
drill controls (and their keys) on every historical message on every rerun.

Does not touch graph.py/compiler/registry/agents' AI Query pipeline -- this
is an entirely separate, additive module.
"""
import os
import time
import uuid

import pandas as pd
import streamlit as st

from investigator.branch_recipients import get_branch_recipients
from investigator.email_draft import draft_priority_email
from investigator.guardrails import validate_and_correct
from investigator.llm import narrate_step_stream, parse_turn
from investigator.state import EntityMemory
from investigator.steps import (
    METRIC_DIRECTION,
    METRIC_DRILL_FILTER,
    PRIORITY_MENU_CATEGORIES,
    STEP_REGISTRY,
    resolve_executive_identity,
    summarize_worst_loans,
)
from investigator.suggestions import suggest_mechanism_steps
from registry.semantic_model import resolve_dimension
from ui.components import _dl_btn

_NEW_CHAT_TITLE = "New chat"
_TITLE_MAX_CHARS = 40
_USER_AVATAR = ":material/person:"
_ASSISTANT_AVATAR = ":material/query_stats:"
_HELP_TRIGGER = "what can you help me with"
_HELP_RESPONSE = (
    "I'm the Investigator. Ask me why a number moved, such as Collection%, "
    "NPA%, or Hard Bucket%, and I'll drill down one level at a time: "
    "BU → Zone → Region → Branch → Executive → Customer, sorted worst first. "
    "Click **Analyse further** at any point to go one level deeper, or just "
    "type a follow-up (\"meanwhile, how is Rahul in Pune doing\" works too, "
    "since I keep track of everyone we've discussed this session, not just "
    "the last one). Every table is the exact underlying data, downloadable "
    "for audit. I only route your question to the right calculation; I "
    "never compute the numbers myself."
)

# Deliberately generic -- no branch/region/executive name is hardcoded here.
# A hardcoded example like "why is Mahad branch underperforming" breaks the
# moment a real user's data has no branch by that name (confirmed directly:
# a live user's data had no "Mahad" at all). Every suggestion below must
# resolve correctly regardless of what's actually in the uploaded file.
_SUGGESTIONS = {
    "What can you help me with?": _HELP_TRIGGER,
    "Summary of my regions": "how are my regions performing",
    "Which branch needs attention?": "which branch is underperforming and why",
    "Who's dragging down collection%?": "who are the bottom performers by collection percent",
}

# smart_alerts.py::run_all_alerts's own alert "title" -> (step_type, params)
# to run it with -- lets the proactive opening message route straight to
# the right Investigator step with ZERO Gemini calls (same "known answer,
# skip the LLM" precedent as _HELP_TRIGGER above), since we already know
# exactly what a clicked alert means. Most map to concept_filter (a
# registered CONCEPTS name); "High Arrears: Loan at Risk" maps to the
# dedicated high_arrears_at_risk step instead, since that ratio can't be
# expressed as a simple CONCEPTS condition (see that step's own docstring).
_ALERT_TITLE_TO_STEP: dict[str, tuple[str, dict]] = {
    "Non Starters": ("concept_filter", {"concept": "non_starter"}),
    "Insurance-Driven Delinquency": ("concept_filter", {"concept": "insurance_driven_delinquency"}),
    "Easy Settlements": ("concept_filter", {"concept": "easy_settlement"}),
    "Recent Advances at Risk": ("concept_filter", {"concept": "recent_advance_high_bucket"}),
    "Co-lending Loans at Risk": ("concept_filter", {"concept": "colending_at_risk"}),
    "High Arrears: Loan at Risk": ("high_arrears_at_risk", {}),
}

# Mirrors report_agent/sections/risk_flags.py::compute_risk_flags's own
# severity ranking EXACTLY (same dict, same tie-break) -- duplicated
# rather than imported because that function re-runs run_all_alerts
# itself, and app.py has ALREADY computed and cached this exact alerts
# list once per upload (reused by the AI Query tab too); recomputing it a
# second time here would throw away that cache for no reason. If this
# ranking is ever retuned, risk_flags.py's copy must be updated too.
_ALERT_SEVERITY_RANK = {"critical": 0, "high": 1, "medium": 2}


def _top_active_alerts(alerts_curr: list | None, limit: int = 3) -> list[dict]:
    if not alerts_curr:
        return []
    active = [a for a in alerts_curr if a.get("count", 0) > 0]
    return sorted(
        active, key=lambda a: (_ALERT_SEVERITY_RANK.get(a.get("severity"), 9), -a.get("count", 0)),
    )[:limit]

# Branch-level scorecard metric names (analysis/portfolio_intelligence.py's
# _branch_aggregates) -> the equivalent executive-level scorecard column
# name (analysis/executive_scorecard.py::compute_executive_scorecard) --
# needed because "drill into the executives behind this branch's worst
# metric" hands that metric name to underperformer_quartile, which operates
# on the executive scorecard's own column names, not the branch scorecard's.
_BRANCH_TO_EXEC_METRIC = {
    "Collection%":   "Collection %",
    "NPA%":          "NPA %",
    "Strike%":       "Strike Rate %",
    "Hard Bucket%":  "Hard Bucket %",
}


def _get_threads() -> dict:
    """Multiple switchable conversation threads within THIS session --
    ChatGPT-style, but session-scoped only: this app has no login and no
    database (confirmed elsewhere), so there is no such thing as history
    surviving a page refresh, a new upload, or a different browser. A
    real cross-session history needs actual backend storage; this is the
    honest version of that feature given the current architecture."""
    if "investigator_threads" not in st.session_state:
        st.session_state["investigator_threads"] = {}
    return st.session_state["investigator_threads"]


def _new_thread() -> str:
    thread_id = uuid.uuid4().hex[:8]
    _get_threads()[thread_id] = {
        "title": _NEW_CHAT_TITLE, "turns": [], "memory": EntityMemory(), "created_at": time.time(),
    }
    return thread_id


def _get_active_thread_id() -> str:
    threads = _get_threads()
    active = st.session_state.get("investigator_active_thread")
    if active not in threads:
        active = _new_thread() if not threads else next(iter(threads))
        st.session_state["investigator_active_thread"] = active
    return active


def _get_memory() -> EntityMemory:
    return _get_threads()[_get_active_thread_id()]["memory"]


def _get_turns() -> list[dict]:
    return _get_threads()[_get_active_thread_id()]["turns"]


def _as_entity_value(entity_type: str, raw_value):
    """Executive entity values are (name, unit) pairs -- JSON/UI round-trips
    them as a 2-element list; EntityMemory keys need a hashable tuple."""
    if entity_type == "executive" and isinstance(raw_value, (list, tuple)):
        return tuple(raw_value)
    return raw_value


def _sanitize_key(value) -> str:
    return "".join(c if c.isalnum() else "_" for c in str(value))


def _turn_dl_key(turn_index: int, entity_type: str, entity_value) -> str:
    """Widget key for one CHAT TURN's rendering -- combines the turn's
    position in the (append-only, never-reordered) `turns` log with the
    entity's identity.

    Neither piece alone is safe: entity-identity-only collided in
    production (a vague follow-up like "whats the issue?" resolved back to
    an entity already asked about earlier in the SAME conversation, so two
    chat bubbles both keyed purely on "region_CS_Nagar" -> Streamlit's
    duplicate-widget-key crash). Position-only was already tried and
    rejected earlier (see the git history / earlier fix note): keying on
    memory.all()'s list position broke because THAT list reorders by
    recency every time an entity is re-touched, so a stale click could land
    on a different entity's widget under a reused position-based key.

    `turns` (unlike memory.all()) is strictly append-only and never
    reorders or removes entries, so a given index always refers to the same
    semantic turn for the life of that list -- combining it with entity
    identity is unique AND stable."""
    return f"inv_t{turn_index}_{entity_type}_{_sanitize_key(entity_value)}"


def _resolve_source_tables(params: dict, memory: EntityMemory) -> list[pd.DataFrame]:
    """A step consuming a previously-computed entity table (intersect_entities,
    non_paying_customers) names it via params["source_entities"] (a list of
    {"entity_type", "entity_value"}) -- resolved here against EntityMemory,
    never by re-sending the table through the LLM.

    entity_type "executive" is a special case: a SINGLE executive's
    entity_summary result has no MNT NAME/Unit columns at all (it's a
    Metric/Value table), which is not the shape non_paying_customers (and
    similar consumers) need -- a table WITH those columns, to merge
    against. Rather than requiring a pre-existing multi-row
    executive_group/branch_executives table in memory (which forced the
    LLM to substitute a stale, unrelated one when asked about a single
    already-focused executive -- confirmed live), synthesize a one-row
    lookup table directly from that executive's own (by now canonicalized,
    see _process_new_question) identity."""
    refs = params.get("source_entities") or []
    tables = []
    for ref in refs:
        etype = ref.get("entity_type")
        evalue = _as_entity_value(etype, ref.get("entity_value"))
        if etype == "executive" and isinstance(evalue, tuple) and len(evalue) == 2:
            tables.append(pd.DataFrame([{"MNT NAME": evalue[0], "Unit": evalue[1]}]))
            continue
        record = memory.get(etype, evalue)
        if record is not None and record.result_df is not None and not record.result_df.empty:
            tables.append(record.result_df)
    return tables


def _derive_entity_key(step_type: str, params: dict, resolved_entity: dict | None) -> tuple[str, object]:
    """The (entity_type, entity_value) EntityMemory key for a free-text turn's
    result -- derived from the STEP TYPE's own canonical shape, never blindly
    trusted from the LLM's "resolved_entity" guess. A live-Gemini check
    caught a real bug from doing the latter: customer_loan_book's
    resolved_entity pointed at a leftover "executive" reference from earlier
    context, which would have silently overwritten that executive's stored
    result instead of creating a new "customer" entry. Every step type here
    has an unambiguous entity shape of its own; only entity_summary's is
    genuinely LLM-supplied (that's the whole point of that step)."""
    if step_type == "dimension_breakdown":
        dimension = params.get("dimension", "unknown")
        scope = params.get("scope_value")
        # Scoped and unscoped breakdowns of the SAME dimension are different
        # results (e.g. "all branches" vs "branches within CS Nagar") -- a
        # bare dimension name as the key would let one silently overwrite
        # the other.
        return "dimension_breakdown", f"{dimension} within {scope}" if scope else dimension
    if step_type == "roll_rate_summary":
        scope = params.get("scope_value") or "portfolio"
        return "roll_rate_summary", scope
    if step_type == "roll_rate_by_dimension":
        return "roll_rate_by_dimension", params.get("dimension")
    if step_type == "vintage_summary":
        scope = params.get("scope_value") or "portfolio"
        return "vintage_summary", scope
    if step_type == "demand_summary":
        scope = params.get("scope_value") or "portfolio"
        return "demand_summary", f"{params.get('dimension')}:{scope}"
    if step_type == "new_advances_summary":
        scope = params.get("scope_value") or "portfolio"
        return "new_advances_summary", scope
    if step_type == "new_advances_by_dimension":
        scope = params.get("scope_value") or "portfolio"
        return "new_advances_by_dimension", f"{params.get('dimension')}:{scope}"
    if step_type == "new_advances_trend":
        scope = params.get("scope_value") or "portfolio"
        return "new_advances_trend", f"{params.get('granularity', 'Monthly')}:{scope}"
    if step_type == "product_analysis":
        scope = params.get("scope_value") or "portfolio"
        return "product_analysis", f"{params.get('axis')}:{scope}"
    if step_type == "top_accounts":
        scope = params.get("scope_value") or "portfolio"
        return "top_accounts", scope
    if step_type in ("fleet_exposure", "repossession_list", "good_customers"):
        scope = params.get("scope_value") or "portfolio"
        return step_type, scope
    if step_type == "entity_summary":
        return params["entity_type"], _as_entity_value(params["entity_type"], params["entity_value"])
    if step_type == "customer_loan_book":
        return "customer", params["cust_mob_no"]
    if step_type == "concept_filter":
        scope = params.get("scope_value")
        return "concept_filter", f"{params.get('concept')} within {scope}" if scope else params.get("concept")
    if step_type == "priority_accounts":
        scope = params.get("scope_value") or "portfolio"
        return "priority_accounts", scope
    if step_type in ("priority_menu", "top_closing_arrears", "high_arrears_at_risk", "fleet_defaulters"):
        scope = params.get("scope_value") or "portfolio"
        return step_type, scope
    if step_type == "worst_loans_by_metric":
        scope = params.get("scope_value") or "portfolio"
        scope = tuple(scope) if isinstance(scope, list) else scope
        return "worst_loans_by_metric", f"{params.get('metric')}:{scope}"
    if step_type == "executive_roster":
        scope = params.get("scope_value") or "portfolio"
        return "executive_group", f"executive_roster:{scope}"
    if step_type in ("underperformer_quartile", "bottom_n_per_group"):
        scope = params.get("scope_value") or params.get("group_col") or "portfolio"
        return "executive_group", f"{step_type}:{params.get('metric')}:{scope}"
    if step_type in ("intersect_entities", "non_paying_customers", "priority_by_executive", "concept_breakdown"):
        sources = params.get("source_entities") or []
        label = ",".join(f"{s.get('entity_type')}:{s.get('entity_value')}" for s in sources) or "unscoped"
        if step_type == "concept_breakdown":
            label = f"{label}:{params.get('dimension')}"
        return step_type, label
    # Fallback for any future step type not special-cased above.
    resolved = resolved_entity or {}
    return resolved.get("entity_type", step_type), resolved.get("entity_value", step_type)


def _metric_focus_for(step_type: str, params: dict) -> str | None:
    """What EntityMemory's metric_focus field should hold for this turn --
    the ranking/filter metric for most steps, but the bare dimension name
    (e.g. "branch") for dimension_breakdown, since that step has no
    "metric" param and llm.py::_entity_memory_context needs the bare
    dimension name to resolve which column to list names from (the
    entity_value itself may carry a " within <scope>" suffix now, so it
    can't be parsed back into a dimension alias directly)."""
    if step_type == "dimension_breakdown":
        return params.get("dimension")
    return params.get("metric")


def _execute_step(
    step_type: str, params: dict, df_curr: pd.DataFrame, df_prev: pd.DataFrame,
    memory: EntityMemory, as_of=None,
) -> pd.DataFrame:
    """Dispatch to one STEP_REGISTRY function, supplying df_curr/df_prev and
    resolving any prior-result references -- the LLM never touches the
    DataFrame itself, only names entities/metrics (see llm.py's docstring).
    as_of is the report's OWN reporting month (app.py's curr_month), passed
    straight through to any step that anchors a relative-date window to it
    (vintage_summary, new_advances_summary, new_advances_by_dimension,
    new_advances_trend, product_analysis, repossession_list) -- never
    wall-clock today, same convention every other as_of-taking function in
    this codebase follows (see vintage_summary's own docstring)."""
    fn = STEP_REGISTRY[step_type]

    if step_type == "dimension_breakdown":
        return fn(
            df_curr, params["dimension"], df_prev=df_prev,
            scope_col=params.get("scope_col"), scope_value=params.get("scope_value"),
            metric=params.get("metric"), higher_is_better=params.get("higher_is_better"),
        )

    if step_type == "roll_rate_summary":
        return fn(
            df_curr, df_prev,
            scope_col=params.get("scope_col"), scope_value=params.get("scope_value"),
        )

    if step_type == "roll_rate_by_dimension":
        return fn(df_curr, df_prev, params["dimension"])

    if step_type == "vintage_summary":
        return fn(
            df_curr, scope_col=params.get("scope_col"), scope_value=params.get("scope_value"),
            as_of=as_of,
        )

    if step_type == "demand_summary":
        return fn(
            df_curr, params["dimension"],
            scope_col=params.get("scope_col"), scope_value=params.get("scope_value"),
        )

    if step_type == "new_advances_summary":
        return fn(
            df_curr, scope_col=params.get("scope_col"), scope_value=params.get("scope_value"),
            as_of=as_of,
        )

    if step_type == "new_advances_by_dimension":
        return fn(
            df_curr, params["dimension"],
            scope_col=params.get("scope_col"), scope_value=params.get("scope_value"),
            as_of=as_of,
        )

    if step_type == "new_advances_trend":
        return fn(
            df_curr, months=params.get("months"), granularity=params.get("granularity", "Monthly"),
            scope_col=params.get("scope_col"), scope_value=params.get("scope_value"), as_of=as_of,
        )

    if step_type == "product_analysis":
        return fn(
            df_curr, params["axis"],
            scope_col=params.get("scope_col"), scope_value=params.get("scope_value"), as_of=as_of,
        )

    if step_type == "top_accounts":
        return fn(
            df_curr, n=params.get("n", 20),
            scope_col=params.get("scope_col"), scope_value=params.get("scope_value"),
        )

    if step_type == "fleet_exposure":
        return fn(df_curr, scope_col=params.get("scope_col"), scope_value=params.get("scope_value"))

    if step_type == "repossession_list":
        return fn(
            df_curr, scope_col=params.get("scope_col"), scope_value=params.get("scope_value"), as_of=as_of,
        )

    if step_type == "good_customers":
        return fn(df_curr, scope_col=params.get("scope_col"), scope_value=params.get("scope_value"))

    if step_type == "entity_summary":
        entity_type = params["entity_type"]
        entity_value = _as_entity_value(entity_type, params["entity_value"])
        return fn(entity_type, entity_value, df_curr, df_prev)

    if step_type == "underperformer_quartile":
        return fn(
            df_curr, params["metric"],
            higher_is_better=params.get("higher_is_better"),
            scope_col=params.get("scope_col"),
            scope_value=params.get("scope_value"),
        )

    if step_type == "bottom_n_per_group":
        return fn(
            df_curr, params["metric"], int(params.get("n", 2)), params["group_col"],
            higher_is_better=params.get("higher_is_better"),
        )

    if step_type == "intersect_entities":
        tables = _resolve_source_tables(params, memory)
        return fn(tables, key_cols=params.get("key_cols"), min_appearances=params.get("min_appearances"))

    if step_type == "non_paying_customers":
        tables = _resolve_source_tables(params, memory)
        entities = tables[0] if tables else pd.DataFrame()
        return fn(df_curr, entities, key_cols=params.get("key_cols"), all_columns=bool(params.get("all_columns")))

    if step_type == "customer_loan_book":
        return fn(df_curr, params["cust_mob_no"], all_columns=bool(params.get("all_columns")))

    if step_type == "concept_filter":
        return fn(
            df_curr, params["concept"],
            scope_col=params.get("scope_col"), scope_value=params.get("scope_value"),
        )

    if step_type == "priority_accounts":
        return fn(
            df_curr, scope_col=params.get("scope_col"), scope_value=params.get("scope_value"),
            all_columns=bool(params.get("all_columns")),
        )

    if step_type in ("priority_menu", "high_arrears_at_risk", "fleet_defaulters"):
        return fn(df_curr, scope_col=params.get("scope_col"), scope_value=params.get("scope_value"))

    if step_type == "top_closing_arrears":
        return fn(
            df_curr, top_n=params.get("top_n", 20),
            scope_col=params.get("scope_col"), scope_value=params.get("scope_value"),
        )

    if step_type == "worst_loans_by_metric":
        scope_value = params.get("scope_value")
        if isinstance(scope_value, list):
            scope_value = tuple(scope_value)
        return fn(
            df_curr, params["metric"], scope_col=params.get("scope_col"), scope_value=scope_value,
            all_columns=bool(params.get("all_columns")),
        )

    if step_type == "executive_roster":
        return fn(
            df_curr, scope_col=params.get("scope_col"), scope_value=params.get("scope_value"),
            n=params.get("n"), metric=params.get("metric"),
            higher_is_better=params.get("higher_is_better"), worst_first=bool(params.get("worst_first")),
        )

    if step_type == "priority_by_executive":
        tables = _resolve_source_tables(params, memory)
        priority_df = tables[0] if tables else pd.DataFrame()
        return fn(df_curr, priority_df)

    if step_type == "concept_breakdown":
        tables = _resolve_source_tables(params, memory)
        filtered_df = tables[0] if tables else pd.DataFrame()
        return fn(filtered_df, params["dimension"])

    raise ValueError(f"No dispatcher wired for step_type '{step_type}'")


def _worst_metric(entity_summary_df: pd.DataFrame) -> str | None:
    """Which metric in an entity_summary result looks worst right now --
    used only to PRE-FILL the "Analyse further" suggestion; the user still
    has to click for anything to actually run."""
    if entity_summary_df is None or entity_summary_df.empty or "Delta" not in entity_summary_df.columns:
        return None
    candidates = entity_summary_df[entity_summary_df["Delta"].notna()]
    if candidates.empty:
        return None

    def _badness(row) -> float:
        hib = row["Higher Is Better"]
        if hib is None or (isinstance(hib, float) and pd.isna(hib)):
            return float("-inf")
        # Worse = delta moved in the direction that's bad for THIS metric's
        # own polarity, not just "the biggest raw delta."
        return -row["Delta"] if hib else row["Delta"]

    worst_idx = candidates.apply(_badness, axis=1).idxmax()
    return candidates.loc[worst_idx, "Metric"]


def _append_drill_turn(question_text: str, entity_type: str, entity_value) -> None:
    """Every "Analyse further" button click is a real conversational turn,
    not a silent table swap -- appends a synthetic user message (the plain-
    English equivalent of what was just clicked) before the assistant's
    result, so the transcript reads as a two-way conversation instead of a
    new table appearing with no visible request behind it (confirmed
    directly: without this it read as a "one way convo")."""
    turns = _get_turns()
    turns.append({"role": "user", "text": question_text})
    turns.append({"role": "assistant", "entity_type": entity_type, "entity_value": entity_value})


def _render_mechanism_suggestions(
    record, df_curr: pd.DataFrame, df_prev: pd.DataFrame | None, as_of,
    memory: EntityMemory, dl_key: str,
) -> None:
    """Deterministic "check mechanism before location" suggestions
    (investigator/suggestions.py's plain Python lookup table, never an LLM
    improvising this) -- rendered ABOVE the plain org-hierarchy drill below,
    since a sharp analyst checks WHY a number moved mechanically (roll-rate/
    vintage/demand composition) before asking WHO is responsible. Still
    fully human-gated: every suggestion here is a button the user must
    click, same as every other drill in this file -- nothing runs on its
    own, and every suggested step is still a discrete, downloadable,
    human-approved table like everything else here. Only called for
    entity_summary at branch/region/zone/BU grain (see call sites below) --
    an executive is already "location," so this doesn't apply there."""
    suggestions = suggest_mechanism_steps(_worst_metric(record.result_df))
    if not suggestions:
        return

    scope_col, scope_value = record.entity_type, record.entity_value
    # ONE-LEVEL-AT-A-TIME, same convention the metric-drill buttons below
    # already use: demand_summary needs a grain to return rows at (a
    # branch's own executives, or a region/zone/BU's own branches) --
    # roll_rate_summary/vintage_summary don't, they explain the CURRENT
    # entity's own mechanism, never decompose further into sub-entities.
    demand_dimension = "executive" if record.entity_type == "branch" else "branch"

    with st.container(border=True):
        st.markdown("**Before drilling into executives, check mechanism:**")
        for suggestion in suggestions:
            step_type = suggestion["step_type"]
            st.caption(suggestion["reason"])
            if step_type == "roll_rate_summary" and (df_prev is None or df_prev.empty):
                st.caption("(Needs a previous month's file uploaded -- not available right now.)")
                continue
            if st.button(suggestion["label"], key=f"mech_{step_type}_{dl_key}"):
                if step_type == "roll_rate_summary":
                    drill_df = STEP_REGISTRY["roll_rate_summary"](
                        df_curr, df_prev, scope_col=scope_col, scope_value=scope_value,
                    )
                    new_type, new_value = "roll_rate_summary", scope_value
                    params = {"scope_col": scope_col, "scope_value": scope_value}
                elif step_type == "vintage_summary":
                    drill_df = STEP_REGISTRY["vintage_summary"](
                        df_curr, scope_col=scope_col, scope_value=scope_value, as_of=as_of,
                    )
                    new_type, new_value = "vintage_summary", scope_value
                    params = {"scope_col": scope_col, "scope_value": scope_value}
                else:  # demand_summary
                    drill_df = STEP_REGISTRY["demand_summary"](
                        df_curr, demand_dimension, scope_col=scope_col, scope_value=scope_value,
                    )
                    new_type, new_value = "demand_summary", f"{demand_dimension}:{scope_value}"
                    params = {"dimension": demand_dimension, "scope_col": scope_col, "scope_value": scope_value}
                memory.touch(new_type, new_value, result_df=drill_df, step_type=step_type, params=params)
                _append_drill_turn(f"{suggestion['label'].lower()} for {scope_value}", new_type, new_value)
                st.rerun()


def _render_analyse_further(
    record, df_curr: pd.DataFrame, memory: EntityMemory, dl_key: str,
    df_prev: pd.DataFrame | None = None, as_of=None,
) -> None:
    result_df = record.result_df

    if record.step_type == "priority_menu" and "Category" in result_df.columns:
        # The menu shows COUNTS only -- one row per category runs the
        # underlying, already-built step (concept_filter/fleet_defaulters/
        # top_closing_arrears, never a second filter implementation),
        # scoped the SAME way the menu itself was (record.params' own
        # scope_col/scope_value). How many to pull is EDITABLE per
        # category (a number input next to its button, defaulting to that
        # row's own Shown, bounded to [1, Count]) -- a real ask: the same
        # count that's right for "analyse further" isn't necessarily right
        # for "how many loans go in the email to the branch manager."
        category_lookup = {c["label"]: c for c in PRIORITY_MENU_CATEGORIES}
        scope_col = (record.params or {}).get("scope_col")
        scope_value = (record.params or {}).get("scope_value")
        is_branch_scoped = scope_col == "branch" and scope_value
        for _, row in result_df.iterrows():
            label = str(row["Category"])
            default_shown = int(row["Shown"])
            total_count = int(row["Count"])
            category = category_lookup.get(label)
            if category is None:
                continue
            # width="stretch" is number_input's default -- inside a
            # horizontal container that made it greedily fill all leftover
            # space as a giant full-width bar next to a comparatively tiny
            # button (the reported UI bug). Pin it to a fixed pixel width
            # instead so it reads as a small counter, not a dominant block.
            with st.container(horizontal=True, vertical_alignment="center"):
                n = st.number_input(
                    f"How many {label}?", min_value=1, max_value=total_count,
                    value=default_shown, step=1, key=f"n_{_sanitize_key(label)}_{dl_key}",
                    label_visibility="collapsed", width=110,
                )
                clicked = st.button(
                    f"Analyse further: {label} (of {total_count})",
                    key=f"drill_{_sanitize_key(label)}_{dl_key}",
                )
                mail_clicked = False
                if is_branch_scoped:
                    mail_clicked = st.button(
                        "Draft email", key=f"mail_{_sanitize_key(label)}_{dl_key}",
                    )
            if clicked or mail_clicked:
                shown = int(n)
                kind = category["kind"]
                if kind == "concept":
                    concept = category["concept"]
                    drill_df = STEP_REGISTRY["concept_filter"](
                        df_curr, concept, scope_col=scope_col, scope_value=scope_value,
                    ).head(shown)
                    new_type, new_value = "concept_filter", concept
                    new_params = {"concept": concept, "scope_col": scope_col, "scope_value": scope_value}
                elif kind == "fleet_defaulters":
                    drill_df = STEP_REGISTRY["fleet_defaulters"](
                        df_curr, scope_col=scope_col, scope_value=scope_value,
                    ).head(shown)
                    new_type, new_value = "fleet_defaulters", scope_value or "portfolio"
                    new_params = {"scope_col": scope_col, "scope_value": scope_value}
                else:  # top_closing_arrears
                    drill_df = STEP_REGISTRY["top_closing_arrears"](
                        df_curr, top_n=shown, scope_col=scope_col, scope_value=scope_value,
                    )
                    new_type, new_value = "top_closing_arrears", scope_value or "portfolio"
                    new_params = {"top_n": shown, "scope_col": scope_col, "scope_value": scope_value}
            if clicked:
                memory.touch(
                    new_type, new_value, result_df=drill_df, step_type=new_type, params=new_params,
                )
                _append_drill_turn(f"show me the top {shown} {label.lower()}", new_type, new_value)
                st.rerun()
            elif mail_clicked:
                branch_name = str(scope_value)
                to_list, cc_list = get_branch_recipients(branch_name)
                _subject, eml_bytes = draft_priority_email(
                    drill_df, branch_name, to_list, cc_list, category_label=label,
                )
                if not to_list:
                    st.caption(
                        "No manager email on file for this branch yet "
                        "(investigator/branch_recipients.py) -- the draft's To field "
                        "will be blank; fill it in yourself before sending."
                    )
                st.download_button(
                    f"Download draft: {label} (.eml)",
                    data=eml_bytes,
                    file_name=f"{branch_name}_{_sanitize_key(label)}.eml",
                    mime="message/rfc822",
                    key=f"dl_mail_{_sanitize_key(label)}_{dl_key}",
                )

    elif record.step_type == "priority_accounts" and "Cust Mob No" in result_df.columns:
        # Two independent drills at once (not an elif chain like the others
        # below) -- a priority list genuinely supports both simultaneously:
        # which executive has the most cases right now, and pulling any one
        # customer's full history. Checked the actual output columns before
        # assuming this had no drill-down value (it does -- loan-level,
        # Cust Mob No/MNT NAME/Unit on every row, the same shape
        # non_paying_customers already has).
        if st.button(
            "Analyse further: which executives have the most cases",
            key=f"drill_exec_{dl_key}",
        ):
            drill_df = STEP_REGISTRY["priority_by_executive"](df_curr, result_df)
            new_type = "priority_by_executive"
            new_value = f"priority_accounts:{record.entity_value}"
            memory.touch(
                new_type, new_value, result_df=drill_df, step_type="priority_by_executive",
                params={"source_entities": [{"entity_type": record.entity_type, "entity_value": record.entity_value}]},
            )
            _append_drill_turn("which executives have the most cases", new_type, new_value)
            st.rerun()

        mob = st.text_input(
            "Enter a customer's mobile number to view their full loan book",
            key=f"mob_{dl_key}",
        )
        if mob.strip() and st.button("View full loan book", key=f"loanbook_{dl_key}"):
            book = STEP_REGISTRY["customer_loan_book"](df_curr, mob.strip())
            memory.touch(
                "customer", mob.strip(), result_df=book, step_type="customer_loan_book",
                params={"cust_mob_no": mob.strip()},
            )
            _append_drill_turn(f"show me {mob.strip()}'s full loan book", "customer", mob.strip())
            st.rerun()

        # A THIRD independent option, same reasoning as the two above --
        # this list IS the "what to focus on" answer; closing the loop from
        # "the tool found it" to "a human can act on it" is the actual
        # differentiator, not another drill. Only offered when scoped to a
        # SINGLE real branch (record.params.get("scope_col") == "branch" --
        # not just "entity_value looks like a name," since portfolio-wide or
        # region-scoped priority lists have no one branch manager to email).
        # NEVER sends anything -- builds a downloadable .eml the user opens
        # and sends themselves (investigator/email_draft.py's own docstring
        # explains why: this must work in a cloud deployment too, where
        # win32com/Outlook automation simply isn't available).
        if (record.params or {}).get("scope_col") == "branch" and "Unit" in result_df.columns:
            branch_name = str(record.entity_value)
            to_list, cc_list = get_branch_recipients(branch_name)
            st.caption(f"Draft an email to {branch_name}'s branch manager with this priority list.")
            if to_list:
                st.caption(f"To: {'; '.join(to_list)}")
            else:
                st.caption(
                    "No manager email on file for this branch yet (investigator/branch_recipients.py) "
                    "-- the draft's To field will be blank; fill it in yourself before sending."
                )
            _subject, eml_bytes = draft_priority_email(result_df, branch_name, to_list, cc_list)
            st.download_button(
                "Draft email to branch manager (.eml)",
                data=eml_bytes,
                file_name=f"{branch_name}_priority_cases.eml",
                mime="message/rfc822",
                key=f"email_draft_{dl_key}",
            )

    elif record.step_type == "entity_summary" and record.entity_type == "branch":
        _render_mechanism_suggestions(record, df_curr, df_prev, as_of, memory, dl_key)
        # Drillable metrics come from whatever entity_summary actually
        # returned, NOT gated on Delta being present -- a first upload with
        # no previous-month file has every Delta as None, and the user must
        # still be able to pick a metric to drill into from current
        # standing alone (this was a real bug caught by an AppTest smoke
        # check: the button silently never appeared at all in that case).
        drillable = [m for m in result_df["Metric"] if m in _BRANCH_TO_EXEC_METRIC]
        if drillable:
            worst = _worst_metric(result_df)
            default_idx = drillable.index(worst) if worst in drillable else 0
            chosen = st.selectbox(
                "Focus on which metric?", drillable, index=default_idx, key=f"metric_pick_{dl_key}",
            )
            if st.button(
                f"Analyse further: executives behind {record.entity_value}'s {chosen}",
                key=f"drill_{dl_key}",
            ):
                exec_metric = _BRANCH_TO_EXEC_METRIC[chosen]
                scoped = df_curr[df_curr["Unit"].astype(str).str.strip().str.upper() == str(record.entity_value).strip().upper()]
                drill_df = STEP_REGISTRY["underperformer_quartile"](scoped, exec_metric)
                new_type, new_value = "branch_executives", f"{record.entity_value}:{exec_metric}"
                memory.touch(
                    new_type, new_value,
                    result_df=drill_df, metric_focus=exec_metric, step_type="underperformer_quartile",
                    params={"metric": exec_metric, "scope_col": "branch", "scope_value": record.entity_value},
                )
                _append_drill_turn(f"executives behind {record.entity_value}'s {chosen}", new_type, new_value)
                st.rerun()

    elif record.step_type == "entity_summary" and record.entity_type == "executive":
        # The LAST hop of the region -> branch -> executive -> loan drill
        # chain -- previously a dead end (no "Analyse further" existed here
        # at all): "deep dive into Rahul" answered how Rahul is doing, but
        # had no button to go one level further into WHICH of his loans are
        # responsible. _EXEC_METRICS' own spelling ("NPA %", with a space)
        # already matches METRIC_DRILL_FILTER's keys directly -- no
        # _BRANCH_TO_EXEC_METRIC translation needed here, unlike the branch
        # block above (that one crosses FROM branch-grain TO executive-grain
        # spelling; this one stays at executive grain throughout).
        drillable = [m for m in result_df["Metric"] if m in METRIC_DRILL_FILTER]
        if drillable:
            worst = _worst_metric(result_df)
            default_idx = drillable.index(worst) if worst in drillable else 0
            chosen = st.selectbox(
                "Focus on which metric?", drillable, index=default_idx, key=f"metric_pick_{dl_key}",
            )
            exec_label = record.entity_value[0] if isinstance(record.entity_value, tuple) else record.entity_value
            if st.button(f"Analyse further: loans behind {exec_label}'s {chosen}", key=f"drill_{dl_key}"):
                drill_df = STEP_REGISTRY["worst_loans_by_metric"](
                    df_curr, chosen, scope_col="executive", scope_value=record.entity_value,
                )
                summary = summarize_worst_loans(drill_df)
                new_type = "worst_loans_by_metric"
                new_value = f"{chosen}:{record.entity_value}"
                scope_value = list(record.entity_value) if isinstance(record.entity_value, tuple) else record.entity_value
                memory.touch(
                    new_type, new_value, result_df=drill_df, step_type="worst_loans_by_metric",
                    narrative=summary,
                    params={"metric": chosen, "scope_col": "executive", "scope_value": scope_value},
                )
                _append_drill_turn(f"loans behind {exec_label}'s {chosen}", new_type, new_value)
                st.rerun()

    elif record.step_type == "entity_summary" and record.entity_type in ("region", "zone", "bu"):
        _render_mechanism_suggestions(record, df_curr, df_prev, as_of, memory, dl_key)
        # ONE-LEVEL-AT-A-TIME: drill into BRANCHES within this region/zone/BU
        # first, never straight to executives -- matches the hierarchy
        # BU > Zone > Region > Branch > Executive > Customer. Uses the same
        # no-space metric names dimension_breakdown's own columns use
        # (Collection%/NPA%/Hard Bucket%), since this hop stays at
        # dimension_breakdown's grain, not the executive scorecard's.
        drillable = [m for m in result_df["Metric"] if m in METRIC_DIRECTION]
        if drillable:
            worst = _worst_metric(result_df)
            default_idx = drillable.index(worst) if worst in drillable else 0
            chosen = st.selectbox(
                "Focus on which metric?", drillable, index=default_idx, key=f"metric_pick_{dl_key}",
            )
            if st.button(
                f"Analyse further: branches within {record.entity_value}'s {chosen}",
                key=f"drill_{dl_key}",
            ):
                drill_df = STEP_REGISTRY["dimension_breakdown"](
                    df_curr, "branch", scope_col=record.entity_type, scope_value=record.entity_value,
                    metric=chosen, higher_is_better=METRIC_DIRECTION.get(chosen),
                )
                new_type = "dimension_breakdown"
                new_value = f"branch within {record.entity_value}"
                memory.touch(
                    new_type, new_value, result_df=drill_df,
                    metric_focus="branch", step_type="dimension_breakdown",
                    params={
                        "dimension": "branch", "scope_col": record.entity_type,
                        "scope_value": record.entity_value, "metric": chosen,
                    },
                )
                _append_drill_turn(f"branches within {record.entity_value}'s {chosen}", new_type, new_value)
                st.rerun()

    elif record.step_type == "dimension_breakdown" and record.metric_focus == "branch" and "Unit" in result_df.columns:
        # Continue the hierarchy: drill into the WORST branch's executives
        # (result_df is already sorted worst-first by dimension_breakdown).
        worst_branch = result_df.iloc[0]["Unit"]
        drillable = [m for m in result_df.columns if m in _BRANCH_TO_EXEC_METRIC]
        if drillable:
            chosen = st.selectbox(
                "Focus on which metric?", drillable, key=f"metric_pick_{dl_key}",
            )
            if st.button(
                f"Analyse further: executives behind {worst_branch}'s {chosen}",
                key=f"drill_{dl_key}",
            ):
                exec_metric = _BRANCH_TO_EXEC_METRIC[chosen]
                scoped = df_curr[df_curr["Unit"].astype(str).str.strip().str.upper() == str(worst_branch).strip().upper()]
                drill_df = STEP_REGISTRY["underperformer_quartile"](scoped, exec_metric)
                new_type, new_value = "branch_executives", f"{worst_branch}:{exec_metric}"
                memory.touch(
                    new_type, new_value, result_df=drill_df, metric_focus=exec_metric,
                    step_type="underperformer_quartile",
                    params={"metric": exec_metric, "scope_col": "branch", "scope_value": worst_branch},
                )
                _append_drill_turn(f"executives behind {worst_branch}'s {chosen}", new_type, new_value)
                st.rerun()

    elif record.step_type in ("underperformer_quartile", "bottom_n_per_group", "intersect_entities"):
        if "MNT NAME" in result_df.columns and st.button(
            "Analyse further: find non-paying customers behind these executives",
            key=f"drill_{dl_key}",
        ):
            drill_df = STEP_REGISTRY["non_paying_customers"](df_curr, result_df)
            new_type, new_value = "non_paying_customers", f"{record.entity_type}:{record.entity_value}"
            source_entities = [{"entity_type": record.entity_type, "entity_value": record.entity_value}]
            memory.touch(
                new_type, new_value, result_df=drill_df, step_type="non_paying_customers",
                params={"source_entities": source_entities},
            )
            _append_drill_turn("find non-paying customers behind these executives", new_type, new_value)
            st.rerun()

        # Companion to the button above, not a replacement -- "find
        # non-paying customers" is a generic hard-bucket threshold across
        # the WHOLE list; this is specifically "which loans are driving the
        # metric THIS list was ranked by" (record.metric_focus), scoped
        # the SAME way the list itself was (record.params' own
        # scope_col/scope_value), so a list of "Mahad's worst NPA%
        # executives" drills into Mahad's own worst NPA loans, not diluted
        # by the rest of the portfolio.
        if record.metric_focus in METRIC_DRILL_FILTER and st.button(
            f"Analyse further: which loans are driving {record.metric_focus}",
            key=f"drill_metric_{dl_key}",
        ):
            list_scope_col = (record.params or {}).get("scope_col")
            list_scope_value = (record.params or {}).get("scope_value")
            drill_df = STEP_REGISTRY["worst_loans_by_metric"](
                df_curr, record.metric_focus, scope_col=list_scope_col, scope_value=list_scope_value,
            )
            summary = summarize_worst_loans(drill_df)
            new_type = "worst_loans_by_metric"
            new_value = f"{record.metric_focus}:{list_scope_value or 'portfolio'}"
            memory.touch(
                new_type, new_value, result_df=drill_df, step_type="worst_loans_by_metric",
                narrative=summary,
                params={"metric": record.metric_focus, "scope_col": list_scope_col, "scope_value": list_scope_value},
            )
            _append_drill_turn(f"which loans are driving {record.metric_focus}", new_type, new_value)
            st.rerun()

    elif record.step_type == "concept_filter":
        # concept_filter otherwise has NO drill at all -- "are there any non
        # starters in my region" was a dead end. This is loan-level data
        # with Unit/MNT NAME on every row, so "which branch/executive has
        # the most" is genuinely answerable -- concept_breakdown groups by
        # EVERY column resolve_dimension returns (not just the first), so
        # "executive" keys on (MNT NAME, Unit) together, never name alone.
        dim_options = [d for d in ("branch", "executive") if all(c in result_df.columns for c in resolve_dimension(d))]
        if dim_options:
            dimension = st.selectbox(
                "Break down by which dimension?", dim_options, key=f"concept_dim_{dl_key}",
            )
            if st.button(f"Analyse further: break down by {dimension}", key=f"drill_{dl_key}"):
                drill_df = STEP_REGISTRY["concept_breakdown"](result_df, dimension)
                new_type, new_value = "concept_breakdown", f"{record.entity_type}:{record.entity_value}:{dimension}"
                source_entities = [{"entity_type": record.entity_type, "entity_value": record.entity_value}]
                memory.touch(
                    new_type, new_value, result_df=drill_df, step_type="concept_breakdown",
                    params={"source_entities": source_entities, "dimension": dimension},
                )
                _append_drill_turn(f"break this down by {dimension}", new_type, new_value)
                st.rerun()

    elif record.step_type == "non_paying_customers" and "Cust Mob No" in result_df.columns:
        mob = st.text_input(
            "Enter a customer's mobile number to view their full loan book",
            key=f"mob_{dl_key}",
        )
        if mob.strip() and st.button("View full loan book", key=f"loanbook_{dl_key}"):
            book = STEP_REGISTRY["customer_loan_book"](df_curr, mob.strip())
            memory.touch(
                "customer", mob.strip(), result_df=book, step_type="customer_loan_book",
                params={"cust_mob_no": mob.strip()},
            )
            _append_drill_turn(f"show me {mob.strip()}'s full loan book", "customer", mob.strip())
            st.rerun()


def _render_result_body(
    record, df_curr: pd.DataFrame, memory: EntityMemory, show_controls: bool, dl_key: str,
    df_prev: pd.DataFrame | None = None, as_of=None,
) -> None:
    """Renders one assistant turn's content: narrative (already-captured,
    static -- streaming only ever happens once, when a result is freshly
    computed) + dataframe + download, and -- only for the latest message --
    the "Analyse further" controls. dl_key is the CALLER's turn-scoped key
    (see _turn_dl_key) -- this function never derives its own from entity
    identity alone, since the same entity can legitimately be asked about
    more than once in one conversation (a real production crash: two chat
    bubbles both keyed "inv_region_CS_Nagar")."""
    result_df = record.result_df
    if result_df is None or result_df.empty:
        st.info("No rows matched for this step.")
        return

    if record.narrative:
        st.markdown(record.narrative)
    st.dataframe(result_df, width='stretch', hide_index=True)
    _dl_btn(result_df, f"investigator_{record.step_type or 'result'}.xlsx", key=dl_key)

    if show_controls:
        _render_analyse_further(record, df_curr, memory, dl_key, df_prev=df_prev, as_of=as_of)


def _maybe_set_thread_title(question: str) -> None:
    """Auto-titles a thread from its first question, same convention
    ChatGPT-style UIs use -- only fires once (while the title is still the
    "New chat" placeholder), never overwrites a title on later turns."""
    thread = _get_threads()[_get_active_thread_id()]
    if thread["title"] == _NEW_CHAT_TITLE:
        title = question.strip()
        thread["title"] = title[:_TITLE_MAX_CHARS] + ("..." if len(title) > _TITLE_MAX_CHARS else "")


def _process_new_question(
    question: str, df_curr: pd.DataFrame, df_prev: pd.DataFrame, memory: EntityMemory, as_of=None,
) -> None:
    turns = _get_turns()
    _maybe_set_thread_title(question)
    turns.append({"role": "user", "text": question})
    with st.chat_message("user", avatar=_USER_AVATAR):
        st.write(question)

    with st.chat_message("assistant", avatar=_ASSISTANT_AVATAR):
        if question.strip().lower() == _HELP_TRIGGER:
            # Answered directly, no Gemini call at all -- "what can you help
            # me with" is a question about the TOOL, not the portfolio data,
            # and parse_turn's vocabulary has no step for that anyway.
            st.markdown(_HELP_RESPONSE)
            turns.append({"role": "assistant", "text": _HELP_RESPONSE, "kind": "info"})
            return

        if not os.environ.get("GOOGLE_API_KEY"):
            st.error("GOOGLE_API_KEY not found in .env file.")
            turns.append({"role": "assistant", "text": "GOOGLE_API_KEY not found in .env file."})
            return

        with st.status("Thinking...", expanded=False) as status:
            status.write("Understanding your question...")
            turn = parse_turn(question, memory)
            turn = validate_and_correct(turn, memory, df_curr)

            if turn["needs_clarification"] or not turn["step_type"]:
                status.update(label="Needs a bit more detail", state="error", expanded=True)
                text = turn.get("clarification_question") or "Could you rephrase that?"
                st.warning(text)
                turns.append({"role": "assistant", "text": text})
                return

            status.write(f"Running: {turn['step_type']}")
            try:
                result_df = _execute_step(turn["step_type"], turn["params"], df_curr, df_prev, memory, as_of=as_of)
            except (KeyError, ValueError) as e:
                status.update(label="Couldn't run that", state="error", expanded=True)
                text = f"I couldn't run that: {e}"
                st.warning(text)
                turns.append({"role": "assistant", "text": text})
                return

            status.update(label=f"Done: {turn['step_type']}", state="complete")

        # Outside the status block -- the "thinking" indicator has
        # collapsed, and the actual chat content (streamed narrative, then
        # the table) renders as the message body, same shape ai_query.py's
        # own st.status(...) -> results-below pattern already uses.
        narrative = ""
        if result_df is not None and not result_df.empty and st.session_state.get("investigator_narrate"):
            narrative = st.write_stream(narrate_step_stream(turn["step_type"], result_df))

        entity_type, entity_value = _derive_entity_key(
            turn["step_type"], turn["params"], turn.get("resolved_entity"),
        )
        # Canonicalize a resolved executive's identity BEFORE storing it --
        # entity_summary's own executive-matching (_executive_row) resolves
        # a typed name ("Taushif Khan") against the real data's MNT NAME
        # ("TAUSIF KHAN NAZIR KH") internally, but that resolution used to
        # be thrown away: only the computed table reached here, never the
        # canonical identity. Storing the raw typed name broke any LATER
        # step needing an exact match (non_paying_customers's source-table
        # merge requires exact MNT NAME/Unit) -- confirmed live, twice:
        # once as a silent empty result, once as the WRONG executive's data
        # (the LLM substituting a stale, unrelated table once the typed
        # name failed to resolve anywhere useful). Every later reference to
        # "this executive" in the conversation, by any step, now resolves
        # against the same real identity.
        if entity_type == "executive":
            raw_name, raw_unit = entity_value if isinstance(entity_value, tuple) else (entity_value, None)
            canonical = resolve_executive_identity(df_curr, raw_name, raw_unit)
            if canonical:
                entity_value = canonical
                turn["params"] = {**turn["params"], "entity_value": list(canonical)}
        record = memory.touch(
            entity_type, entity_value, result_df=result_df,
            metric_focus=_metric_focus_for(turn["step_type"], turn["params"]), step_type=turn["step_type"],
            narrative=narrative, params=turn["params"],
        )
        turns.append({"role": "assistant", "entity_type": entity_type, "entity_value": entity_value})
        dl_key = _turn_dl_key(len(turns) - 1, entity_type, entity_value)

        if result_df is None or result_df.empty:
            st.info("No rows matched for this step.")
        else:
            st.dataframe(result_df, width='stretch', hide_index=True)
            _dl_btn(result_df, f"investigator_{turn['step_type']}.xlsx", key=dl_key)
            _render_analyse_further(record, df_curr, memory, dl_key, df_prev=df_prev, as_of=as_of)


def _render_header() -> None:
    """One bordered toolbar holding the title, thread switcher, New chat,
    and settings -- previously these were separate floating rows (title,
    colored badges, pills, button) with no visual grouping, which read as
    cluttered/disconnected rather than one cohesive control (confirmed
    directly from a screenshot: "its mixing"). Minimal by design: one plain
    caption line instead of two colored badge chips crowding the title."""
    threads = _get_threads()
    active_id = _get_active_thread_id()
    thread_ids = sorted(threads.keys(), key=lambda tid: threads[tid]["created_at"])
    on_empty_thread = not _get_turns()

    with st.container(border=True, key="investigator_header_card"):
        with st.container(horizontal=True, vertical_alignment="center"):
            st.markdown(
                "<div style='font-size:20px;font-weight:800;color:#FFC000;'>"
                "🕵️ Investigator</div>",
                unsafe_allow_html=True,
            )
            with st.container(horizontal=True, horizontal_alignment="right", vertical_alignment="center", gap="small"):
                # Only show the picker once there's something to actually
                # pick between -- a single untouched thread is titled "New
                # chat", and a lone pill reading "New chat" sitting right
                # next to a "New chat" BUTTON is confusing (they look like
                # two controls doing the same thing). One thread needs no
                # switcher at all. This generalizes to ANY number of blank
                # "New chat" threads, not just one -- confirmed directly
                # from a screenshot: clicking "New chat" more than once
                # accumulates several untouched threads, each showing as
                # its OWN pill still reading "New chat", multiplying the
                # exact confusion this was meant to prevent. A blank thread
                # you're NOT currently on adds nothing selectable (there's
                # nothing in it yet to switch back to), so only the
                # CURRENTLY active thread is ever shown blank; every other
                # blank one is hidden from the switcher entirely.
                selectable_ids = [
                    tid for tid in thread_ids
                    if tid == active_id or threads[tid]["title"] != _NEW_CHAT_TITLE
                ]
                if len(selectable_ids) > 1:
                    # thread_ids themselves are the pill VALUES (format_func
                    # only controls the display text) -- matching on title
                    # text alone would be ambiguous the moment two threads
                    # share a title, e.g. two untouched "New chat" threads
                    # before either gets a first question.
                    #
                    # Key is scoped to active_id, NOT a fixed name --
                    # st.pills only honors `default` on a key's FIRST
                    # render; on every later rerun it returns whatever was
                    # last stored under that key, ignoring `default`
                    # entirely. A fixed key also can't be reassigned
                    # programmatically once instantiated in the same run
                    # (Streamlit raises StreamlitAPIException -- hit this
                    # directly clicking "New chat" right after the pills
                    # widget had already rendered this run). Scoping the
                    # key to active_id sidesteps both: switching threads
                    # always lands on a brand-new key, so `default` is
                    # honored fresh every time, with nothing to reassign.
                    # Truncated independently of the full stored thread
                    # title (_TITLE_MAX_CHARS=40, sized for the thread list/
                    # tab elsewhere) -- a 40-char pill label in this tight
                    # horizontal toolbar overflowed past the "New chat"
                    # button with no wrapping, visually overlapping it
                    # (confirmed directly from a screenshot). This truncation
                    # is display-only; the full title is untouched.
                    def _pill_label(tid: str) -> str:
                        title = threads[tid]["title"]
                        return title if len(title) <= 18 else title[:18] + "..."

                    chosen_id = st.pills(
                        "Conversation", selectable_ids, default=active_id, required=True,
                        format_func=_pill_label,
                        key=f"investigator_thread_picker_{active_id}", label_visibility="collapsed",
                    )
                else:
                    chosen_id = active_id
                if st.button(
                    "New chat", key="investigator_new_chat", icon=":material/add_circle:",
                    disabled=on_empty_thread,
                    help="You're already starting a fresh conversation" if on_empty_thread else None,
                ):
                    new_id = _new_thread()
                    st.session_state["investigator_active_thread"] = new_id
                    st.rerun()
                with st.popover("", icon=":material/tune:", key="investigator_settings"):
                    st.toggle(
                        "Narrate each step", key="investigator_narrate", value=False,
                        help="Adds one extra Gemini call per step to write a short "
                             "observation above the table. Off by default to keep "
                             "answers instant and cheap.",
                    )
        st.caption("Ask anything about your portfolio. AI only routes the question; every table is exact source data, downloadable for audit.")

    if chosen_id != active_id:
        st.session_state["investigator_active_thread"] = chosen_id
        st.rerun()


def _render_proactive_opener(df_curr: pd.DataFrame, alerts_curr: list | None, memory: EntityMemory) -> bool:
    """Shows up to 3 active smart alerts (already computed once by app.py's
    own cached run_all_alerts call, reused here -- never a second pass over
    the whole DataFrame) as the FIRST thing a new/empty thread sees, each
    with a one-click "analyse further" straight into concept_filter -- ZERO
    Gemini calls, since a clicked alert already names its own concept
    exactly (same "known answer, skip the LLM" precedent as _HELP_TRIGGER
    above). A genuinely clean portfolio (nothing active) renders nothing
    here, silently -- the plain suggestion pills below are the fallback,
    not a message insisting something's wrong when it isn't. Returns
    whether anything was actually rendered, so the caller can adapt the
    suggestion pills' own caption ("Or try asking:" vs "Try asking:")."""
    flags = _top_active_alerts(alerts_curr)
    if not flags:
        return False
    with st.chat_message("assistant", avatar=_ASSISTANT_AVATAR):
        st.markdown("Here's what moved most this month, before you even ask:")
        for i, flag in enumerate(flags):
            st.markdown(f"**{flag.get('icon', '')} {flag['title']}**: {flag['count']} accounts. {flag.get('subtitle', '')}")
            mapping = _ALERT_TITLE_TO_STEP.get(flag["title"])
            if mapping and st.button(f"Analyse further: {flag['title']}", key=f"proactive_{i}"):
                step_type, params = mapping
                drill_df = STEP_REGISTRY[step_type](df_curr, **params)
                entity_type, entity_value = _derive_entity_key(step_type, params, None)
                memory.touch(entity_type, entity_value, result_df=drill_df, step_type=step_type, params=params)
                _append_drill_turn(f"are there any {flag['title'].lower()}", entity_type, entity_value)
                st.rerun()
    return True


def render_investigator_tab(
    df_curr: pd.DataFrame, df_prev: pd.DataFrame, data_version: int = 0, filter_key: str = "",
    alerts_curr: list | None = None, curr_month: str | None = None,
) -> None:
    # st.chat_input is a native widget with no styling of its own -- it
    # renders on secondaryBackgroundColor (.streamlit/config.toml: "#1a1a1a",
    # near-black) but inherits the app-wide textColor ("#000000", black) for
    # what you type, which is unreadable (black text on a near-black box).
    # Scoped fix, not a global theme change -- textColor also drives every
    # OTHER native-widget/default-text element on this app's WHITE main
    # background, so changing it globally would break readability
    # elsewhere. Matches this app's existing convention of hand-styling
    # light text on its own dark panels (e.g. ui/tabs/ai_query.py's #e6edf3
    # on dark cards) rather than fighting the global theme.
    st.markdown(
        """<style>
        /* "Focus on which metric?" selectbox (_render_analyse_further's
           several st.selectbox(key=f"metric_pick_{dl_key}") calls) --
           same root cause as the chat_input fix above: this app's theme
           (.streamlit/config.toml) sets secondaryBackgroundColor (widget
           background) to near-black AND textColor to pure black, so any
           unstyled widget outside the sidebar (which already has its own
           override, see ui/styles.py) renders black text on a black box.
           Matched via the "st-key-<key>" class Streamlit adds to any
           keyed element, with a substring selector since dl_key varies
           per rendered step. Also shrunk padding/min-height -- the
           default render was taller than the rest of this app's controls. */
        [class*="st-key-metric_pick_"] { max-width: 320px !important; }
        [class*="st-key-metric_pick_"] [data-baseweb="select"] > div,
        [class*="st-key-metric_pick_"] [role="combobox"] {
            background: #1a1a1a !important; border: 1px solid #4a4f57 !important;
            border-radius: 8px !important; min-height: 34px !important;
            color: #fff !important;
        }
        [class*="st-key-metric_pick_"] [data-baseweb="select"] *,
        [class*="st-key-metric_pick_"] [role="combobox"] * {
            color: #fff !important;
        }
        [data-testid="stChatInput"] textarea { color: #e6edf3 !important; }
        [data-testid="stChatInputTextArea"] { color: #e6edf3 !important; }
        [data-testid="stChatInput"] textarea::placeholder { color: #6b7280 !important; opacity: 1 !important; font-size: 0.9em !important; }
        [data-testid="stChatInput"] {
            border: 2px solid #4a4f57 !important;
            border-radius: 14px !important;
        }
        [data-testid="stChatInput"]:focus-within {
            border-color: #f5a623 !important;
        }
        [data-testid="stChatInputSubmitButton"] button,
        [data-testid="stChatInput"] button {
            background-color: #f5a623 !important;
            color: #1a1a1a !important;
        }
        [data-testid="stChatInputSubmitButton"] button svg,
        [data-testid="stChatInput"] button svg {
            fill: #1a1a1a !important;
        }

        /* Match the AI Query tab's dark panel look (ui/tabs/ai_query.py's
           #0d1117/#21262d/#FFC000 palette) instead of the default white
           bordered card, scoped via st.container(key=...)'s "st-key-*"
           class so nothing else on this page is affected. */
        .st-key-investigator_header_card,
        .st-key-investigator_suggestions_wrap {
            background: #0d1117 !important;
            border: 1px solid #21262d !important;
            border-radius: 12px !important;
        }
        /* Breathing room from whatever renders right above it (the
           proactive alerts, or nothing on a genuinely clean portfolio) --
           the card previously sat flush against it, reading as cramped. */
        .st-key-investigator_suggestions_wrap {
            margin-top: 12px !important; padding: 4px 4px !important;
        }
        .st-key-investigator_suggestions_wrap [data-testid="stButtonGroup"] button {
            padding: 6px 16px !important; font-size: 14px !important;
        }
        .st-key-investigator_header_card [data-testid="stCaptionContainer"],
        .st-key-investigator_suggestions_wrap [data-testid="stCaptionContainer"] {
            color: #c9d1d9 !important;
        }
        .st-key-investigator_header_card [data-testid="stButtonGroup"],
        .st-key-investigator_suggestions_wrap [data-testid="stButtonGroup"] {
            background: transparent !important; border: none !important;
            box-shadow: none !important; padding: 0 !important;
        }
        .st-key-investigator_header_card [data-testid="stButtonGroup"] button,
        .st-key-investigator_suggestions_wrap [data-testid="stButtonGroup"] button {
            background: #161b22 !important; color: #8b949e !important;
            border: 1px solid #2d333b !important; border-radius: 6px !important;
            font-style: italic; font-weight: 500 !important;
        }
        /* Belt-and-suspenders against the thread-picker pill overflowing
           past the "New chat" button (confirmed directly from a
           screenshot) -- the pill label is already truncated in Python
           (_pill_label above), this just guarantees any edge case (a very
           narrow viewport, a future longer label) degrades to an ellipsis
           instead of overlapping a sibling control. flex-wrap lets the
           toolbar drop to a second line rather than overlap if it ever
           genuinely runs out of horizontal room. */
        .st-key-investigator_header_card [data-testid="stButtonGroup"] button {
            max-width: 160px !important; overflow: hidden !important;
            text-overflow: ellipsis !important; white-space: nowrap !important;
        }
        .st-key-investigator_header_card [data-testid="stHorizontalBlock"] {
            flex-wrap: wrap !important; row-gap: 8px !important;
        }
        .st-key-investigator_header_card [data-testid="stButtonGroup"] button:hover,
        .st-key-investigator_suggestions_wrap [data-testid="stButtonGroup"] button:hover {
            background: #2a2a2a !important; color: #d0d0d0 !important; border-color: #555 !important;
        }
        .st-key-investigator_header_card [data-testid="stButtonGroup"] button[aria-checked="true"],
        .st-key-investigator_header_card [data-testid="stButtonGroup"] button[aria-pressed="true"] {
            background: #FFC000 !important; color: #000000 !important;
            font-style: normal; font-weight: 700 !important; box-shadow: none !important;
        }
        </style>""",
        unsafe_allow_html=True,
    )

    _render_header()
    memory = _get_memory()
    turns = _get_turns()

    last_assistant_idx = max(
        (i for i, t in enumerate(turns) if t["role"] == "assistant"), default=-1,
    )
    for i, t in enumerate(turns):
        if t["role"] == "user":
            with st.chat_message("user", avatar=_USER_AVATAR):
                st.write(t["text"])
        else:
            with st.chat_message("assistant", avatar=_ASSISTANT_AVATAR):
                if "entity_type" in t:
                    record = memory.get(t["entity_type"], t["entity_value"])
                    if record is not None:
                        dl_key = _turn_dl_key(i, t["entity_type"], t["entity_value"])
                        _render_result_body(
                            record, df_curr, memory, show_controls=(i == last_assistant_idx), dl_key=dl_key,
                            df_prev=df_prev, as_of=curr_month,
                        )
                    else:
                        st.info("This result is no longer available.")
                elif t.get("kind") == "info":
                    st.markdown(t.get("text", ""))
                else:
                    st.warning(t.get("text", ""))

    if not turns:
        had_alerts = _render_proactive_opener(df_curr, alerts_curr, memory)
        with st.container(key="investigator_suggestions_wrap"):
            st.caption("Or try asking:" if had_alerts else "Try asking:")
            suggestion = st.pills(
                "Suggestions", list(_SUGGESTIONS), label_visibility="collapsed",
                key="investigator_suggestions",
            )
        if suggestion:
            _process_new_question(_SUGGESTIONS[suggestion], df_curr, df_prev, memory, as_of=curr_month)
            st.rerun()

    prompt = st.chat_input(
        "Ask anything about your portfolio, e.g. \"which branches are "
        "underperforming\" or \"meanwhile, how are executive in my region performing\"",
    )
    if prompt:
        _process_new_question(prompt.strip(), df_curr, df_prev, memory, as_of=curr_month)
