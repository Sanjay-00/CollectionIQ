"""IR-1 Planner (Gemini)  -  the v2 semantic layer / live query pipeline entry-point.

Phase 3: this module is now the SOLE LLM entry-point for query understanding.
It replaces the Domain Expert, Plan Critic, and Query Parser from the legacy pipeline.
The model emits a flat declarative IR-1; the deterministic compiler (compiler.core)
lowers it to the physical plan; the pandas executor runs it.

Prompt structure:
  - NBFC glossary (key terms + column names the model needs to refer accurately)
  - Registry catalog (concepts, metrics, dimensions)  -  full, no RAG at ~30 concepts
  - Extended IR-1 schema (intent, metadata, filters, dimensions, measures, time, etc.)
  - Rules for routing, clarification, priority, bucket ops, and display_columns
"""
import re
from functools import lru_cache
from pathlib import Path

from langsmith import traceable

from config import GEMINI_MODEL, PLANNER_TEMPERATURE
from gemini_client import JSONReplyError, json_config, make_client, parse_json_object
from gemini_client import ai_key, ai_unavailable_reason
from registry.ontology import CONCEPTS, METRICS, AMBIGUOUS_TERMS
from registry.semantic_model import DIMENSIONS
from registry.views import VIEWS, _METRIC_DIRECTION
from gemini_client import add_token_usage as _add_token_usage, call_gemini_hedged as _call_gemini_hedged


@lru_cache(maxsize=1)
def build_catalog() -> str:
    """Generate the catalog section from the registry (injected whole  -  no RAG).

    Cached: this is a zero-argument function depending only on registry/ module
    contents, which are static for the process lifetime -- _build_full_system_prompt
    rebuilt this from scratch on EVERY query (including a second time on any
    compile/validate repair retry) despite it never actually varying per-query."""
    lines = [
        "CONCEPTS (use the name in filters; the compiler expands to the full definition):",
    ]
    for name, c in CONCEPTS.items():
        lines.append(f"  {name}: {c['description']}")
    lines.append("\nMETRICS (use the name in measures):")
    for name, m in METRICS.items():
        lines.append(f"  {name}: {m['description']}")
    lines.append("\nDIMENSIONS (use the alias in dimensions; maps to the real column):")
    for alias, cols in DIMENSIONS.items():
        lines.append(f"  {alias} → {', '.join(cols)}")
    return "\n".join(lines)


@lru_cache(maxsize=1)
def build_views_catalog() -> str:
    """Generate the VIEWS catalog section  -  pre-computed analyses to prefer over
    building filters/dimensions/measures from scratch when one matches exactly.

    Cached -- same rationale as build_catalog() above."""
    lines = ["VIEWS (pre-computed analyses: try to match one of these FIRST):"]
    for name, v in VIEWS.items():
        lines.append(f"  {name}: {v['description']}")
        if v.get("params"):
            param_bits = ", ".join(
                f"{p} (default {d['default']})" for p, d in v["params"].items()
            )
            lines.append(f"    params: {param_bits}")
        if not v.get("filterable", True):
            lines.append("    (not filterable: never attach \"filters\" to this view)")
        if v.get("metrics"):
            # Annotate each metric with its direction (higher=worse / higher=better) so
            # the model can resolve qualitative phrases like "best/worst performing
            # first" or "who is falling behind" for ANY metric via one general rule,
            # instead of needing a one-off worked example per metric name.
            annotated = ", ".join(
                f"{m} ({'higher=worse' if _METRIC_DIRECTION.get(m) == 'high_bad' else 'higher=better'})"
                for m in v["metrics"]
            )
            lines.append(f"    highlightable metrics: {annotated}")
    return "\n".join(lines)


# The planner's instructions live in agents/prompts/planner_system.txt (code
# review item 6, 2026-10-27): a 380-line f-string inside this function made the
# prompt hard to read and diff. str.format placeholders ({catalog} etc.); a
# literal brace is doubled ({{ }}). Bump the version whenever the text changes:
# it is recorded on every LangSmith trace of the planner. (The plan cache keys
# on the prompt's full text, so it needs no bump to stay correct.)
PLANNER_PROMPT_VERSION = 2   # 2 (2026-10-28): catalog gained npa_pct, sma2_pct, roll_forward_pct, roll_backward_pct
_PROMPT_FILE = Path(__file__).parent / "prompts" / "planner_system.txt"


@lru_cache(maxsize=1)
def _planner_template() -> str:
    return _PROMPT_FILE.read_text(encoding="utf-8")


def _build_full_system_prompt(snapshot_dates: dict | None = None, allow_clarification: bool = True) -> str:
    catalog = build_catalog()
    views_catalog = build_views_catalog()

    snapshot_block = ""
    if snapshot_dates and snapshot_dates.get("prev"):
        snapshot_block = (
            f"\nSNAPSHOTS: previous-period data IS loaded. "
            f"curr={snapshot_dates.get('curr','?')}  prev={snapshot_dates.get('prev','?')}. "
            "Bucket/portfolio movement queries and prev_* column references are valid."
        )
    else:
        snapshot_block = (
            "\nSNAPSHOTS: NO previous-period file loaded. "
            "Do NOT reference prev_* columns or bucket movement (bucket_worse_than / bucket_better_than)."
        )

    if allow_clarification:
        ambiguous_terms_block = "\n".join(
            f'  - "{t["term"]}": {t["note"]}' for t in AMBIGUOUS_TERMS
        )
        clarification_rule = (
            'Set needs_clarification=true ONLY for genuine material ambiguity where two '
            'interpretations give materially different numbers (e.g. user says "big accounts" '
            'without a threshold). For clear queries, ALWAYS proceed (needs_clarification=false).\n'
            "  KNOWN DOMAIN-AMBIGUOUS TERMS: if the query uses one of these WITHOUT already "
            "specifying which reading is meant elsewhere in the wording, you MUST set "
            "needs_clarification=true with 2-4 concrete options; do NOT silently pick a default "
            "interpretation for these specifically, even if one reading seems more likely. THIS "
            "CHECK TAKES PRIORITY OVER THE OUT OF SCOPE RULE BELOW: a query naming one of these "
            "terms (e.g. \"branch business details\", \"which region has best business\") is a "
            "real, on-topic portfolio question that needs disambiguating: it is NEVER out of "
            "scope just because it's vague or uses an ambiguous word. Only classify as out of "
            "scope when the query has NO recognizable portfolio intent at all (small talk, "
            "general knowledge, gibberish), not merely because a term in it is ambiguous:\n"
            f"{ambiguous_terms_block}"
        )
        out_of_scope_precedence_note = (
            "\n  DO NOT use this path for a vaguely-worded but clearly portfolio-related question "
            "(branches, regions, executives, accounts, business, risk, performance, etc.): that "
            "is the KNOWN DOMAIN-AMBIGUOUS TERMS case above, or ordinary underspecified phrasing "
            "the CLARIFICATION RULE already covers, never \"out of scope\". Reserve this ONLY for "
            "input with no recognizable portfolio intent whatsoever."
        )
    else:
        clarification_rule = (
            'needs_clarification MUST be false: the user has already clarified. Proceed with best interpretation.'
        )
        # No AMBIGUOUS_TERMS block exists in this branch (see clarification_rule
        # above) -- referencing it here would be a dangling pointer to a
        # section that isn't in the prompt this call, and would also break the
        # "ambiguous-terms text never appears in a post-clarification retry"
        # contract that keeps this path from re-triggering the same clarification loop.
        out_of_scope_precedence_note = ""

    return _planner_template().format(
        snapshot_block=snapshot_block,
        views_catalog=views_catalog,
        catalog=catalog,
        clarification_rule=clarification_rule,
        out_of_scope_precedence_note=out_of_scope_precedence_note,
    )


def _coerce_dim(d) -> str:
    """Ensure a dimension entry is a plain string alias."""
    if isinstance(d, str):
        return d
    if isinstance(d, dict):
        return d.get("alias") or d.get("name") or d.get("column") or str(d)
    return str(d)


def _coerce_highlight_metrics(raw) -> list[dict]:
    """Basic type-safety pass on the planner's highlight_metrics list -- keeps only
    well-formed {column, agg} entries, capped at 4. Whether the column is actually
    valid for the matched view (in its "metrics" list, has a known direction) is
    validated later in graph.py::view_node, which has the resolved view spec."""
    out = []
    for item in (raw or []):
        if not isinstance(item, dict):
            continue
        col, agg = item.get("column"), item.get("agg")
        if col and agg in ("max", "min"):
            out.append({"column": col, "agg": agg})
        if len(out) >= 4:
            break
    return out


def _coerce_sort_by(raw) -> dict | None:
    """Basic type-safety pass on the planner's sort_by request -- a single
    {column, dir} or None. Whether the column actually exists in the matched
    view's result_df is validated later in graph.py::view_node; an unknown
    column there is silently ignored (the view keeps its own default order),
    never an error -- a bad sort request shouldn't kill an otherwise-good
    view match."""
    if not isinstance(raw, dict) or not raw.get("column"):
        return None
    return {"column": raw["column"], "dir": raw.get("dir") if raw.get("dir") in ("asc", "desc") else "desc"}


def _coerce_view(v) -> dict | None:
    """Ensure a "view" entry is either None or a well-formed
    {name, params, filters, highlight_metrics, sort_by} dict."""
    if not v or not isinstance(v, dict) or not v.get("name"):
        return None
    return {
        "name": v["name"],
        "params": v.get("params") or {},
        "filters": v.get("filters") or [],
        "highlight_metrics": _coerce_highlight_metrics(v.get("highlight_metrics")),
        "sort_by": _coerce_sort_by(v.get("sort_by")),
    }


def _normalize_ir1(raw: dict) -> dict:
    """Coerce a raw model response into a well-formed IR-1 with safe defaults."""
    raw = raw or {}
    return {
        "intent":                 raw.get("intent") or "loan_table",
        "query_title":            raw.get("query_title") or "Custom Query",
        "risk_flag":              raw.get("risk_flag") or "medium",
        "description":            raw.get("description") or "",
        "needs_clarification":    bool(raw.get("needs_clarification", False)),
        "clarification_question": raw.get("clarification_question") or "",
        "clarification_options":  raw.get("clarification_options") or [],
        "view":                   _coerce_view(raw.get("view")),
        "filters":                raw.get("filters") or [],
        "dimensions":             [_coerce_dim(d) for d in (raw.get("dimensions") or [])],
        "measures":               raw.get("measures") or [],
        "entity_filters":         raw.get("entity_filters") or [],
        "metrics":                raw.get("metrics") or [],
        "having":                 raw.get("having") or [],
        "order_by":               raw.get("order_by") or [],
        "limit":                  raw.get("limit"),
        "display_columns":        raw.get("display_columns") or [],
        "show_all_columns":       bool(raw.get("show_all_columns", False)),
        "time":                   raw.get("time"),
    }


from config import MAX_QUERY_CHARS  # noqa: E402 -- shared with investigator/llm.py

# Cheap, deterministic backstop for "show all/every column(s)" phrasing. The
# Logical Planner setting show_all_columns itself is an LLM judgment call, not
# a guarantee -- observed in practice to be inconsistent even across two calls
# with the EXACT same query text (one call correctly set it, the next call for
# an identical real-world query did not, and also built a display_columns list
# missing the very columns the user named). A keyword match can't cover every
# possible phrasing the model might otherwise parse correctly, but it reliably
# catches the common, explicit case as a SECOND, independent layer -- same
# two-layers-not-one philosophy this codebase already applies to the derive
# -expression sandbox (compile-time AND execute-time) and HTML escaping.
_ALL_COLUMNS_PATTERN = re.compile(r"\b(?:all|every)\s+(?:the\s+)?columns?\b", re.IGNORECASE)


def _query_requests_all_columns(query: str) -> bool:
    return bool(_ALL_COLUMNS_PATTERN.search(query))


# Deterministic fast-path for the "business" AMBIGUOUS_TERMS case, same
# two-layers-not-one philosophy as _query_requests_all_columns above. Unlike a
# generic ambiguity ("big accounts" -- needs real judgment about what "big"
# means), THIS specific case is fully specifiable: it only depends on (a) is
# the term present without an already-disambiguating word, and (b) which
# entity grain (a plain dictionary lookup against real registry.views.VIEWS
# data, never a copied/hardcoded sentence) -- exactly the kind of decision
# this codebase's "vocabulary, not logic, in the LLM" pattern says should be a
# lookup, not an LLM judgment call. Scoped ONLY to "business", not "risk": risk
# is a within-view metric choice (no clean two-competing-views split to look
# up), a genuinely different shape of ambiguity that a keyword lookup can't
# resolve as cleanly -- kept on the existing (temperature-lowered) LLM path.
# Deliberately narrow and easy to remove wholesale if it ever causes more
# trouble than the rare case it fixes is worth: everything below is additive
# and short-circuits BEFORE any Gemini call -- delete this block and
# plan_logical falls back to exactly the behavior it had before it existed.
_BUSINESS_TERM_RE = re.compile(r"\bbusiness\b", re.IGNORECASE)
_NEW_ADVANCES_HINT_RE = re.compile(
    r"\b(new\s+advance|new\s+loan|originat|disburs|\bfund(?:ed|ing)?\b)", re.IGNORECASE
)
_PERFORMANCE_HINT_RE = re.compile(
    r"\b(collection|npa|delinquen|concern\s+score|strike|recovery|sma-?2)\b", re.IGNORECASE
)
_GRAIN_KEYWORDS = {
    "branch":    (r"\bbranch(?:es)?\b", r"\bunit(?:s)?\b"),
    "executive": (r"\bexecutive(?:s)?\b", r"\bexec(?:s)?\b", r"\bfield\s+officer\b", r"\bagent(?:s)?\b"),
    "region":    (r"\bregion(?:s)?\b", r"\bzone(?:s)?\b"),
}
_GRAIN_DISPLAY = {"branch": "branches", "executive": "executives", "region": "regions"}
_GRAIN_TO_VIEWS = {
    "branch":    ("new_advances_by_branch", "branch_quadrant"),
    "executive": ("new_advances_by_executive", "executive_recovery"),
    "region":    ("new_advances_by_region", "region_scorecard"),
}


def _resolve_business_ambiguity_deterministically(query: str) -> dict | None:
    """Returns a ready-to-use IR-1 clarification dict for the fully-specifiable
    "business" case, or None to fall through to the normal LLM-driven path
    unchanged (same graceful-fallthrough convention view_node already uses).
    Every word in the returned option text comes straight from registry.views.
    VIEWS ("label"/"metrics"), never a copied/hardcoded sentence -- correct by
    construction for whatever grain was actually detected, and automatically
    stays correct if those views are ever renamed or re-metriced."""
    if not _BUSINESS_TERM_RE.search(query):
        return None
    if _NEW_ADVANCES_HINT_RE.search(query) or _PERFORMANCE_HINT_RE.search(query):
        return None  # already disambiguated by other wording -- let the LLM proceed normally
    grain = next(
        (g for g, patterns in _GRAIN_KEYWORDS.items()
         if any(re.search(p, query, re.IGNORECASE) for p in patterns)),
        None,
    )
    if grain is None:
        return None  # can't confidently tell WHO the question is about -- fall through

    adv_name, perf_name = _GRAIN_TO_VIEWS[grain]
    if adv_name not in VIEWS or perf_name not in VIEWS:
        return None  # registry changed under us -- fail safe, never crash on a stale mapping
    perf_spec = VIEWS[perf_name]
    perf_metrics = ", ".join((perf_spec.get("metrics") or [])[:4]) or "portfolio performance"

    return _normalize_ir1({
        "intent": "loan_table",
        "query_title": "Business: Which Reading?",
        "needs_clarification": True,
        "clarification_question": (
            f"When you say 'business' for {_GRAIN_DISPLAY[grain]}, do you mean new loans "
            "originated this period, or portfolio/collection performance?"
        ),
        "clarification_options": [
            "New advances (accounts/funded this period)",
            f"{perf_spec['label']} ({perf_metrics})",
        ],
    })


# Standalone example queries offered whenever the input couldn't be
# understood at all -- without these, the out-of-scope path was a dead end:
# empty clarification_options renders NO buttons (unlike a genuine ambiguity,
# which always gives the user something clickable), leaving "go retype your
# question from scratch" as the only way forward. Kept in sync with the 2
# examples already named in the OUT OF SCOPE prompt section below by hand --
# both describe the same fixed pair, just for the two different code paths
# (LLM-classified vs. the Python-side guardrails that never call the model).
_OUT_OF_SCOPE_EXAMPLE_OPTIONS = [
    "Top 10 delinquent customers by SOH",
    "NPA% by branch",
]


def _out_of_scope_ir(message: str, options: list[str] | None = None) -> dict:
    """A safe, structured "can't help with that" response -- reuses the SAME
    needs_clarification path the UI already renders (no new UI code needed),
    for both genuinely off-topic input and input we couldn't safely process
    (oversized, or the model broke JSON format under adversarial/gibberish
    input). Never invents a fabricated query or leaks a raw exception.
    `options` defaults to a fixed, safe example pair -- ui/query_answer.py
    runs a clicked option as a FRESH query (never appended to the original
    nonsense text as an "interpretation"), since these are standalone
    examples, not clarifying details about the same original query."""
    return _normalize_ir1({
        "intent": "loan_table",
        "query_title": "Out of Scope",
        "needs_clarification": True,
        "clarification_question": message,
        "clarification_options": options if options is not None else _OUT_OF_SCOPE_EXAMPLE_OPTIONS,
    })


@traceable(run_type="chain", name="LogicalPlanner", tags=["gemini", "nbfc", "ir1"],
           metadata={"prompt_version": PLANNER_PROMPT_VERSION})
def plan_logical(
    query: str,
    snapshot_dates: dict | None = None,
    repair_feedback: str = "",
    allow_clarification: bool = True,
) -> dict:
    """Produce IR-1 for a natural-language query (Phase 3 live path).

    Returns a fully-normalized IR-1 dict including metadata fields
    (query_title, risk_flag, description, needs_clarification, …).
    """
    api_key = ai_key()
    if not api_key:
        raise ValueError(ai_unavailable_reason())

    # Input hygiene: reject absurdly long input before ever calling the model --
    # cheap cost/DoS guardrail, and a real business question is never this long.
    if len(query) > MAX_QUERY_CHARS:
        return _out_of_scope_ir(
            f"That question is too long ({len(query)} characters). Please ask a "
            "shorter, more specific question about the portfolio, e.g. "
            "\"top 10 delinquent customers by SOH\" or \"NPA% by branch\"."
        )

    # Deterministic fast-path for the fully-specifiable "business" ambiguity --
    # zero Gemini calls when it resolves (cheaper AND faster, not just more
    # reliable than the LLM-judgment path it replaces for this one case). Gated
    # on allow_clarification=True the same way the AMBIGUOUS_TERMS prompt
    # section is: False means either a clarification follow-up (already
    # resolved -- must not re-trigger) or the unrelated compiler repair loop
    # (graph.py's one-shot retry), neither of which this should ever touch.
    if allow_clarification:
        resolved = _resolve_business_ambiguity_deterministically(query)
        if resolved is not None:
            return resolved

    system_prompt = _build_full_system_prompt(snapshot_dates, allow_clarification)
    repair_context = f"[REPAIR: {repair_feedback}] " if repair_feedback else ""

    # A clarification follow-up query carries its own resolved interpretation
    # IN THE TEXT ITSELF -- ui/query_answer.py appends "(interpretation: X)"
    # when a user clicks a clarification option. Detected here via the query
    # TEXT, deliberately NOT via allow_clarification: that flag is ALSO set
    # False by the unrelated compiler repair loop above (graph.py's one-shot
    # retry on a validation error), which must never receive this -- there is
    # no interpretation to act on there, and injecting this instruction into
    # a repair retry would confuse an already-tested, working mechanism.
    # Query-text detection naturally does the right thing on a clarification
    # follow-up's OWN repair retry too (state["query"] still carries the
    # marker then), without needing a third flag threaded through the graph.
    clarification_followup_context = (
        "[CLARIFICATION RESOLVED: the \"(interpretation: ...)\" text above names the "
        "metric/reading the user just chose in response to a clarifying question. You "
        "MUST act on it: if it names one of the matched view's own highlightable metrics, "
        "set BOTH sort_by (rank on that metric, applying the existing good/bad-direction "
        "rule based on whether the ORIGINAL wording said best/top/winning vs worst/falling "
        "behind) AND highlight_metrics for that SAME metric+direction. Do not leave "
        "sort_by/highlight_metrics empty just because a view already matched: the whole "
        "point of asking was to determine exactly this.] "
    ) if "(interpretation:" in query else ""

    client = make_client(api_key)
    response = _call_gemini_hedged(
        client, GEMINI_MODEL, repair_context + clarification_followup_context + query,
        json_config(
            system_instruction=system_prompt,
            # This is structured classification/extraction (pick an intent, pick
            # filters/views from a fixed vocabulary, decide needs_clarification),
            # not creative writing -- the default temperature (~1.0) is tuned for
            # open-ended generation and was the direct cause of confirmed,
            # reproducible run-to-run inconsistency observed live this session
            # (the identical query sometimes asking for clarification with
            # correct grain-aware options, sometimes silently guessing, sometimes
            # misrouting to the out-of-scope fallback). Lowered, not zeroed: 0.0
            # (fully greedy) risks the model getting stuck repeating a wrong
            # answer with zero chance to recover on a retry; a low-but-nonzero
            # value keeps the model strongly favoring its single best-judged
            # interpretation while still allowing the compiler's own repair loop
            # (plan_logical's OWN retry path above) a real chance to differ if
            # the first attempt was actually wrong, not just re-confirm the same
            # mistake deterministically forever.
            temperature=PLANNER_TEMPERATURE,
        ),
    )
    _add_token_usage(response)

    try:
        parsed = parse_json_object(response.text)
    except JSONReplyError:
        # The model broke JSON format -- in practice this happens on wildly
        # off-topic/adversarial input where it drops into prose instead of
        # complying (JSON mode makes it rare). Never surface a raw parser
        # exception to the user; treat it the same as an out-of-scope question.
        return _out_of_scope_ir(
            "I couldn't understand that as a portfolio question. I can answer "
            "things like \"top 10 delinquent customers by SOH\" or \"NPA% by branch\"."
        )
    ir1 = _normalize_ir1(parsed)
    if _query_requests_all_columns(query):
        ir1["show_all_columns"] = True
    # The model's OWN live out-of-scope classification (per the OUT OF SCOPE
    # prompt section) explicitly sets clarification_options=[] -- same dead-end
    # UI state _out_of_scope_ir's own default now avoids for the Python-side
    # guardrail paths. Applied as a blanket fallback (any needs_clarification
    # with empty options, not just this specific case): a genuine ambiguity is
    # always instructed to give concrete options of its own, so empty options
    # reaching here means the model didn't, and offering these safe examples
    # is strictly better than a clickable-nothing dead end either way.
    if ir1.get("needs_clarification") and not ir1.get("clarification_options"):
        ir1["clarification_options"] = list(_OUT_OF_SCOPE_EXAMPLE_OPTIONS)
    return ir1
