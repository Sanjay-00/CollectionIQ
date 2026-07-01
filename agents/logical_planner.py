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
import os
import json
import re

from google import genai
from langsmith import traceable

from config import GEMINI_MODEL
from registry.ontology import CONCEPTS, METRICS
from registry.semantic_model import DIMENSIONS
from agents.domain_expert import (
    _call_gemini_with_retry,
    _add_token_usage,
    build_snapshot_context,
)


def build_catalog() -> str:
    """Generate the catalog section from the registry (injected whole  -  no RAG)."""
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


def _build_full_system_prompt(snapshot_dates: dict | None = None, allow_clarification: bool = True) -> str:
    catalog = build_catalog()

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

    clarification_rule = (
        'Set needs_clarification=true ONLY for genuine material ambiguity where two '
        'interpretations give materially different numbers (e.g. user says "big accounts" '
        'without a threshold). For clear queries, ALWAYS proceed (needs_clarification=false).'
    ) if allow_clarification else (
        'needs_clarification MUST be false  -  the user has already clarified. Proceed with best interpretation.'
    )

    return f"""You are the semantic layer of CollectionIQ, an NBFC loan-collection analytics engine.
Convert a natural language portfolio query into a flat logical IR (JSON).
A deterministic compiler lowers the IR to execution  -  you declare WHAT, not HOW.
Never author execution steps, group keys, or computation details.

NBFC GLOSSARY:
- SOH = POS + Closing Arrears = total exposure if a customer fully defaults
- Arrears/EMI: delinquency ratio. >0=delinquent, >1=SMA-1, >2=SMA-2, ≥3=NPA (90+ DPD)
- curr_bucket: current DPD bucket  -  STD | 1-30 DPD | SMA-1 | SMA-2 | NPA
- prev_bucket: same bucket, previous month (only when a prior file is uploaded)
- Loan Status: RUN (active loan) | MAT (matured, arrears outstanding) | S&S (seized & sold)
- Non Starter = Y: customer never paid their first EMI  -  highest NPA risk
- Strike = Y: this customer has a payment obligation due this month
- CoLending_Loans = Y: partner-bank co-lending loan  -  priority for collections
- No Coll 3 Months and >6 EMI: Y = zero payment for ≥3 consecutive months AND >6 EMI arrears
- LCC% = cumulative collection efficiency = Cum Coll / Cum Due × 100, capped at 100
{snapshot_block}

{catalog}

KEY COLUMNS FOR FILTERS (exact names; prefer catalog concepts when they fit):
  Flags (Y/N/NA): Non Starter | Strike | NACHStatus | LGL_FLAG | CoLending_Loans
  Bucket cols: curr_bucket | prev_bucket  (values: STD | 1-30 DPD | SMA-1 | SMA-2 | NPA)
  Special flag: "No Coll 3 Months and >6 EMI"   -  for "3 months no payment" or "chronic defaulters"
  Loan Status: RUN | MAT | S&S
  Numeric: SOH | POS | Closing Arrears | Arrears / EMI | LCC%
           Month Collection (Excluding Reserve Collection) | Net Collection Demand Inst+Exp+BC
           ARREARS AGAINST INST | ARREARS AGAINST EXP
  Dates (YYYY-MM-DD for filter values): Ag_Date | Last Receipt Date
  Identity: Loan No | Cust Name | Cust Mob No | RegionName | Unit | MNT NAME | MNT CODE | SRC Name

BUCKET MOVEMENT (requires snapshot file):
  "worsened / rolled forward / degraded"  → {{"column":"curr_bucket","op":"bucket_worse_than","value":"prev_bucket"}}
  "improved / cured / rolled back"        → {{"column":"curr_bucket","op":"bucket_better_than","value":"prev_bucket"}}
  These work in top-level "filters" and inside count measure "where" clauses.

ENTITY FILTERS (for nested "per-group with per-entity threshold"):
  Example: "branches with customers who have >3 loans"
  → entity_filters: [{{"entity":"customer","having":[{{"agg":"count","distinct":"Loan No","op":">","value":3}}]}}]
  entity values: customer | loan | executive | branch | region
  ENTITY CONCEPT shorthand: {{"concept":"fleet_operator"}} = customer with ≥3 loans

OUTPUT  -  return a JSON object with EXACTLY these keys:
{{
  "intent":               "loan_table | aggregation | single_value | priority_action",
  "query_title":          "5-7 word descriptive title",
  "risk_flag":            "high | medium | low",
  "description":          "one-sentence restatement of what will be computed",
  "needs_clarification":  false,
  "clarification_question": "",
  "clarification_options": [],
  "filters":        [...],        // row-level conditions applied before any aggregation
  "dimensions":     [...],        // dimension aliases (branch/region/executive/customer)
  "measures":       [...],        // what to compute per group (or in total)
  "entity_filters": [],           // per-entity nested predicates (see ENTITY FILTERS above)
  "metrics":        [...],        // derived columns: expr over measure aliases
  "having":         [...],        // post-aggregation group threshold (SQL HAVING)
  "order_by":       [...],        // sort order
  "limit":          null,
  "display_columns": [],          // for loan_table: columns to show (see DISPLAY COLUMNS)
  "time":           null
}}

FILTER format:
  {{"concept": "<catalog concept name>"}}
  {{"column": "<exact col>", "op": "==|!=|>|>=|<|<=|in|bucket_worse_than|bucket_better_than", "value": <v>}}

MEASURE format:
  {{"metric": "<catalog metric>", "alias": "<name>"}}
  {{"column": "<col>", "agg": "sum|mean|min|max|nunique", "alias": "<name>"}}
  {{"agg": "count", "alias": "<name>"}}
  {{"agg": "count", "concept": "<concept>", "alias": "<name>"}}
  {{"agg": "count", "where": [<filter items>], "alias": "<name>"}}
  {{"agg": "count", "distinct": "<col>", "alias": "<name>"}}

METRICS format (derived columns computed from measure aliases  -  use for differences/ratios):
  {{"alias": "npa_reduction_pct", "expr": "(prev_npa - curr_npa) / prev_npa * 100"}}
  {{"alias": "collection_pct", "expr": "collection / demand * 100"}}
  CRITICAL: expr must reference the exact ALIAS you set in measures  -  never the metric name.
  If you wrote {{"metric": "prev_npa_count", "alias": "prev_npa"}}, use "prev_npa" in expr, NOT "prev_npa_count".
  Always add an order_by on the derived alias when the user asks to sort by it.

COLUMN ORDER for before/after comparisons: always list the earlier-period (prev) measure FIRST,
  then the current-period measure. Example: prev_npa first, curr_npa second.

ORDER_BY format: [{{"by": "<measure alias or metric alias>", "dir": "asc|desc"}}]
  Always use the key "by" (never "column", "alias", "field", or "measure").
  Example: [{{"by": "npa_reduction_pct", "dir": "desc"}}]

TIME format:
  {{"grain":"month","compare":{{"type":"snapshot|change","from":"prev","to":"curr"}}}}
  snapshot = both periods side-by-side; change = also compute the delta column (curr - prev)

DISPLAY COLUMNS (for loan_table intent):
  Leave display_columns EMPTY for all general queries  -  the system returns ALL columns by default.
  Only populate display_columns when the user EXPLICITLY asks for specific columns
  (e.g. "show me only Loan No, SOH and branch" or "give me just the contact details").
  Never set display_columns just because a column is relevant to the query.

INTENT RULES:
  loan_table:       result is individual loan/customer rows. Use filters + display_columns.
                    No dimensions or measures needed.
  aggregation:      result is one row per group  -  rankings, comparisons, breakdowns.
                    Covers "top branches by X", "which branch has most", "X per region".
                    Use dimensions + measures. No display_columns. limit is null unless
                    user explicitly says a number ("top 3", "top 5").
  single_value:     a SCALAR answer with NO group breakdown  -  portfolio-wide totals and
                    counts ("how many NPA accounts total", "what is total SOH", "what % are
                    in SMA"). When any dimension grouping is needed, use aggregation instead.
  priority_action:  user says "what to focus on" | "urgent cases" | "prioritize" | "action needed".
                    Leave filters/dimensions/measures/display_columns EMPTY  -  system applies 7-tier framework.

ROUTING RULES:
  - "show/list/find accounts/customers" → loan_table
  - "haven't paid for 3 months" / "no payment for 3 months" / "chronic defaulters"
      → concept "no_collection_3m" in filters, intent loan_table
      (do NOT compute from Last Receipt Date  -  use the catalog concept)
  - "unpaid this month" / "zero collection this month" → concept "no_collection" in filters
  - "by/per/across branch|region|executive" → aggregation
  - "top branch|region|executive by X" → aggregation (ALL groups ranked, limit null)
  - "which branch has most/highest/lowest" → aggregation (ALL branches ranked, limit null)
  - "top N branches" where N is a number → aggregation with limit N
  - "how many total / total portfolio SOH / overall %" → single_value (no dimensions)
  - "vs last month" / "change since last month" / "MoM change" → time.compare type:"change"
  - "last month vs this month" / "then vs now" / "both periods" → time.compare type:"snapshot"
  - "biggest reduction in [metric]" / "sorted by reduction" / "who reduced most" →
      aggregation with two measures (curr + prev) + metrics derive (reduction = prev - curr) + sort desc
      Do NOT use time.compare for these  -  they are ranking queries, not time-series snapshots.
  - "what to focus on" / "prioritize" / "urgent" → priority_action
  - For month-over-month queries, ALWAYS set time block (never manual prev_* columns).
  - NEVER set limit unless the user says an explicit number ("top 3", "top 10", "5 branches").

CATALOG PREFERENCE ORDER (correctness guarantee):
  1. Use a catalog CONCEPT whenever it fits a filter need  -  never restate its conditions.
  2. Use a catalog METRIC whenever it fits a measure need  -  never restate its formula.
  3. Use a catalog DIMENSION alias (branch/region/executive)  -  never raw column names for grouping.
  4. Fall back to raw column/agg only when NO catalog item fits.

CLARIFICATION RULE: {clarification_rule}

RISK FLAG:
  high = NPA/SMA/CoLending/NonStarter/Legal/Strike queries; medium = general delinquency;
  low = analytics, performance, ranking queries without direct default risk.

Return ONLY valid JSON. Use a single hyphen (-); never an em/en dash ( -  or  - ).
"""


_IR1_KEYS = (
    "intent", "query_title", "risk_flag", "description",
    "needs_clarification", "clarification_question", "clarification_options",
    "filters", "dimensions", "measures", "entity_filters",
    "metrics", "having", "order_by", "limit", "display_columns", "time",
)


def _coerce_dim(d) -> str:
    """Ensure a dimension entry is a plain string alias."""
    if isinstance(d, str):
        return d
    if isinstance(d, dict):
        return d.get("alias") or d.get("name") or d.get("column") or str(d)
    return str(d)


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
        "filters":                raw.get("filters") or [],
        "dimensions":             [_coerce_dim(d) for d in (raw.get("dimensions") or [])],
        "measures":               raw.get("measures") or [],
        "entity_filters":         raw.get("entity_filters") or [],
        "metrics":                raw.get("metrics") or [],
        "having":                 raw.get("having") or [],
        "order_by":               raw.get("order_by") or [],
        "limit":                  raw.get("limit"),
        "display_columns":        raw.get("display_columns") or [],
        "time":                   raw.get("time"),
    }


@traceable(run_type="chain", name="LogicalPlanner", tags=["gemini", "nbfc", "ir1"])
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
    api_key = os.environ.get("GOOGLE_API_KEY", "")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable not set.")

    system_prompt = _build_full_system_prompt(snapshot_dates, allow_clarification)
    repair_context = f"[REPAIR  -  {repair_feedback}] " if repair_feedback else ""

    client = genai.Client(api_key=api_key)
    response = _call_gemini_with_retry(
        client, GEMINI_MODEL, repair_context + query,
        {"system_instruction": system_prompt},
    )
    _add_token_usage(response)

    raw = response.text.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    return _normalize_ir1(json.loads(raw))
