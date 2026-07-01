"""Plan critic  -  a self-review pass over a generated step-plan.

The column-validator (plan_executor.validate_plan) catches *structurally invalid*
plans (hallucinated columns). It cannot catch a plan that is structurally valid
but does not match intent  -  e.g. a DROPPED condition, or a step that references a
column an earlier group_aggregate removed. This LLM pass reviews the plan against
the original question for completeness and coherence and returns a corrected plan.

It is non-fatal by design: any failure returns the original plan unchanged, and
the deterministic validator remains the final safety gate.
"""

import os
import re
import json
import time

from google import genai
from langsmith import traceable

from config import GEMINI_MODEL
from agents.domain_expert import build_snapshot_context


def _call_gemini_with_retry(client, model: str, contents: str, config: dict, max_retries: int = 2):
    for attempt in range(max_retries + 1):
        try:
            return client.models.generate_content(model=model, contents=contents, config=config)
        except Exception:
            if attempt == max_retries:
                raise
            time.sleep(2 ** attempt)


CRITIC_PROMPT = """You review a STEP-PLAN built to answer an analytics question over an NBFC loan dataset.
Judge the plan ONLY against the user's question. Check, in order:

1. COMPLETENESS - every condition/constraint stated in the question must appear in the plan, either as
   a 'filter' step or as a 'where' on an aggregation. A DROPPED condition is the most serious error.
2. COHERENCE - each step may only use columns available at that point. A 'group_aggregate' REPLACES the
   table with its group_by columns plus the new aliases; any later step that references a column an
   earlier group_aggregate dropped is invalid and must be fixed.
3. CORRECTNESS - the final group_by and the output columns match what the user asked for; counts of
   ENTITIES (e.g. customers) use nunique on the entity key, not row counts.

Step operations: group_aggregate (each aggregation may carry an optional "where": [conditions]),
filter, derive, sort, limit. The customer key is "Cust Mob No". Conditions look like
{{"column": "<col or alias>", "op": ">|>=|<|<=|==|!=|in|contains", "value": <v>}}.

Available columns (use these EXACT names only): {columns}

Return ONLY JSON, no markdown:
{{"ok": true|false, "issues": ["short description per problem"], "plan": [ ...the plan... ]}}
If the plan is already correct, set "ok": true and return it unchanged in "plan".
If not, set "ok": false, list the issues, and return a corrected "plan".
Use a single hyphen (-) for any dash. Never invent column names.
"""


@traceable(run_type="chain", name="PlanCritic", tags=["gemini", "nbfc", "plan-review"])
def critique_plan(query: str, plan: list, columns, snapshot_dates: dict | None = None) -> tuple[list, str]:
    """Review a plan against the query. Returns (plan, issues_note).
    On any problem reaching the model, returns the original plan unchanged (non-fatal)."""
    if not plan:
        return plan, ""
    api_key = os.environ.get("GOOGLE_API_KEY", "")
    if not api_key:
        return plan, ""

    system_prompt = CRITIC_PROMPT.format(columns=", ".join(map(str, columns)))
    snapshot_context = build_snapshot_context(snapshot_dates)
    contents = (
        snapshot_context
        + f'User question: "{query}"\n\n'
        + f"Plan to review:\n{json.dumps(plan, ensure_ascii=False)}"
    )

    try:
        client = genai.Client(api_key=api_key)
        response = _call_gemini_with_retry(
            client, GEMINI_MODEL, contents, {"system_instruction": system_prompt},
        )
        raw = response.text.strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        parsed = json.loads(raw)
    except Exception:
        return plan, ""  # non-fatal  -  keep original, validator is the safety net

    revised = parsed.get("plan")
    if isinstance(revised, list) and revised:
        note = "" if parsed.get("ok") else "; ".join(parsed.get("issues") or [])
        return revised, note
    return plan, ""
