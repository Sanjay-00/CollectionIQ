import re

from langsmith import traceable
from config import GEMINI_MODEL
from gemini_client import add_token_usage as _add_token_usage, call_gemini_with_retry as _call_gemini_with_retry, make_client
from gemini_client import ai_key, ai_unavailable_reason

SYSTEM_PROMPT = """You are a senior credit risk analyst at an NBFC (Non-Banking Financial Company).
Generate concise, actionable observations from loan portfolio query results.
Write exactly 4-5 bullet points.
Each bullet must start with "• ".
Be specific: use the numbers provided. Focus on risk, action, and urgency.
Do not use generic statements. Do not repeat the same fact in different words.
Use a single hyphen (-) when a dash is needed. Never use a double hyphen, an em dash, or an en dash.
Quote every number exactly as it appears in the data. Do not convert units, round, add up or compute new numbers.
Only describe the rows you are shown.
"""

_NUM = re.compile(r"\d[\d,]*(?:\.\d+)?")
# Counting words ("3 of the 5 branches", "top 10") are fine without a source.
_SMALL_INT_MAX = 10


def _values(text: str) -> set[float]:
    return {float(n.replace(",", "")) for n in _NUM.findall(text)}


def _unverified(line: str, allowed: set[float]) -> list[float]:
    return [v for v in _values(line)
            if v not in allowed and not (v.is_integer() and v <= _SMALL_INT_MAX)]


def check_numbers(answer: str, facts: str) -> str:
    """Keep only the bullets whose every number is in `facts` (compared by
    value: "12.5" matches "12.50"), the same guard as the report's AI summary
    (report_agent/ai_summary.py). Says how many were removed."""
    allowed = _values(facts)
    kept, removed = [], 0
    for line in (ln for ln in answer.splitlines() if ln.strip()):
        if _unverified(line, allowed):
            removed += 1
        else:
            kept.append(line)
    if removed and not kept:
        return "• AI observations removed: they used numbers that are not in the result."
    if removed:
        kept.append(f"• {removed} observation(s) removed: they used numbers that are not in the result.")
    return "\n".join(kept)


def _cell(v):
    """A table value as the model should quote it: floats to 2 decimals."""
    return round(v, 2) if isinstance(v, float) else v


def _fmt_money(val: float) -> str:
    if abs(val) >= 1_00_00_000:
        return f"₹{val / 1_00_00_000:.2f}Cr"
    if abs(val) >= 1_00_000:
        return f"₹{val / 1_00_000:.2f}L"
    return f"₹{val:,.0f}"


@traceable(run_type="chain", name="InsightGenerator", tags=["gemini", "nbfc", "insight-generation"])
def generate_insights(
    query: str,
    plain_english: str,
    kpis: dict,
    rankings: dict,
    insight_focus: str = "",
) -> str:
    api_key = ai_key()
    if not api_key:
        return f"• AI observations unavailable. {ai_unavailable_reason()}"

    agg_rows = kpis.get("_agg_rows")

    if agg_rows:
        # Aggregation result  -  build context from the grouped table rows
        rows_text = "\n".join(
            "  " + ", ".join(f"{k}: {_cell(v)}" for k, v in row.items() if k != "Rank")
            for row in agg_rows
        )
        total = kpis.get("Count", len(agg_rows))
        shown = (f"all {total} rows" if len(agg_rows) >= total
                 else f"the first {len(agg_rows)} of {total} rows, in the table's order")
        context = f"""
User query: "{query}"
Interpreted as: {plain_english}
Focus area: {insight_focus if insight_focus else "General portfolio risk analysis"}

RESULT TABLE ({shown}):
{rows_text}
"""
    else:
        # Loan-level filter result
        region_top = list(rankings.get("region_counts", {}).items())
        branch_top = list(rankings.get("branch_counts", {}).items())
        branch_pos_top = list(rankings.get("branch_pos", {}).items())
        bucket_dist = rankings.get("bucket_dist", {})

        context = f"""
User query: "{query}"
Interpreted as: {plain_english}
Focus area: {insight_focus if insight_focus else "General portfolio risk analysis"}

RESULTS SUMMARY:
- Matching accounts: {kpis.get('Count', 0)}
- Total outstanding (POS): {_fmt_money(kpis.get('Total POS', 0))}
- Average Arrears/EMI ratio: {kpis.get('Avg Arrears/EMI', 0):.2f}
- Monthly demand: {_fmt_money(kpis.get('Total Demand', 0))}
- Monthly collection: {_fmt_money(kpis.get('Total Collection', 0))}
- Collection %: {kpis.get('Collection %', 0):.2f}%

TOP REGIONS BY ACCOUNT COUNT:
{chr(10).join(f"  {r}: {c} accounts" for r, c in region_top[:3]) if region_top else "  No data"}

TOP BRANCHES BY ACCOUNT COUNT:
{chr(10).join(f"  {b}: {c} accounts" for b, c in branch_top[:3]) if branch_top else "  No data"}

TOP BRANCHES BY POS:
{chr(10).join(f"  {b}: {_fmt_money(p)}" for b, p in branch_pos_top[:3]) if branch_pos_top else "  No data"}

BUCKET DISTRIBUTION (% of matched accounts):
{chr(10).join(f"  {k}: {v}%" for k, v in bucket_dist.items()) if bucket_dist else "  No data"}
"""

    client = make_client(api_key)
    response = _call_gemini_with_retry(
        client, GEMINI_MODEL, context,
        {"system_instruction": SYSTEM_PROMPT},
    )
    _add_token_usage(response)
    return check_numbers(response.text.strip(), context)
