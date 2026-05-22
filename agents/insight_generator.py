import os
from google import genai

SYSTEM_PROMPT = """You are a senior credit risk analyst at an NBFC (Shriram Finance).
Generate concise, actionable observations from loan portfolio query results.
Write exactly 4-5 bullet points.
Each bullet must start with "• ".
Be specific — use the numbers provided. Focus on risk, action, and urgency.
Do not use generic statements. Do not repeat the same fact in different words.
"""


def _fmt_money(val: float) -> str:
    if abs(val) >= 1_000_000:
        return f"₹{val / 1_000_000:.1f}M"
    if abs(val) >= 1_000:
        return f"₹{val / 1_000:.1f}K"
    return f"₹{val:.0f}"


def generate_insights(
    query: str,
    plain_english: str,
    kpis: dict,
    rankings: dict,
    insight_focus: str = "",
) -> str:
    api_key = os.environ.get("GOOGLE_API_KEY", "")
    if not api_key:
        return "• GOOGLE_API_KEY not set — AI observations unavailable."

    # Build a structured context summary for Gemini
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

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=context,
        config={"system_instruction": SYSTEM_PROMPT},
    )
    return response.text.strip()
