import os
import time
from google import genai
from langsmith import traceable
from report_agent.state import ReportState


def _call_gemini_with_retry(client, model, contents, config, max_retries=2):
    for attempt in range(max_retries + 1):
        try:
            return client.models.generate_content(model=model, contents=contents, config=config)
        except Exception:
            if attempt == max_retries:
                raise
            time.sleep(2 ** attempt)


def _add_token_usage(response) -> None:
    try:
        from langsmith.run_helpers import get_current_run_tree
        rt = get_current_run_tree()
        if rt is None:
            return
        um = getattr(response, "usage_metadata", None)
        if um:
            rt.add_metadata({
                "input_tokens":  int(getattr(um, "prompt_token_count",     0) or 0),
                "output_tokens": int(getattr(um, "candidates_token_count", 0) or 0),
                "total_tokens":  int(getattr(um, "total_token_count",      0) or 0),
            })
    except Exception:
        pass


def _fmt_money(val):
    if abs(val) >= 1_00_00_000:
        return f"₹{val/1_00_00_000:.2f}Cr"
    if abs(val) >= 1_00_000:
        return f"₹{val/1_00_000:.2f}L"
    return f"₹{val:,.0f}"


def _build_prompt(section_data: dict, curr_month: str) -> str:
    parts = [f"Portfolio reporting month: {curr_month}\n"]

    ph = section_data.get("portfolio_health")
    if ph:
        kpis = ph.get("kpis", {})
        parts.append("PORTFOLIO HEALTH SNAPSHOT:")
        for k, v in kpis.items():
            parts.append(f"  {k}: {v['formatted']} (MoM {v['mom']:+.1f}%, traffic: {v['traffic']})")

    rf = section_data.get("risk_flags")
    if rf:
        parts.append("\nCRITICAL RISK FLAGS:")
        for f in rf.get("flags", []):
            parts.append(f"  [{f['severity'].upper()}] {f['title']}: {f['count']} accounts, POS {_fmt_money(f['pos'])}")
            parts.append(f"    Action: {f['action']}")

    bm = section_data.get("bucket_migration")
    if bm:
        parts.append("\nBUCKET MIGRATION (prev month to curr month):")
        parts.append(f"  Roll-forward rate (worsened): {bm['roll_forward_rate']}%")
        parts.append(f"  Roll-backward rate (returned to STD): {bm['roll_backward_rate']}%")
        parts.append(f"  NPA formation rate:            {bm['npa_formation_rate']}%")
        parts.append(f"  Matched accounts:              {bm['matched_count']:,}")

    bp = section_data.get("branch_performance")
    if bp:
        parts.append("\nBRANCH PERFORMANCE:")
        parts.append("  Top branches:")
        for b in bp.get("top5", []):
            parts.append(f"    {b['branch']}: {b['coll_pct']}% collection ({b['accounts']} accounts)")
        parts.append("  Bottom branches:")
        for b in bp.get("bottom5", []):
            parts.append(f"    {b['branch']}: {b['coll_pct']}% collection ({b['accounts']} accounts)")

    er = section_data.get("executive_rankings")
    if er:
        parts.append("\nFIELD EXECUTIVE RANKINGS:")
        parts.append("  Top performers:")
        for e in er.get("top5", []):
            parts.append(f"    {e['name']}: {e['coll_pct']}% collection, {e['strike_rate']}% strike rate")
        parts.append("  Bottom performers:")
        for e in er.get("bottom5", []):
            parts.append(f"    {e['name']}: {e['coll_pct']}% collection, {e['strike_rate']}% strike rate")

    return "\n".join(parts)


NARRATIVE_PROMPT = """You are the Chief Risk Officer of Shriram Finance preparing a monthly portfolio briefing for the Regional Director.
Write exactly 3 paragraphs in plain text (no bullet points, no markdown, no em dashes, no double dashes):

Paragraph 1: Portfolio health summary - overall collection performance, key wins, major concerns with specific numbers.
Paragraph 2: Risk concentration - where is the most risk (geography, bucket, executive) with specific names and figures.
Paragraph 3: Forward-looking - what needs priority attention next month and why.

Be precise. Use Indian NBFC terminology. Professional, direct tone. No hedging. Use single hyphen (-) only, never double dash (--)."""

ACTION_PROMPT = """You are a collections strategy consultant for Shriram Finance.
Based on the portfolio data provided, generate exactly 5 numbered action items for the collection team this month.

Format each item exactly as:
N. [Specific action with branch/executive name if available] - Owner: [Role] - Timeline: [When]

Rules:
- Be specific (name the branch, bucket, or executive where data is available)
- Owner must be one of: Branch Manager, Field Executive, Regional Manager, Collection Head
- Timeline must be one of: Immediate (this week), By month-end, Next 48 hours
- No markdown, no em dashes, no double dashes
- Use ' - ' (space hyphen space) as the separator"""


@traceable(run_type="chain", name="RiskNarrator", tags=["gemini", "nbfc", "report-generation"])
def risk_narrator_node(state: ReportState) -> ReportState:
    api_key = os.environ.get("GOOGLE_API_KEY", "")
    if not api_key:
        return {
            **state,
            "executive_narrative": "AI narrative unavailable - GOOGLE_API_KEY not configured.",
            "action_plan": "Action plan unavailable.",
        }

    sd = state["section_data"]
    if not sd:
        return {
            **state,
            "executive_narrative": "Insufficient data to generate narrative.",
            "action_plan": "Insufficient data to generate action plan.",
        }

    prompt = _build_prompt(sd, state["curr_month"])
    client = genai.Client(api_key=api_key)

    try:
        resp = _call_gemini_with_retry(client, "gemini-2.0-flash", prompt, {"system_instruction": NARRATIVE_PROMPT})
        _add_token_usage(resp)
        narrative = resp.text.strip()
    except Exception as e:
        narrative = f"Narrative generation failed: {e}"

    try:
        resp2 = _call_gemini_with_retry(client, "gemini-2.0-flash", prompt, {"system_instruction": ACTION_PROMPT})
        _add_token_usage(resp2)
        action_plan = resp2.text.strip()
    except Exception as e:
        action_plan = f"Action plan generation failed: {e}"

    return {**state, "executive_narrative": narrative, "action_plan": action_plan}
