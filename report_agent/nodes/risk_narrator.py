import os
import time
from google import genai
from langsmith import traceable
from config import GEMINI_MODEL
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

    nm = section_data.get("npa_sma2_movement")
    if nm and nm.get("portfolio"):
        p = nm["portfolio"]
        parts.append("\nNPA & SMA-2 MOVEMENT (this month vs last month):")
        parts.append(f"  NPA: {p['npa_current']} (was {p['npa_prev']}), change {p['npa_delta']} ({p['npa_pct_change']}%)")
        parts.append(f"  SMA-2: {p['sma2_current']} (was {p['sma2_prev']}), change {p['sma2_delta']} ({p['sma2_pct_change']}%)")

    vd = section_data.get("verdict")
    if vd:
        parts.append("\nGOOD / BAD SIGNALS (pandas-computed):")
        for g in vd.get("good", []):
            parts.append(f"  [GOOD] {g}")
        for b in vd.get("bad", []):
            parts.append(f"  [BAD] {b}")

    ta = section_data.get("top_accounts")
    if ta:
        s = ta.get("summary", {})
        parts.append(f"\nTOP AT-RISK ACCOUNTS: top {ta.get('n', 0)} delinquent accounts by SOH total {_fmt_money(s.get('total_soh_cr', 0) * 1e7)} "
                     f"({s.get('pct_of_portfolio', 0)}% of portfolio SOH), {s.get('npa_count', 0)} already NPA.")

    fe = section_data.get("fleet_exposure")
    if fe:
        parts.append(f"\nFLEET EXPOSURE: {fe.get('count', 0)} fleet operators (3+ loans), "
                     f"{_fmt_money(fe.get('total_soh_cr', 0) * 1e7)} exposure, {fe.get('npa_operators', 0)} operators with an NPA loan.")

    ri = section_data.get("risk_indicators")
    if ri:
        parts.append("\nRISK INDICATORS (early warning):")
        for ind in ri.get("indicators", []):
            parts.append(f"  {ind['Signal']}: {ind['This Month']} ({ind['Direction']}) - {ind['Note']}")

    bq = section_data.get("branch_quadrant")
    if bq:
        parts.append("\nBRANCH QUADRANT (highest concern):")
        for c in bq.get("top_concern", []):
            parts.append(f"  {c['Branch']}: concern score {c['Concern Score']}, Coll {c['Collection%']}%, NPA {c['NPA%']}%")

    pa = section_data.get("product_analysis")
    if pa:
        parts.append("\nSEGMENT-WISE NPA:")
        for r in pa.get("rows", [])[:5]:
            parts.append(f"  {r.get('Segment', '')}: NPA {r.get('NPA%', 0)}%, Collection {r.get('Collection%', 0)}%")

    rp = section_data.get("repossession")
    if rp:
        parts.append(f"\nREPOSSESSION CANDIDATES: {rp.get('total', 0)} accounts eligible (SMA-2/NPA, recent loans).")

    gc = section_data.get("good_customers")
    if gc:
        parts.append(f"\nGOOD CUSTOMERS: {gc.get('total', 0)} loyal customers eligible for refinance/retention outreach.")

    er_recovery = section_data.get("executive_recovery")
    if er_recovery:
        parts.append("\nEXECUTIVE RECOVERY LEADERBOARD:")
        for r in er_recovery.get("top", [])[:3]:
            parts.append(f"  {r['Executive']}: net {r['Net Recovery']} accounts recovered (Rescued {r['Rescued']}, Slipped {r['Slipped']})")
        for r in er_recovery.get("bottom", [])[:3]:
            parts.append(f"  {r['Executive']}: net {r['Net Recovery']} accounts deteriorated (Rescued {r['Rescued']}, Slipped {r['Slipped']})")

    return "\n".join(parts)


NARRATIVE_PROMPT = """You are the Chief Risk Officer of Shriram Finance preparing a monthly portfolio briefing for the Regional Director.
Write exactly 6 to 8 bullet points in plain text. Each bullet must start with a hyphen and a space (- ).

Cover these areas across the bullets (not as headers, just as content):
- Overall collection performance with key numbers
- Key wins this month with specific figures
- Major risk concerns with specific branch/bucket/executive names
- NPA and delinquency observations
- Bucket migration highlights (if data available)
- Concentration risk (top at-risk accounts, fleet operator exposure) if data available
- What needs priority attention and why

Rules:
- Each bullet is one concise sentence (max 20 words)
- Use specific numbers, names, and percentages from the data
- No markdown, no paragraphs, no headers, no em dashes, no double dashes
- Use single hyphen (-) at the start of each bullet only
- Professional, direct NBFC tone"""

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
    _empty = {**state, "executive_narrative": "", "action_plan": "", "ai_skipped": True}

    api_key = os.environ.get("GOOGLE_API_KEY", "")
    if not api_key:
        return _empty

    sd = state.get("section_data")
    if not sd:
        return _empty

    prompt = _build_prompt(sd, state["curr_month"])
    client = genai.Client(api_key=api_key)

    try:
        resp = _call_gemini_with_retry(client, GEMINI_MODEL, prompt, {"system_instruction": NARRATIVE_PROMPT})
        _add_token_usage(resp)
        narrative = resp.text.strip()

        resp2 = _call_gemini_with_retry(client, GEMINI_MODEL, prompt, {"system_instruction": ACTION_PROMPT})
        _add_token_usage(resp2)
        action_plan = resp2.text.strip()
    except Exception:
        return _empty

    return {**state, "executive_narrative": narrative, "action_plan": action_plan, "ai_skipped": False}
