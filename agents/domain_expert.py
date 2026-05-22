import os
import json
import re
from google import genai

# ── Business Priority Framework ───────────────────────────────────────────────
# Defined here as ground truth — used both in the system prompt and by the
# data executor when priority_mode is active.
PRIORITY_RULES = [
    {
        "rank": 1,
        "label": "Non Starters",
        "why": "Never paid even 1st EMI — highest credit risk, possible fraud or disbursement issue",
        "conditions": [{"column": "Non Starter", "op": "==", "value": "Y"}],
    },
    {
        "rank": 2,
        "label": "Easy Settlements",
        "why": "Closing arrears < ₹1000 — one call can clear these, quick wins for collection team",
        "conditions": [
            {"column": "Closing Arrears", "op": ">",  "value": 0},
            {"column": "Closing Arrears", "op": "<",  "value": 1000},
        ],
    },
    {
        "rank": 3,
        "label": "Recent Advances — High Bucket",
        "why": "Loans sanctioned within last 12 months already in SMA-1 or worse — early warning of sourcing quality issues",
        "conditions": [
            {"column": "Ag_Date",       "op": ">=", "value": "__CUTOFF_1Y__"},
            {"column": "Arrears / EMI", "op": ">=", "value": 1},
        ],
    },
    {
        "rank": 4,
        "label": "Insurance-Driven Delinquency",
        "why": "Customer paid EMI but insurance charge is creating artificial arrears — fixable via cash or child loan",
        "conditions": [
            {"column": "Month Due-Inst", "op": "<=", "value": 0},
            {"column": "Month Due-Exp",  "op": ">",  "value": 0},
            {"column": "Arrears / EMI",  "op": ">",  "value": 0},
        ],
    },
    {
        "rank": 5,
        "label": "Co-lending at Risk",
        "why": "Partner bank co-lending loans with any delinquency — SLA breach risk",
        "conditions": [
            {"column": "CoLending_Loans", "op": "==", "value": "Y"},
            {"column": "Arrears / EMI",   "op": ">",  "value": 0},
        ],
    },
    {
        "rank": 6,
        "label": "No Collection 3 Months",
        "why": "No payment for 3+ months AND >6 EMI arrears — pre-NPA deterioration signal",
        "conditions": [{"column": "No Coll 3 Months and >6 EMI", "op": "==", "value": "Y"}],
    },
    {
        "rank": 7,
        "label": "NPA Accounts",
        "why": "Fully non-performing — requires legal/recovery escalation",
        "conditions": [{"column": "curr_bucket", "op": "==", "value": "NPA"}],
    },
]

SYSTEM_PROMPT = f"""You are a Senior Credit & Collection Risk Expert at an NBFC (Non-Banking Financial Company)
with 15+ years of experience in loan portfolio management, collections, and credit risk.

Your job is to interpret a layman or poorly worded query about a loan portfolio and convert it
into a rich, precise, domain-expert-level query that a data analyst can execute.

You deeply understand NBFC terminology:
- Advances = loans disbursed. "November 2025 advances" = loans where Ag_Date >= 2025-11-01
- Ag_Date = Agreement date = the date the loan was originally given to the customer
- Delinquency / overdue / dues = Arrears/EMI > 0 (any value above 0 means payment is overdue)
- NPA = Non-Performing Asset (Arrears/EMI >= 3 or 90+ days past due)
- SMA-1 = Special Mention Account 1 (Arrears/EMI between 1-2, stressed accounts)
- SMA-2 = Special Mention Account 2 (Arrears/EMI between 2-3, severely stressed)
- STD = Standard / current accounts (Arrears/EMI = 0, no dues)
- Hard bucket = SMA-1 + SMA-2 + NPA combined (risky accounts)
- Strike = field collection attempt this month (Y = attempted, N = not attempted)
- POS = Principal Outstanding (total remaining loan exposure)
- ClosingPC = Amount customer needs to pay RIGHT NOW to have zero arrears — key recovery KPI
- LCC% = Collection efficiency percentage
- Non Starter = Customer has NOT paid even 1st EMI (Y). CRITICAL — highest risk.
- NACHStatus = Y = NACH active, N = inactive (field collection only option)
- LGL_FLAG = Y = legal proceedings ongoing
- CoLending_Loans = Y = high-priority co-lending loan (MUST NOT default)
- No Coll 3 Months and >6 EMI = Y = no payment 3+ months AND >6 EMI arrears
- MNT NAME = Field collection executive
- SRC Name = Sourcing dealer or DSA
- CUSTOMER_STATUS = Alive or Dead
- scheme = Loan product type
- Month Due-Inst = Monthly installment demand; if <= 0 means no EMI due this month
- Month Due-Exp = Monthly expense demand (insurance etc.); if > 0 with no EMI = insurance-only delinquency
- Closing Arrears = Total arrears at month close in rupees

BUSINESS PRIORITY FRAMEWORK (apply when query is vague about what to look at):
Priority 1 — Non Starters (Non Starter == Y): Never paid 1st EMI
Priority 2 — Easy Settlements (0 < Closing Arrears < 1000): Quick wins
Priority 3 — Recent Advances at Risk (Ag_Date within 1 year AND Arrears/EMI >= 1): New portfolio deterioration
Priority 4 — Insurance-Driven Delinquency (Month Due-Inst <= 0 AND Month Due-Exp > 0 AND Arrears/EMI > 0)
Priority 5 — Co-lending at Risk (CoLending_Loans == Y AND Arrears/EMI > 0)
Priority 6 — No Collection 3 Months (No Coll 3 Months and >6 EMI == Y)
Priority 7 — NPA Accounts (curr_bucket == NPA)

VAGUE QUERY DETECTION — set priority_mode to true when the query contains phrases like:
"prioritize", "what should I focus on", "urgent cases", "action needed", "most important",
"what to work on", "cases to handle first", "where to start", "which cases first",
"top cases", "critical cases", "immediate attention", or any similar intent.

Your output must be a JSON object with this exact structure:
{{
  "enriched_query": "A precise, detailed restatement — include exact column names and conditions.",
  "query_category": "One of: delinquency_analysis | geographic_analysis | collection_performance | portfolio_quality | executive_performance | bucket_analysis | new_advances | recovery_analysis | priority_action | general",
  "query_title": "Short 5-7 word title describing the query",
  "focus_kpis": ["3-5 relevant KPI names from: Count, Total POS, Avg Arrears/EMI, Total Demand, Total Collection, Collection %"],
  "insight_focus": "What angle should AI observations focus on? 1-2 sentences.",
  "risk_flag": "high | medium | low",
  "suggested_sort": "Column and direction (e.g. 'Arrears / EMI descending')",
  "priority_mode": false
}}

Set priority_mode to true ONLY for vague/priority queries. When true, the system will
automatically apply the full business priority framework — you do NOT need to specify filters.

Return ONLY valid JSON. No markdown, no explanation outside the JSON.
"""


def enrich_query(raw_query: str) -> dict:
    api_key = os.environ.get("GOOGLE_API_KEY", "")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not set.")

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=raw_query,
        config={"system_instruction": SYSTEM_PROMPT},
    )

    raw = response.text.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    result = json.loads(raw)

    result.setdefault("query_category", "general")
    result.setdefault("query_title", "Custom Query")
    result.setdefault("focus_kpis", ["Count", "Total POS", "Avg Arrears/EMI"])
    result.setdefault("insight_focus", "Provide key risk observations and actionable recommendations.")
    result.setdefault("risk_flag", "medium")
    result.setdefault("suggested_sort", "POS descending")
    result.setdefault("priority_mode", False)

    return result
