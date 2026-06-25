import os
import json
import re
import time
from google import genai
from langsmith import traceable
from config import GEMINI_MODEL


def _call_gemini_with_retry(client, model: str, contents: str, config: dict, max_retries: int = 2) -> object:
    for attempt in range(max_retries + 1):
        try:
            return client.models.generate_content(model=model, contents=contents, config=config)
        except Exception as e:
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

SCHEMA_DESCRIPTION = """
Available columns in the loan dataset:

DATE COLUMNS (use YYYY-MM-DD format in filters):
- Ag_Date: Agreement/loan disbursement date
- Due Dt: Monthly installment due date
- Last Receipt Date: Date of last payment received

CATEGORY COLUMNS:
- curr_bucket: DPD bucket in the CURRENT month. Values: "STD", "1-30 DPD", "SMA-1", "SMA-2", "NPA", "NA"
- prev_bucket: DPD bucket in the PREVIOUS month (only present when a previous month file is uploaded). Same values as curr_bucket. Use to detect bucket movement between months.
- Loan Status: Current loan lifecycle status. Exact values (case-sensitive):
  "RUN" = loan is active and within its original tenure
  "MAT" = loan tenure is over but closing arrears still pending (matured, recovery mode)
  "S&S" = vehicle seized and sold, balance still outstanding (most severe)
- RegionName: Region name (e.g. "PUNE", "MUMBAI")
- Unit: Branch name (e.g. "MAHAD", "BELAP", "PNJIM")
- Strike: Whether the account is current on its installment (EMI) obligation this month. Y = current (at least one of: collection >= Month Due-Inst, OR LCC%=100, OR ARREARS AGAINST INST<=0). N = not current on installment. Strike is ONLY about installment — insurance/expense arrears do NOT affect it.
- CHANNEL: Loan channel
- Segment: Loan segment
- NACHStatus: NACH payment status
- CUSTOMER_STATUS: Customer classification

NUMERIC COLUMNS:
- POS: Principal Outstanding = future principal balance remaining on the loan. Can be 0 for MAT/S&S accounts.
- SOH: Sum of Hire = POS + Closing Arrears = total exposure if customer defaults. Use SOH for any "exposure", "outstanding", "at risk" queries. SOH is more accurate than POS because it includes overdue arrears.
- ClosingPC: Closing Penal Charges = accumulated penalties on the loan. Completely different from POS or SOH.
- Arrears / EMI: PRIMARY delinquency indicator. 0 = fully current (no dues). >0 = has some overdue/delinquency. >1 = SMA-1. >2 = SMA-2. >3 = NPA. Use this column for ANY query about "delinquency", "overdue", "dues", "arrears".
- LCC%: Cumulative collection efficiency = Cum Coll (Inst+Exp) / (Cum Due-Inst + Cum Due-Exp) × 100. Capped at 100. LCC% = 100 means customer has paid all cumulative installment + expense dues. LCC% < 100 means some historical dues are unpaid.
- Loan Amount: Original loan amount
- Month Receipt Amount: Total cash received this month including reserve/advance collection
- Month Collection (Excluding Reserve Collection): Effective collection that clears current demand AND pending arrears, excluding reserve (advance) payments. Reserve = excess payment beyond all dues. Use this column for Collection % and under-collection analysis.
- Net Collection Demand Inst+Exp+BC: Total monthly demand (installment + expense + bounce charges). Collection % = Month Collection (Excluding Reserve Collection) / Net Collection Demand Inst+Exp+BC * 100
- DelinquencyDays: Days past due (alternative to Arrears/EMI)

STRING COLUMNS:
- Loan No: Unique loan identifier
- Cust Name: Customer full name
- Cust Mob No: Customer mobile number
- MNT NAME / MNT CODE: Field collection executive assigned to the account
- SRC Name / SRC Code: Sourcing dealer or DSA who originated the loan
- scheme: Loan product type (e.g. New Passenger Vehicle, Used Light Commercial Vehicle)
- NACHStatus: Whether NACH (auto-debit mandate) is enabled — Y = enabled, N = not enabled. N means field collection required.
- Non Starter: Customer has not paid even their 1st EMI — Y = non-starter. CRITICAL accounts to monitor.
- LGL_FLAG: Legal proceedings ongoing — Y = yes, N = no
- LGL_DESCRIPTION: Description of legal action taken
- CUSTOMER_STATUS: Whether customer is alive or dead
- CoLending_Loans: High-priority co-lending loan — Y = yes. These should NOT go into default.
- No Coll 3 Months and >6 EMI: Y = no collection for 3+ months AND arrears exceed 6 EMIs, N = no, NA = not applicable

FLAG COLUMNS (all Y/N unless noted):
- Strike: Y = installment obligation current this month (collection>=inst due, OR LCC%=100, OR no inst arrears), N = installment not current. Insurance/expense arrears do NOT affect Strike.
- Non Starter: Y = missed first EMI ever
- NACHStatus: Y = NACH active, N = NACH inactive
- LGL_FLAG: Y = legal action filed
- CoLending_Loans: Y = co-lending loan
- No Coll 3 Months and >6 EMI: Y/N/NA

BUCKET ORDER (best to worst): STD < 1-30 DPD < SMA-1 < SMA-2 < NPA
- "roll forward" or "worsened" = account moved to a WORSE bucket (e.g. STD to 1-30 DPD, SMA-1 to SMA-2). Filter: curr_bucket is worse than prev_bucket.
- "roll backward" or "cured" or "improved" = account moved to a BETTER bucket (e.g. SMA-2 to SMA-1, NPA to STD). Filter: curr_bucket is better than prev_bucket.
- For roll forward: use condition {"column": "curr_bucket", "op": "bucket_worse_than", "value": "prev_bucket"}
- For roll backward / cured: use condition {"column": "curr_bucket", "op": "bucket_better_than", "value": "prev_bucket"}

SPECIAL INTERPRETATIONS:
- "mature cases" or "matured loans" → Loan Status == "MAT"
- "running cases" or "running loans" or "active loans" → Loan Status == "RUN"
- "seized" or "seized and sold" or "S&S cases" → Loan Status == "S&S"
- "haven't paid for 3 months" or "no collection for 3 months" or "no collection last 3 months" or "3 months no payment" → No Coll 3 Months and >6 EMI == "Y"
- "3 bucket" or "bucket 3" or "3 EMI arrears" or "arrears >= 3" or "3 or more EMI" → Arrears / EMI >= 3 (NPA threshold — do NOT use No Coll 3 Months column for this)
- CRITICAL RULE: "3 months" (time period) → No Coll 3 Months and >6 EMI column. "3 bucket/EMI" (delinquency level) → Arrears / EMI >= 3. Never mix these two.
- "overdue" or "defaulter" → curr_bucket in ["SMA-1", "SMA-2", "NPA"]
- "at risk" or "risky" → curr_bucket in ["SMA-2", "NPA"]
- "regular" or "good customer" → curr_bucket == "STD"
- "high outstanding" → sort by POS descending
- "some delinquency" or "any delinquency" or "delinquent" or "any overdue" → Arrears / EMI > 0
- "advances" or "disbursed" or "sanctioned" or "loans given" → refers to Ag_Date (agreement/disbursement date)
- "november 2025 onward" or "after november 2025" or "from november 2025" → Ag_Date >= 2025-11-01
- "non starters" or "never paid" or "missed first emi" → Non Starter == "Y"
- "no nach" or "nach inactive" or "no auto debit" → NACHStatus == "N"
- "legal accounts" or "legal action" or "under legal" → LGL_FLAG == "Y"
- "co-lending" or "co lending" or "colending" or "co-lending cases" or "colending loans" → CoLending_Loans == "Y" (all co-lending loans)
- "co-lending at risk" or "co-lending delinquent" or "co-lending with arrears" or "co-lending defaults" or "co-lending overdue" → CoLending_Loans == "Y" AND Arrears / EMI > 0. These are partner bank co-lending loans showing delinquency - highest priority, SLA breach risk.
- "dead customer" or "deceased" → CUSTOMER_STATUS == "Dead" (or similar value)
- "no strike" or "no full payment" or "payment not received" → Strike == "N"
- "strike" or "full payment received" or "fully collected" → Strike == "Y"
- "insurance cases" or "arrears due to insurance" or "insurance delinquency" or "insurance arrears" or "arrears only due to insurance" or "delinquent due to insurance" → ARREARS AGAINST INST <= 0 AND ARREARS AGAINST EXP > 5000 AND Arrears / EMI > 0. Threshold is > 5000 (not > 0) because insurance charges are typically above ₹3000 and smaller expense arrears like legal fees can be falsely flagged. Do NOT use Month Due-Inst or Month Due-Exp for this.
- "easy settlement" or "easy settlements" or "quick wins" or "small arrears" or "low arrears cases" → Closing Arrears > 0 AND Closing Arrears < 1000. These are accounts with tiny outstanding amounts that can be cleared in one call or visit.
- "no collection" or "zero collection" or "no payment this month" or "not paid this month" or "zero payers" → Month Collection (Excluding Reserve Collection) == 0 AND Net Collection Demand Inst+Exp+BC > 0. The demand check excludes accounts that genuinely had no EMI due this month.
- "short collection" or "partial collection" or "partial payment" or "partially paid" or "short payers" or "underpaid" → Month Collection (Excluding Reserve Collection) > 0 AND Month Collection (Excluding Reserve Collection) < Net Collection Demand Inst+Exp+BC. These paid something but less than the full demand.
- "under collected" or "under collection" or "both no and short collection" or "no collection and short collection" or "not fully collected" or "less than demand" or "collection shortfall" → Month Collection (Excluding Reserve Collection) < Net Collection Demand Inst+Exp+BC AND Net Collection Demand Inst+Exp+BC > 0. This covers BOTH zero payers and partial payers in one condition.
"""

SYSTEM_PROMPT = f"""You are a data analyst assistant for a loan portfolio management system.
Your job is to convert a natural language query into a structured JSON filter specification.

{SCHEMA_DESCRIPTION}

Return ONLY valid JSON (no markdown, no explanation) with this exact structure:
{{
  "conditions": [
    {{"column": "column_name", "op": "operator", "value": "value"}}
  ],
  "display_columns": ["Loan No", "Cust Name", ...],
  "sort_by": "column_name",
  "sort_asc": false,
  "plain_english": "brief restatement of what is being queried"
}}

Operators: ==, !=, >, >=, <, <=, in (value must be a list), contains, bucket_worse_than (value = another column name), bucket_better_than (value = another column name)
For "in" operator, value must be a JSON array: ["val1", "val2"]
Always include these in display_columns: Loan No, Cust Name, Cust Mob No, RegionName, Unit, ARREARS AGAINST INST, ARREARS AGAINST EXP
Add relevant columns based on the query (e.g. Ag_Date if date filter, POS if amount mentioned).
Use a single hyphen (-) when a dash is needed. Never use double dash (--), em dash (—), or en dash (–).
"""


@traceable(run_type="chain", name="QueryParser", tags=["gemini", "nbfc", "filter-generation"])
def parse_query(query: str) -> dict:
    from datetime import date as _date
    from dateutil.relativedelta import relativedelta as _rd

    api_key = os.environ.get("GOOGLE_API_KEY", "")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable not set.")

    today = _date.today()
    date_context = (
        f"[CURRENT DATE: {today.isoformat()} | "
        f"12 months ago: {(today - _rd(months=12)).isoformat()} | "
        f"6 months ago: {(today - _rd(months=6)).isoformat()} | "
        f"3 months ago: {(today - _rd(months=3)).isoformat()}] "
    )
    client = genai.Client(api_key=api_key)
    response = _call_gemini_with_retry(
        client, GEMINI_MODEL, date_context + query,
        {"system_instruction": SYSTEM_PROMPT},
    )
    _add_token_usage(response)
    raw = response.text.strip()

    # Strip markdown code fences if present
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    parsed = json.loads(raw) #string to python dict

    # Ensure required display columns are always present
    required = ["Loan No", "Cust Name", "Cust Mob No", "RegionName", "Unit"]
    for col in required:
        if col not in (parsed.get("display_columns") or []):
            parsed.setdefault("display_columns", []).insert(0, col)

    return parsed
