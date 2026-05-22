import os
import json
import re
from google import genai

SCHEMA_DESCRIPTION = """
Available columns in the loan dataset:

DATE COLUMNS (use YYYY-MM-DD format in filters):
- Ag_Date: Agreement/loan disbursement date
- Due Dt: Monthly installment due date
- Last Receipt Date: Date of last payment received

CATEGORY COLUMNS:
- curr_bucket: DPD bucket derived from Arrears/EMI ratio
  Values: "STD", "1-30 DPD", "SMA-1", "SMA-2", "NPA", "NA"
- Loan Status: Current loan status (e.g. "RUN", "CLOSED", "NPA")
- RegionName: Region name (e.g. "PUNE", "MUMBAI")
- Unit: Branch name (e.g. "MAHAD", "BELAP", "PNJIM")
- Strike: Whether collection was attempted this month ("Y" or "N")
- CHANNEL: Loan channel
- Segment: Loan segment
- NACHStatus: NACH payment status
- CUSTOMER_STATUS: Customer classification

NUMERIC COLUMNS:
- POS: Principal outstanding amount
- Arrears / EMI: PRIMARY delinquency indicator. 0 = fully current (no dues). >0 = has some overdue/delinquency. >1 = SMA-1. >2 = SMA-2. >3 = NPA. Use this column for ANY query about "delinquency", "overdue", "dues", "arrears".
- LCC%: Collection efficiency percentage
- Loan Amount: Original loan amount
- Month Receipt Amount: Amount collected this month
- NET Collection Demand Inst+Exp: Monthly demand
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
- Strike: Y = field collection attempted this month, N = not attempted (NA excluded)
- Non Starter: Y = missed first EMI ever
- NACHStatus: Y = NACH active, N = NACH inactive
- LGL_FLAG: Y = legal action filed
- CoLending_Loans: Y = co-lending loan
- No Coll 3 Months and >6 EMI: Y/N/NA

SPECIAL INTERPRETATIONS:
- "haven't paid for 3 months" or "no collection for 3 months" → No Coll 3 Months and >6 EMI == "Y"
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
- "co-lending" or "co lending" or "colending" → CoLending_Loans == "Y"
- "dead customer" or "deceased" → CUSTOMER_STATUS == "Dead" (or similar value)
- "field not visited" or "no strike" → Strike == "N"
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

Operators: ==, !=, >, >=, <, <=, in (value must be a list), contains
For "in" operator, value must be a JSON array: ["val1", "val2"]
Always include these in display_columns: Loan No, Cust Name, Cust Mob No, RegionName, Unit
Add relevant columns based on the query (e.g. Ag_Date if date filter, POS if amount mentioned).
Never use em dash, en dash, or hyphen as a dash anywhere in your output.
"""


def parse_query(query: str) -> dict:
    api_key = os.environ.get("GOOGLE_API_KEY", "")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable not set.")

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=query,
        config={"system_instruction": SYSTEM_PROMPT},
    )
    raw = response.text.strip()

    # Strip markdown code fences if present
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    parsed = json.loads(raw)

    # Ensure required display columns are always present
    required = ["Loan No", "Cust Name", "Cust Mob No", "RegionName", "Unit"]
    for col in required:
        if col not in parsed.get("display_columns", []):
            parsed.setdefault("display_columns", []).insert(0, col)

    return parsed
