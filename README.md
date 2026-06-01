# CollectionIQ

### AI-Powered Portfolio Intelligence for NBFC Collection Leaders

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-FF4B4B?logo=streamlit&logoColor=white)
![Gemini](https://img.shields.io/badge/Google%20Gemini-2.0%20Flash-4285F4?logo=google&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-0.2+-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

&nbsp;

## The Problem

In a large NBFC, a Regional Business Head or Zonal Head manages thousands of loan accounts across dozens of branches and field executives. Every morning, the same questions come up:

> *"Which accounts should my team prioritize today?"*
> *"Which executive is underperforming and why?"*
> *"How many accounts from last year's advances have gone delinquent?"*
> *"Show me customers who haven't paid in 3 months in the Western region."*

Getting answers meant raising a request to an analyst, waiting for a report, then asking a follow-up. The cycle repeated for every business question, every day.

Leaders were dependent on coordinators and analysts for information that should have been at their fingertips.

&nbsp;

## What CollectionIQ Does

CollectionIQ is a self-serve portfolio intelligence dashboard built for collection leaders. Upload your monthly LCC Excel extract and the entire portfolio becomes queryable, visual, and explainable, without writing a single formula or waiting for a report.

It answers questions in plain English, surfaces risks automatically, and generates board-ready analysis on demand.

&nbsp;

**Live Link : [collectioniq.streamlit.app](https://collectioniq.streamlit.app/)**

&nbsp;

## Try It in 30 Seconds

No data? No setup? Click **Fill Sample Data** on the landing page, it loads a real 1000-loan LCC extract directly from this repo and builds the full dashboard instantly. No file upload needed.

&nbsp;

## Screenshots

**Landing Page - Upload or fill sample data instantly**

![Landing Page](docs/screenshots/01-landing.png)

&nbsp;

**KPI Dashboard - Portfolio health at a glance with MoM movement**

![KPI Dashboard](docs/screenshots/02-kpi-dashboard.png)

&nbsp;

**Portfolio Analysis - DPD bucket distribution, branch collection %, arrears exposure**

![Portfolio Analysis](docs/screenshots/03-portfolio-analysis.png)

&nbsp;

<table>
<tr>
<td width="50%">

**Executive Scorecard - Every field executive ranked by collection %, strike rate and roll rates**

![Executive Scorecard](docs/screenshots/04-executive-scorecard.png)

</td>
<td width="50%">

**Smart Alerts - Automatic risk flags with POS exposure and recommended actions**

![Smart Alerts](docs/screenshots/05-smart-alerts.png)

</td>
</tr>
<tr>
<td width="50%">

**AI Query Assistant - Plain English queries powered by Gemini + LangGraph**

![AI Query](docs/screenshots/06-ai-query.png)

</td>
<td width="50%">

**AI Query Result - KPIs, rankings, and domain-aware observations**

![AI Query Result](docs/screenshots/07-ai-query-result.png)

</td>
</tr>
</table>

&nbsp;

**Filtered Customer Table + AI Observations**

![Customer Table](docs/screenshots/08-customer-table.png)

&nbsp;



## Who It Is Built For

| Role | How They Use It |
|---|---|
| Regional Business Head | Monitor region-wide collection efficiency, bucket migration, and executive rankings |
| Zonal Head | Compare branch performance, identify concentration risk, track NPA formation |
| Business Unit Head | Query specific customer segments, get prioritized action lists, analyse new advances |

&nbsp;

## Core Capabilities

**Plain English Query Engine**

Ask questions the way you think them. The system understands NBFC domain language - "show me MAT accounts with no collection in last 1 year", "rank executives by strike rate", "how many advances from last 6 months are already in SMA-2" and returns the right result, whether that is a filtered customer table, a ranked executive comparison, or a single summary answer. Every result includes AI-generated observations and a download to Excel.

**Automated Priority Action List**

A seven-tier business priority framework identifies accounts that need immediate attention non-starters, easy settlements, recent advances already delinquent, insurance-driven arrears, co-lending at risk, long-term non-payers, and NPA accounts. Leaders get a ready-to-act list sorted by business impact, not just EMI count.

**Field Executive Performance Scorecard**

Every field executive is ranked by collection efficiency, strike rate, NPA percentage, and bucket roll rates. Top performers and underperformers are identified dynamically using quartile analysis so rankings are always relative to the current portfolio, not fixed benchmarks.

**Bucket Migration Analysis**

When two months of data are uploaded, the system shows exactly how accounts moved between DPD buckets, how many cured, how many worsened, and the NPA formation rate. This is the early warning signal that tells a leader whether the portfolio is improving or deteriorating before the numbers become a crisis.

Roll forward and roll backward queries are also supported through the plain English engine. Ask "show me accounts that worsened this month" or "which accounts cured from SMA-1" and get the filtered list instantly.

**Smart Risk Alerts**

Automatic alerts fire for non-starters, insurance-driven delinquency, easy settlements, recent advances at risk, and co-lending loans showing arrears. Each alert includes account count, outstanding exposure, and a recommended action, with a direct Excel download of the flagged accounts.

**Monthly Portfolio Intelligence Report**

A board-ready HTML report with AI-written executive narrative, branch performance league tables, executive rankings, risk flags, and a five-point prioritized action plan. Download it or send it by email directly from the dashboard.

&nbsp;

## The Impact

Before CollectionIQ, a leader needed to raise a request, wait for an analyst to pull data, and then ask a follow-up for any change in filter or angle. Each loop took hours to a day.

With CollectionIQ, the same question is answered in under 30 seconds — directly by the leader, without involving anyone else.

**What this eliminates:**
- Dependency on analysts and coordinators for portfolio queries
- Manual Excel work for bucket-level or executive-level breakdowns
- Delays in identifying which accounts to target for collection
- Subjective prioritization replaced by a structured, data-driven action framework
- Waiting for month-end reports to understand portfolio health

**What this enables:**
- Leaders making faster, evidence-based collection decisions
- Field executives held accountable through transparent scorecards
- Early detection of portfolio stress through bucket migration tracking
- Consistent prioritization logic applied across all regions and branches

&nbsp;

## Architecture

CollectionIQ runs two independent AI pipelines orchestrated with LangGraph — one for answering queries in real time, one for generating the monthly portfolio report.

&nbsp;

### Query Pipeline

Every question typed in plain English passes through four agents in sequence before a result appears on screen.

```mermaid
flowchart TD
    User(["Plain English Query\ne.g. rank executives by MAT/RUN ratio"])

    User --> DE

    subgraph QP ["  Query Pipeline  (LangGraph)  "]
        direction TB
        DE["Domain Expert Agent\nGemini 2.0 Flash\n\nUnderstands NBFC terminology · Maps intent to columns\nDecides result shape · Detects priority mode\nInjects today's date for relative time filters"]
        PP["Query Parser Agent\nGemini 2.0 Flash\n\nTranslates enriched query into structured filter spec\nHandles conditions · GROUP BY · HAVING · sort"]
        EX["Data Executor\nPandas\n\nApplies row filters · Aggregations · Priority rules\nComputes KPIs and executive rankings"]
        IG["Insight Generator Agent\nGemini 2.0 Flash\n\nReads computed KPIs and rankings\nGenerates domain-aware observations and recommendations"]
        DE --> PP --> EX --> IG
    end

    IG --> R1["Loan Table\nFiltered customer records"]
    IG --> R2["Ranked Table\nOne row per executive / branch / region"]
    IG --> R3["Single Stat\nDirect answer with supporting context"]
```

&nbsp;

### Report Pipeline

Triggered on demand. Runs fully autonomously — no user input needed after clicking Generate.

```mermaid
flowchart TD
    Trigger(["Generate Monthly Report"])

    Trigger --> PA

    subgraph RP ["  Report Pipeline  (LangGraph)  "]
        direction TB
        PA["Portfolio Analyzer\nPandas\n\nComputes all five report sections in parallel\nHealth snapshot · Risk flags · Bucket migration\nBranch performance · Executive rankings"]
        RN["Risk Narrator\nGemini 2.0 Flash\n\nWrites 3-paragraph board-level executive narrative\nGenerates 5 prioritized action items with owner and timeline"]
        RB["Report Builder\nPython\n\nAssembles fully self-contained HTML report\nNo external CSS · No CDN · Email-safe"]
        ED["Email Dispatcher\nSMTP\n\nSends report as body and attachment\nFires only if SMTP is configured in .env"]
        PA --> RN --> RB --> ED
    end

    RB --> DL["Download HTML Report"]
    ED --> EM["Email to Configured Recipients"]
```

&nbsp;

### Data Layer

Both pipelines operate on the same in-memory DataFrame loaded from the Excel upload. No database, no cloud storage. Data never leaves the machine.

```mermaid
flowchart LR
    XL["LCC Excel File\nMonthly extract"] --> VAL["Validation\nSchema check · Type coercion"]
    VAL --> BK["Bucketing\nDPD bucket assignment\nSTD · 1-30 · SMA-1 · SMA-2 · NPA"]
    BK --> KPI["KPI Computation\nCollection % · POS · Arrears · MoM delta"]
    KPI --> DF[("In-Memory\nDataFrame")]
    DF --> QP2["Query Pipeline"]
    DF --> RP2["Report Pipeline"]
    DF --> DB["Dashboard\nKPIs · Charts · Alerts · Scorecard"]
```

&nbsp;

## Project Structure

```
CollectionIQ/
├── app.py                          # Main Streamlit app — all UI layout and state
├── graph.py                        # AI query pipeline (LangGraph state machine)
├── utils.py                        # Data loading, metrics, charts, HTML export
├── smart_alerts.py                 # 5 rule-based risk alerts (pure pandas, no LLM)
│
├── agents/
│   ├── domain_expert.py            # Agent 0 — query enrichment + intent detection
│   ├── query_parser.py             # Agent 1 — natural language to filter spec
│   ├── data_executor.py            # Agent 2 — pandas filter / aggregation / priority
│   └── insight_generator.py        # Agent 3 — AI observations on query results
│
├── analysis/
│   ├── executive_scorecard.py      # Per-executive KPIs with quartile tier ranking
│   └── roll_rate.py                # Bucket migration matrix and roll-rate KPIs
│
├── report_agent/
│   ├── graph.py                    # Report pipeline (LangGraph)
│   └── nodes/
│       ├── portfolio_analyzer.py   # Computes all report sections
│       ├── risk_narrator.py        # AI executive narrative + action plan
│       ├── report_builder.py       # Assembles self-contained HTML report
│       └── email_dispatcher.py     # SMTP delivery
│
├── sample_data/
│   ├── Current_Month_Demo.xlsx     # Sample LCC extract — current month
│   └── Previous_Month_Demo.xlsx    # Sample LCC extract — previous month
│
└── requirements.txt
```

&nbsp;

## Getting Started

### Prerequisites

- Python 3.10 or higher
- A [Google AI Studio](https://aistudio.google.com/app/apikey) API key (free tier works)

&nbsp;

### 1 — Clone the repository

```bash
git clone https://github.com/Sanjay-00/CollectionIQ.git
cd CollectionIQ
```

&nbsp;

### 2 — Create a virtual environment and install dependencies

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
```

&nbsp;

### 3 — Set up environment variables

Create a `.env` file in the project root:

```env
# Required — AI query engine and report generation
GOOGLE_API_KEY=your_google_api_key_here

# Optional — LangSmith observability (query tracing + feedback)
LANGSMITH_API_KEY=your_langsmith_key
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=CollectionIQ

# Optional — Email delivery for monthly reports
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your_email@gmail.com
SMTP_PASS=your_app_password
```

> KPIs, charts, smart alerts, and executive scorecard work without any API key. Only the AI query engine and monthly report require `GOOGLE_API_KEY`.

&nbsp;

### 4  Run the app

```bash
streamlit run app.py
```

Opens at **http://localhost:8501**

&nbsp;

### 5  Load data

**Option A - Fill Sample Data (instant)**
Click the **Fill Sample Data** button on the landing page. It fetches a real LCC extract from this repo and loads the full dashboard in seconds, no file needed.

**Option B - Upload your own LCC extract**
Upload your `.xlsx` or `.xls` file using the file uploader. Upload a second file for the previous month to enable bucket migration and roll-rate analysis.

&nbsp;

## Technology

| Layer | Technology | Role |
|---|---|---|
| UI and Dashboard | Streamlit | Interactive web interface, session state, file upload |
| AI Models | Google Gemini 2.0 Flash | All four AI agents across both pipelines |
| Agent Orchestration | LangGraph | Stateful multi-agent graph with conditional routing |
| Data Processing | Pandas | Filtering, aggregation, bucketing, KPI computation |
| Charts | Plotly | DPD distribution, bucket migration heatmap, branch charts |
| AI SDK | google-genai | Gemini API with retry and exponential backoff |
| Report Delivery | Python smtplib | SMTP email with HTML body and attachment |
| Observability | LangSmith | Query tracing and result quality feedback |
| Date Handling | python-dateutil | Relative date resolution for time-based queries |

The domain knowledge layer: NBFC terminology, loan status values, priority framework, is embedded in the agent system prompts. The AI understands the difference between a RUN account, a MAT account, and an S&S account without any fine-tuning. Business context is injected at query time, making it straightforward to extend with new domain rules.

Both pipelines are stateless between runs. Each query or report generation starts fresh, no stale context, no memory leak, no shared state between users.

&nbsp;
