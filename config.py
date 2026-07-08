# Single source of truth for shared configuration values.
# Import from here rather than hardcoding strings across modules.

import os

GEMINI_MODEL = "gemini-2.5-flash-lite"

# Query outcome logging: every AI Query run appends one line (timestamp, raw query
# text, outcome classification, matched view/intent, error if any) to a local
# JSONL file. Purpose: find out what real users ask that the registry vocabulary
# (registry/ontology.py CONCEPTS/METRICS, registry/views.py VIEWS) doesn't cover
# yet, so it can grow from evidence instead of guesswork. Local disk only, never
# a database, never sent anywhere -- gitignored. Query text can contain a name/
# number a user typed into their question, so this is deliberately NOT committed
# or shared, same spirit as .env.
QUERY_LOG_ENABLED = os.environ.get("COLLECTIONIQ_QUERY_LOG", "1").strip().lower() not in ("0", "false", "no", "off")
QUERY_LOG_PATH = os.environ.get("COLLECTIONIQ_QUERY_LOG_PATH", "logs/query_log.jsonl")

# Entries older than this are dropped the next time the log is pruned (query_log.py
# checks at most once/day via a sidecar marker file, so a busy log doesn't pay a
# full read+rewrite on every single query).
QUERY_LOG_RETENTION_DAYS = int(os.environ.get("COLLECTIONIQ_QUERY_LOG_RETENTION_DAYS", "30"))

#  Date validation ──────────────────────────────────────────────────────────
# Plausible range for Ag_Date / Last Receipt Date / ParentLDueDate after parsing
# (utils.py::load_and_validate). A parsed date outside this range becomes NaT
# instead of silently passing through as a wrong-but-still-a-valid-Timestamp
# value. Found in real production data: a handful of rows where a RUPEE AMOUNT
# (e.g. 150000, 400000) had ended up in the "Last Receipt Date" column of the
# source LCC extract - numerically small enough to not overflow pandas'
# Timestamp range, so it silently parsed as a real (nonsense, e.g. year 2170)
# date. That's dangerous specifically because Last Receipt Date feeds live
# business logic ("paid this month" / "unpaid this month" AI Query filters
# compare it directly against a cutoff date) - a garbage far-future date makes
# an account look paid when it isn't. 1990-2035 comfortably covers any real
# loan in this portfolio (observed Ag_Date range in production data: 2005-2026)
# with headroom for near-term future-dated schedules.
LCC_DATE_MIN_YEAR = 1990
LCC_DATE_MAX_YEAR = 2035

#  Smart Alert thresholds ────────────────────────────────────────────────────
# Tune these to adjust sensitivity without touching business logic code.

# Easy Settlements: closing arrears below this are considered quick wins
EASY_SETTLEMENT_MAX_ARREARS = 1_000          # ₹

# Insurance Delinquency: expense arrears above this with zero inst arrears = insurance-driven
INSURANCE_EXP_ARREARS_MIN = 5_000            # ₹

# High Arrears: total arrears as a fraction of original loan amount
HIGH_ARREARS_LOAN_RATIO = 0.50               # 50 %

# Recent Advances: loans sanctioned within this window that already have delinquencies
RECENT_ADVANCES_MONTHS = 12                  # months

# Executive Scorecard: executives with fewer accounts than this are excluded from rankings
SCORECARD_MIN_ACCOUNTS = 5

#  Portfolio Intelligence group-size thresholds ───────────────────────────────
# A group below its threshold is excluded from that breakdown table entirely
# (too few accounts for the %/ratio to be statistically meaningful).

# Region/Branch/Executive dimension breakdowns (NPA-SMA2 comparison, branch
# quadrant, executive recovery leaderboard)
MIN_ACCOUNTS_DIMENSION_BREAKDOWN = 3

# Vehicle segment / fuel type breakdowns. _group_npa_table keeps a group when
# n >= this value, so 11 enforces "more than 10 accounts" (strictly >10).
MIN_ACCOUNTS_PRODUCT_SEGMENT = 11

# Sourcing channel and disbursement-vintage cohort breakdowns
MIN_ACCOUNTS_SOURCE_VINTAGE = 10

# Overdue vs Month Demand Collection breakdown (Section 2b / report) -- executive
# grain only. An executive with a handful of loans can swing to 0% or 100% on a
# single account, which isn't a meaningful signal at the top/bottom of a league
# table. n >= this value, so 11 enforces "more than 10 accounts" (strictly >10).
# Region/Branch grains aren't filtered this way -- they don't suffer the same
# tiny-N volatility at realistic portfolio sizes.
MIN_ACCOUNTS_OVERDUE_DEMAND_EXECUTIVE = 11

# New Advances (Business tab) trend chart -- how many trailing months to plot
# by default. User-adjustable in the UI (dropdown, see
# NEW_ADVANCES_TREND_MONTH_OPTIONS); this is only the pre-selected default.
NEW_ADVANCES_TREND_DEFAULT_MONTHS = 24
NEW_ADVANCES_TREND_MONTH_OPTIONS = [6, 12, 24, 36, 60, "All"]

# The monthly report is a static, non-interactive document (no dropdown to
# pick a window), so its own New Advances Trend chart uses a fixed window --
# deliberately NOT reusing NEW_ADVANCES_TREND_DEFAULT_MONTHS, since the report
# and dashboard defaults are allowed to diverge for good reason (a printed
# report favors a longer, more complete trend; the dashboard default favors a
# faster first render).
NEW_ADVANCES_REPORT_TREND_MONTHS = 36

# Report top-N cap for the New Advances by Region/Branch/Executive section --
# same "printed document, not a scrollable table" reasoning as
# branch_performance.py's own top5/bottom5 cap.
NEW_ADVANCES_REPORT_TOP_N = 5

# SegmentName/Segment values are sometimes truncated inconsistently by the
# source system at DIFFERENT lengths for the SAME real segment (observed in
# real production data: "Passenger Commerc" / "Passenger Commerci" /
# "Passenger Commercial" all the same segment, split into 3 separate rows in
# every NPA/SOH breakdown instead of one). Two values whose first this-many
# characters match are treated as the same segment and merged under whichever
# variant is longest (the most complete-looking name available, since there's
# no canonical enum to match against). 15 was verified against real data to
# merge exactly the truncation clusters present, with zero false merges
# against any of the ~25 other distinct real segment names -- a deliberate,
# evidence-based choice, not an arbitrary one, but still a heuristic: two
# genuinely different segments sharing the same first 15 characters would be
# wrongly merged (accepted risk, given the alternative is a hardcoded alias
# list that needs maintaining every time a new truncation length appears).
SEGMENT_NAME_PREFIX_MATCH_CHARS = 15

#  Portfolio Intelligence business-rule thresholds ────────────────────────────

# "Hard Bucket": accounts this many EMIs or more overdue - a narrower, more
# severe signal than NPA (NPA starts at 3 EMI overdue; Hard Bucket is deeper).
HARD_BUCKET_ARREARS_EMI_MIN = 6

# Repossession Priority List: eligible accounts must have been sanctioned
# within this many months (older loans have less collateral value left)
REPOSSESSION_WINDOW_MONTHS = 18

# Portfolio Pulse "NOV'25 Onward Delinquency" KPI: a fixed cohort-start date
# (NOT a rolling window relative to the report month, unlike
# REPOSSESSION_WINDOW_MONTHS above or RECENT_ADVANCES_MONTHS below -- both of
# those are rolling windows relative to today/report month; this is a fixed
# calendar date) -- a management change occurred on this date, and this
# tracks delinquency quality specifically among originations since then.
# Deliberately time-scoped: this cohort only grows over time, so its
# usefulness as a tight "how's the new regime underwriting" signal will
# naturally fade -- expect to eventually retire or redefine it.
RECENT_ADVANCES_COHORT_START = "2025-11-01"

# Good Customers (refinance/relationship candidates): must have completed at
# least this % of their loan tenure...
GOOD_CUSTOMER_MIN_TENURE_PCT = 70
# ...AND have collected at least this % of everything ever due (no lifetime shortfall)
GOOD_CUSTOMER_MIN_LCC_PCT = 100

# Fleet Operator Exposure: customers with at least this many loans are
# treated as a fleet operator rather than an individual borrower
FLEET_MIN_LOANS = 3

# Region Scorecard status label: NPA% move larger than this (in percentage
# points) is labelled Worsening/Improving; smaller moves are "Stable"
REGION_STATUS_DELTA_PP = 1.0

# Good vs Bad summary: a region's NPA% move must exceed this (pp) to be
# called out as a notable improvement/concern
GOOD_BAD_REGION_DELTA_PP = 0.5

# Good vs Bad summary: minimum branch Concern Score to flag as a top concern /
# minimum-or-below to flag as the healthiest branch
CONCERN_SCORE_BAD_THRESHOLD = 60
CONCERN_SCORE_GOOD_THRESHOLD = 35

# Branch Concern Score component weights - must sum to 1.0
CONCERN_SCORE_WEIGHTS = {"NPA%": 0.45, "Hard Bucket%": 0.25, "Roll Fwd%": 0.2, "Chronic (3M+)": 0.1}

# Risk Indicators (Is the Risk Profile Changing?): a % indicator moving less
# than this (pp) is "Stable" rather than Improving/Worsening
RISK_INDICATOR_STABLE_PP = 0.2

# Good vs Bad summary: minimum move for a risk indicator to be worth
# mentioning - percentage-point indicators use the pp value, count-based
# indicators (e.g. "+3 accounts") use the count value
RISK_INDICATOR_MATERIALITY_PP = 0.3
RISK_INDICATOR_MATERIALITY_COUNT = 1

# Vintage chart (NPA %/SMA-2 % by disbursement cohort): marker color and
# "Critical"/"Watch" reference-band cutoffs. The label text in
# build_vintage_chart reads these same constants, so a retuned threshold
# can't silently go stale in the chart's own annotation.
VINTAGE_CHART_CRITICAL_PCT = 10
VINTAGE_CHART_WATCH_PCT = 5
