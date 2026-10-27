# Single source of truth for shared configuration values.
# Import from here rather than hardcoding strings across modules.

import os

GEMINI_MODEL = "gemini-2.5-flash-lite"

# The Logical Planner (agents/logical_planner.py::plan_logical) does structured
# classification/extraction against a fixed vocabulary, not open-ended writing --
# the SDK default temperature (~1.0, tuned for creative generation) was the
# confirmed, reproducible cause of run-to-run inconsistency observed live: the
# IDENTICAL query sometimes correctly asked a grain-aware clarification
# question, sometimes silently guessed an interpretation, sometimes misrouted
# to the out-of-scope fallback. Lowered, not zeroed: 0.0 (fully greedy) risks
# the model deterministically repeating a wrong answer with no chance to
# recover, including on the compiler's own repair retry. Does not apply to the
# Insight Generator (agents/insight_generator.py) -- that IS open-ended bullet-
# point writing, where some variety is fine and arguably desirable.
PLANNER_TEMPERATURE = 0.1

# AI Query plans kept for reuse across sessions (agents/plan_cache.py). A plan
# is a small dict; 256 covers a team's repeated questions many times over.
PLAN_CACHE_MAX = 256

# Longest free-text question either chat surface (AI Query, Investigator)
# will send to Gemini -- a cost/abuse guardrail; a real portfolio question
# never needs more. One constant so the two surfaces can't drift apart.
MAX_QUERY_CHARS = 1000

# Per-request Gemini timeout (gemini_client.make_client). Without one, a hung
# connection blocked a user's query indefinitely. Flash-Lite answers in a few
# seconds; 60s leaves room for the longer report-narrative prompts.
GEMINI_TIMEOUT_MS = 60_000

# Calls the user waits on with nothing on screen (AI Query planner,
# Investigator router) send one identical backup request if the first hasn't
# answered by then (gemini_client.call_gemini_hedged). Measured 2026-10-27 on
# 40 planner calls: half answer by about 2.3s and 9 in 10 by about 3.5s, but
# the same question can take 7.5s by chance. 0 turns hedging off.
GEMINI_HEDGE_AFTER_S = 3.5

# Investigator session caps -- threads and remembered result tables live in
# st.session_state for the whole browser session, and each table can be
# thousands of rows. Oldest (least recently used) entries are evicted first.
INVESTIGATOR_MAX_THREADS = 10
INVESTIGATOR_MAX_ENTITIES = 30

# Raw LCC columns that identify a customer or guarantor. Must never reach a
# Gemini prompt or a LangSmith trace in a multi-customer result. Shared by
# graph.py (AI Query) and investigator/llm.py so the two lists can't diverge.
PII_COLUMNS = ("Cust Name", "Cust Mob No", "Guar Name", "Guar Mob No")

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

# Date columns. These are always converted to dates on upload (Excel serial
# numbers like 46236, text like 15/08/2026, or real dates -> one date type)...
KNOWN_DATE_COLUMNS = ("Ag_Date", "Last Receipt Date", "ParentLDueDate")
# ...and so is any OTHER column whose name contains the word "date" or ends in
# "DT" (e.g. "Last Received Date", "NPA_DATE", "AG DATE", "NPA DT"), unless
# it's listed here. "Due Dt" is the EMI due DAY (5, 10, 15...), not a date.
NOT_DATE_COLUMNS = ("Due Dt",)
# A column found by name is converted only if at least this share of its
# non-blank values are real dates; otherwise it's left as-is (a mislabelled
# column is never blanked) and a warning is shown.
DATE_COLUMN_MIN_PARSED_SHARE = 0.5

#  Smart Alert thresholds ────────────────────────────────────────────────────
# Tune these to adjust sensitivity without touching business logic code.

# Easy to Nil: closing arrears below this are considered quick wins
EASY_SETTLEMENT_MAX_ARREARS = 1_000          # ₹

# Insurance-only delinquency: EMI fully paid (no installment arrears) but the
# insurance/expense arrears is above this amount. ONE rule for the Pulse card,
# the Alert and the Root Cause split (utils.insurance_only_mask).
INSURANCE_EXP_ARREARS_MIN = 3_000            # ₹

# High Arrears: total arrears as a fraction of original loan amount
HIGH_ARREARS_LOAN_RATIO = 0.50               # 50 %

# Recent Advances: loans sanctioned within this window that already have delinquencies
RECENT_ADVANCES_MONTHS = 12                  # months

# Executive tables (Scorecard, NPA & SMA-2 comparison, Recovery leaderboard):
# executives with fewer accounts than this are left out, so every executive
# table lists the same people. (Overdue vs Month Demand keeps its own, stricter
# floor below.)
MIN_ACCOUNTS_EXECUTIVE = 5
SCORECARD_MIN_ACCOUNTS = MIN_ACCOUNTS_EXECUTIVE

#  Portfolio Intelligence group-size thresholds ───────────────────────────────
# A group below its threshold is excluded from that breakdown table entirely
# (too few accounts for the %/ratio to be statistically meaningful).

# Region/Branch breakdowns (NPA-SMA2 comparison, branch quadrant)
MIN_ACCOUNTS_DIMENSION_BREAKDOWN = 3

# Vehicle segment / fuel type breakdowns. _group_npa_table keeps a group when
# n >= this value. 1 means every segment shows, however small -- chosen so a
# newly-launched business line (e.g. a segment with 2 loans) is visible from
# its first month instead of hidden until it crosses a materiality floor.
# (Was 11, i.e. "strictly more than 10 accounts", before that call.)
MIN_ACCOUNTS_PRODUCT_SEGMENT = 1

# Sourcing channel and disbursement-vintage cohort breakdowns
MIN_ACCOUNTS_SOURCE_VINTAGE = 10
# The Business tab's vintage summary only names a disbursement period holding
# at least this % of the loans shown (a dozen old loans at 100% NPA is noise).
VINTAGE_HIGHLIGHT_MIN_SHARE_PCT = 2

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
# ...in one of these buckets...
REPOSSESSION_BUCKETS = ("SMA-2", "NPA")
# ...and not already seized and sold (Loan Status "S&S").
REPOSSESSION_EXCLUDE_STATUSES = ("S&S",)

# "Not paying 3M+": behind on payment and no receipt in the NOT_PAYING_RECEIPT_MONTHS
# months up to month end, worked out from Last Receipt Date at upload
# (utils.add_not_paying_flag) into NOT_PAYING_3M_FLAG_COL. Every count, list and
# table reads that column. The file's own flag (NOT_PAYING_SOURCE_FLAG_COL) is
# not trusted: on a real Sept closing it was Y for every 6+ EMI loan, including
# 1,267 that had paid since 1 Jul (business decision 2026-10-21). It is used
# only when a file has no Last Receipt Date.
NOT_PAYING_SOURCE_FLAG_COL = "No Coll 3 Months and >6 EMI"
NOT_PAYING_3M_FLAG_COL = "Not Paying 3M (receipts)"
NOT_PAYING_RECEIPT_MONTHS = 3

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

# Large customers: a customer (matched by mobile number) whose loans add up to
# at least this much SOH, across the whole upload, in ₹ crore.
LARGE_CUSTOMER_MIN_SOH_CR = 2.0

# Region spellings that mean the same region, mapped to one name at load time
# (every uploaded file and the due-date-missed list). Matching ignores case,
# spaces and punctuation, so "C.S. Nagar" and "cs nagar" both match "CS NAGAR".
# Add a line here when a new variant shows up.
REGION_NAME_ALIASES = {
    "CS NAGAR": "CHHATRAPATI SAMBHAJI NAGAR",
    "CSN": "CHHATRAPATI SAMBHAJI NAGAR",
}

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

# Branch Concern Score component weights - must sum to 1.0. Every component
# is a rate (% of the branch's accounts), so branch size doesn't bias it.
CONCERN_SCORE_WEIGHTS = {"NPA%": 0.45, "Hard Bucket%": 0.25, "Roll Fwd%": 0.2, "Not Paying 3M+%": 0.1}

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

# Roll / flow analysis (Migration tab): the early-warning steps, in the order
# a collection manager should act on them. Each is "loans in this bucket last
# month that are in a WORSE bucket now", measured by loans and by last month's
# SOH. The first two (new defaulters, 1-30 slipping further) are the cheapest
# to bring back and the ones that otherwise become NPA a few months later.
ROLL_STEPS = {
    "STD → Delinquent": "STD",
    "1-30 → SMA-1+": "1-30 DPD",
    "SMA-1 → SMA-2+": "SMA-1",
    "SMA-2 → NPA": "SMA-2",
}

# Action overview (top of the Dashboard tab): what to fix first.
ACTION_HEADLINES_MAX = 6          # most headline sentences shown
ACTION_TOP_N = 5                  # branches / executives listed under "Needs attention"
# A unit is only judged when it has at least this many loans matched in both
# months, so a 2-of-4 swing in a tiny branch never tops the list.
ACTION_MIN_ACCOUNTS = {"branch": 30, "executive": 15}
ACTION_HOTSPOT_MULTIPLE = 1.5     # a rate this many times the portfolio's is a hotspot...
ACTION_MIN_ROLLED = 3             # ...when at least this many loans made the move
ACTION_RISING_PP = 2.0            # delinquency up/down by this many points is "rising"/"improving"
ACTION_COLLECTION_GAP_PP = 5.0    # collection this many points below the portfolio is flagged
ACTION_RANGE_Z = 1.96             # forecast range: about 95% of months land inside it

# Dashboard summary: how many units each short list / league table shows.
DASHBOARD_TOP_N = 5

# The report's "New Book" section follows every loan disbursed (Ag_Date) on or
# after this date, month by month (user, 2026-10-12: "November onward advances").
NEW_BOOK_FROM = "2025-11-01"

# Deep Investigation (investigation/; design: docs/superpowers/specs/
# 2026-10-23-investigation-agent-design.md). Tuned with the evaluation set.
INVESTIGATION_MIN_SEGMENT = 200           # eligible loans for a segment to stand alone in a book-wide scan
INVESTIGATION_MIN_SEGMENT_DRILL = 50      # the same inside one segment
INVESTIGATION_NOISE_SE = 2.0              # noise band: this many standard errors of a rate change
INVESTIGATION_NO_CHANGE_PP = 0.20         # a total change under this (% points) is not material
INVESTIGATION_RECONCILE_TOL_PP = 0.01     # parts of a change must add up to within this (% points)
INVESTIGATION_DPD_DISAGREE_WARN_PCT = 2.0 # warn when the days column and the EMI buckets disagree on 30+ for more %
INVESTIGATION_POP_CHANGE_HALT_PCT = 15.0  # stop when eligible loans change by more than this % month on month
INVESTIGATION_MISSING_UNIT_HALT_PCT = 1.0 # stop when a branch missing this month held more than this % of last month
INVESTIGATION_TOP_SEGMENTS = 2            # segments cited per breakdown
# A dimension whose label changed for more than this % of the same loans
# between the two files can't be compared month on month (real Sept 2026:
# CHANNEL 31% (a file name), FUEL_TYPE 10%, Zone 9.8%; natural changes such
# as EMI date or loans per customer stay under 3%).
INVESTIGATION_LABEL_DRIFT_WARN_PCT = 5.0
# A dimension "concentrates" a material change when its concentration score
# (investigation/tools/mix.py::_concentration) reaches this; below it no
# slice explains the change and the answer is "inconclusive". Provisional:
# set from the Phase 2 cases, tuned again with the baseline.
INVESTIGATION_CONCENTRATION_MIN_PCT = 20.0
# Compare like for like (leave out branches in one file only) only while
# what is left out stays under this % of either month's eligible loans; more
# means a truncated or different extract, and the investigation stops. The
# same line as a population jump: one large branch missing (about 11%) is a
# gap, six of ten branches is another book (real Sept 2026: 1.8%).
INVESTIGATION_LIKE_FOR_LIKE_MAX_PCT = 15.0
# A driver of a material change accounts for at least this % of it: a small
# segment's real but minor move is reported in its breakdown, not named as
# a cause (Phase 2 case 12: about 6%; case 11's second cause about 32%).
INVESTIGATION_DRIVER_MIN_SHARE_PCT = 10.0

# Portfolio Review (investigation/review/; design: docs/superpowers/specs/
# 2026-10-24-portfolio-review-agent-design.md, section 5.5). Provisional,
# pending the user's confirmation (spec open question 5).
REVIEW_MIN_ACCOUNTS = 200        # eligible loans for a region to be ranked (smaller: shown, not ranked)
REVIEW_FDR = 0.10                # Benjamini-Hochberg false discovery rate, across every list of one review
# A region with more than this % of its base in strata no other region has
# (e.g. only it holds 2019-2022 agreements) has no like-for-like benchmark
# there, so it is shown but not listed (Phase 1 seasoning eval).
REVIEW_MAX_UNMATCHED_PCT = 20.0
REVIEW_MIN_RS = 10_00_000        # a shortfall or excess exposure under this (Rs) is not listed
REVIEW_MIN_PTS = 0.5             # a gap under this many % points is not listed
REVIEW_MAX_DEEP_DIVES = 3        # background deep-dives per review (the top candidate of each list)
REVIEW_MAX_JOBS = 3              # finished reviews kept in memory (one per upload / sidebar filter)
# The decision log's local file (investigation/review/decisions.py). On
# Streamlit Cloud it lasts until a restart; the panel's download keeps it.
REVIEW_DECISION_LOG = ".collectioniq/decision_log.jsonl"
