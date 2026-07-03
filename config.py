# Single source of truth for shared configuration values.
# Import from here rather than hardcoding strings across modules.

import os

GEMINI_MODEL = "gemini-2.5-flash-lite"

# v2 shadow mode: when enabled, each query ALSO runs the new
# LLM -> IR-1 -> compiler -> engine path and records a comparison against the
# legacy result, WITHOUT changing what the user sees. Off by default; enable with
# COLLECTIONIQ_SHADOW=1 to validate the v2 path against real traffic.
SHADOW_MODE = os.environ.get("COLLECTIONIQ_SHADOW", "").strip().lower() in ("1", "true", "yes", "on")

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

# Vehicle segment / fuel type breakdowns
MIN_ACCOUNTS_PRODUCT_SEGMENT = 5

# Sourcing channel and disbursement-vintage cohort breakdowns
MIN_ACCOUNTS_SOURCE_VINTAGE = 10

#  Portfolio Intelligence business-rule thresholds ────────────────────────────

# "Hard Bucket": accounts this many EMIs or more overdue - a narrower, more
# severe signal than NPA (NPA starts at 3 EMI overdue; Hard Bucket is deeper).
HARD_BUCKET_ARREARS_EMI_MIN = 6

# Repossession Priority List: eligible accounts must have been sanctioned
# within this many months (older loans have less collateral value left)
REPOSSESSION_WINDOW_MONTHS = 18

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
