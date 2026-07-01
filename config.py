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
