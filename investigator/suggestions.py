"""
Deterministic "check mechanism before location" recommendations -- domain
intelligence encoded as plain Python rules, never LLM improvisation (same
"vocabulary/logic in code, LLM only for phrasing/routing" principle
CLAUDE.md documents for graph.py's IR-1/compiler split, applied here too).

The problem this solves: without it, the Investigator only ever answers
exactly what's asked. A real senior collections analyst, told "NPA% is
worse for Mahad branch," doesn't jump straight to "which executive is
worst" -- they first ask WHY it's worse, mechanically: is it roll-forward
(existing accounts sliding backward -- a collections-execution problem) or
vintage-driven (recent originations defaulting early -- an underwriting-
quality problem)? Those imply different fixes. This module encodes that
judgment as a lookup table from "which metric moved worst" to "which
mechanism step(s) to suggest checking first," rather than asking the
narration LLM to freely improvise a suggestion -- an LLM could suggest a
step that doesn't exist, misjudge when a rule applies, or phrase the same
situation two different ways on two different runs. A lookup table can't.

Suggesting is NOT autonomy: every suggestion here is rendered as a
one-click button the user still has to press (see
ui/tabs/investigator.py's use of suggest_mechanism_steps) -- nothing in
this module executes anything on its own, and every suggested step is
still a discrete, downloadable, human-approved table like everything else
in this app.
"""
from __future__ import annotations

# Metric -> which mechanism step(s) to suggest checking BEFORE drilling
# into executives, and why. Only the no-space spelling is registered --
# this is only ever consulted against entity_summary's branch/region/zone/
# bu output (_BRANCH_METRICS / _GENERIC_DIMENSION_METRICS in
# investigator/steps.py, both no-space), never executive entity_summary
# (_EXEC_METRICS, WITH a space) -- an executive is already "location," so
# "check mechanism before location" doesn't apply once you're already at
# executive grain.
MECHANISM_SUGGESTIONS: dict[str, list[dict]] = {
    "NPA%": [
        {
            "step_type": "roll_rate_summary",
            "label": "Check roll-rate",
            "reason": (
                "NPA% moved for the worse. Before drilling into executives, "
                "check whether this is roll-forward driven (existing accounts "
                "sliding into worse buckets) or roll-backward driven (fewer "
                "accounts recovering) -- these need different fixes."
            ),
        },
        {
            "step_type": "vintage_summary",
            "label": "Check vintage",
            "reason": (
                "Also worth checking: is this concentrated in RECENT "
                "originations (an underwriting-quality signal) or spread "
                "across the seasoned book (an execution signal)?"
            ),
        },
    ],
    "Hard Bucket%": [
        {
            "step_type": "roll_rate_summary",
            "label": "Check roll-rate",
            "reason": (
                "Hard Bucket% moved for the worse. Before drilling into "
                "executives, check whether accounts are sliding into deeper "
                "arrears (roll-forward) or recovering (roll-backward) -- "
                "these need different fixes."
            ),
        },
        {
            "step_type": "vintage_summary",
            "label": "Check vintage",
            "reason": (
                "Also worth checking: is this concentrated in RECENT "
                "originations or spread across the seasoned book?"
            ),
        },
    ],
    "Collection%": [
        {
            "step_type": "demand_summary",
            "label": "Break down by overdue vs demand",
            "reason": (
                "Collection% moved for the worse. Before drilling into "
                "executives, check whether it's driven by uncollected OLD "
                "arrears or by this month's own EMI demand not being "
                "collected -- these need different fixes."
            ),
        },
    ],
}


def suggest_mechanism_steps(worst_metric: str | None) -> list[dict]:
    """Which mechanism step(s) to suggest for the given worst-moved metric,
    each as {"step_type", "label", "reason"} -- empty list when the metric
    has no registered mechanism angle (e.g. Strike%, SOH, Accounts: there's
    no roll-rate/vintage/demand angle on those)."""
    if not worst_metric:
        return []
    return MECHANISM_SUGGESTIONS.get(worst_metric, [])
