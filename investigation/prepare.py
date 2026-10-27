"""
Get two months ready to compare (user decision 2026-10-24): run data_health,
and when a branch is in one month's file only (missing this month, or
missing from last month's file although its loans already existed), compare
like for like by leaving it out of BOTH months instead of stopping. Say what
was left out, with loan counts, and the change with and without it.

Like for like is only for a gap: when what would be left out passes
INVESTIGATION_LIKE_FOR_LIKE_MAX_PCT of either month's eligible loans the
files aren't the same book (a truncated or different extract) and the
investigation stops. Any other halt (a population jump, for example) stops
it too. A branch renamed or merged into another is not a gap.
"""
from __future__ import annotations

from dataclasses import dataclass

from config import INVESTIGATION_LIKE_FOR_LIKE_MAX_PCT
from investigation.evidence import Evidence, EvidenceStore, ToolError
from investigation.metrics import get_metric
from investigation.panel import PeriodPair
from investigation.tools.change import side_counts
from investigation.tools.health import data_health, unit_of

_COVERAGE_CHECKS = ("units_missing", "units_new")


@dataclass
class Prepared:
    pair: PeriodPair
    status: str                      # "ok" | "halted_data"
    files: Evidence                  # data_health of the two whole files (coverage gaps are found here)
    health: Evidence                 # data_health of the pair compared, in the scope asked about
    adjustment: Evidence | None      # the like-for-like record, when branches were left out

    @property
    def flags(self) -> list[str]:
        """Every health check not passed, in either record."""
        seen = []
        for ev in (self.files, self.health):
            for f in ev.summary["findings"]:
                if f["status"] != "pass" and f["check"] not in seen:
                    seen.append(f["check"])
        return seen


def _change(pair: PeriodPair, scope: dict | None) -> float:
    metric = get_metric("thirty_plus_rate")
    p = pair.scoped(scope)
    prev, curr = side_counts(p.prev, metric), side_counts(p.curr, metric)
    if prev["rate"] is None or curr["rate"] is None:
        raise ToolError("no eligible loans in one of the months for this scope")
    return curr["rate"] - prev["rate"]


def prepare(store: EvidenceStore, pair: PeriodPair, scope: dict | None = None) -> Prepared:
    """Which branches are in one file only is a property of the two files, so
    it's decided on the whole files; the health check of the scope asked
    about then runs on the like-for-like pair and decides the status."""
    files = data_health(store, pair)
    findings = files.summary["findings"]
    units = sorted({u for f in findings if f["check"] in _COVERAGE_CHECKS and f["status"] in ("warn", "halt")
                    for u in f.get("units", [])})
    if not units:
        health = data_health(store, pair, scope) if scope else files
        return Prepared(pair, "halted_data" if health.summary["status"] == "halt" else "ok", files, health, None)

    metric = get_metric("thirty_plus_rate")
    p_unit, c_unit = unit_of(pair.prev), unit_of(pair.curr)
    p_elig, c_elig = metric.eligible(pair.prev), metric.eligible(pair.curr)
    left_p = 100 * int((p_elig & p_unit.isin(units)).sum()) / max(int(p_elig.sum()), 1)
    left_c = 100 * int((c_elig & c_unit.isin(units)).sum()) / max(int(c_elig.sum()), 1)
    if max(left_p, left_c) > INVESTIGATION_LIKE_FOR_LIKE_MAX_PCT:
        return Prepared(pair, "halted_data", files, files, None)      # not a gap: a different or truncated book

    for dim_id, value in (scope or {}).items():
        if dim_id == "branch" and str(value).strip().upper() in units:
            raise ToolError(f"{value} is in one month's file only, so it can't be compared month on month")
    adjusted = pair.without_units(units)
    reasons = {u: f["check"] for f in findings if f["check"] in _COVERAGE_CHECKS for u in f.get("units", [])}
    excluded = [{
        "unit": u,
        "reason": "missing this month" if reasons[u] == "units_missing" else "missing from last month's file",
        "loans_prev": int((p_unit == u).sum()), "loans_curr": int((c_unit == u).sum()),
    } for u in units]
    adjustment = store.add("like_for_like", {"scope": dict(scope or {})}, {
        "excluded": excluded, "left_out_pct": {"prev": left_p, "curr": left_c},
        "full_change_pp": _change(pair, scope), "like_for_like_change_pp": _change(adjusted, scope),
    })
    recheck = data_health(store, adjusted, scope)
    status = "halted_data" if recheck.summary["status"] == "halt" else "ok"
    return Prepared(adjusted, status, files, recheck, adjustment)
