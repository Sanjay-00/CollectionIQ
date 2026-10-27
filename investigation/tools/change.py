"""
How much the metric moved, and the loan flows behind the move (spec 7.2, 7.3).
"""
from __future__ import annotations

import pandas as pd

from config import INVESTIGATION_NO_CHANGE_PP, INVESTIGATION_RECONCILE_TOL_PP
from investigation.evidence import Check, Evidence, EvidenceStore, ToolError
from investigation.metrics import ELIGIBLE_BUCKETS, THIRTY_PLUS_BUCKETS, MetricDef, get_metric
from investigation.panel import PeriodPair
from investigation.tools.noise import rate_noise_pp

_IN_LINES = ("in_from_std", "in_from_1_30", "in_from_no_emi", "new_to_book")
_OUT_LINES = ("cured_to_std", "cured_to_1_30", "became_no_emi", "left_book")
LINES = ("opening", *_IN_LINES, *_OUT_LINES, "closing")
_BASE_IN, _BASE_OUT = ("new_to_book", "in_from_no_emi"), ("became_no_emi", "left_book")
BASE_LINES = ("opening", *_BASE_IN, *_BASE_OUT, "closing")


def month_word(month: str) -> str:
    """'2026-08' -> 'Aug 2026'."""
    return pd.Period(month, "M").strftime("%b %Y")


def _counts(running, eligible, numerator) -> dict:
    running, eligible, numerator = int(running.sum()), int(eligible.sum()), int(numerator.sum())
    return {"running": running, "eligible": eligible, "numerator": numerator,
            "rate": 100 * numerator / eligible if eligible else None,
            "rate_all_running": 100 * numerator / running if running else None}


def side_counts(df: pd.DataFrame, metric: MetricDef) -> dict:
    """Running, eligible and numerator counts and both rates (None when the
    base is empty: an undefined rate is never 0)."""
    return _counts(metric.running(df), metric.eligible(df), metric.numerator(df))


def pair_counts(p: PeriodPair, metric: MetricDef, month: str) -> dict:
    """side_counts for one month of a pair, from the pair's cached masks."""
    return _counts(*p.masks(metric)[month])


def material_change(p: PeriodPair, metric: MetricDef) -> tuple[float, float | None, bool]:
    """(change in % points, its paired noise band, material) for a scoped pair.
    Material (spec 9.3): at least INVESTIGATION_NO_CHANGE_PP AND outside the
    noise band, so a small scope's wobble is never read as a change."""
    prev, curr = pair_counts(p, metric, "prev"), pair_counts(p, metric, "curr")
    if prev["rate"] is None or curr["rate"] is None:
        raise ToolError("no eligible loans in one of the months for this scope")
    (_, e0, y0), (_, e1, y1) = p.masks(metric)["prev"], p.masks(metric)["curr"]
    i0, i1 = p._both_months()
    both = e0[i0] & e1[i1]                     # eligible in both months
    a, z = y0[i0][both], y1[i1][both]
    noise = rate_noise_pp(prev["eligible"], prev["numerator"], curr["eligible"], curr["numerator"],
                          int(both.sum()), int((a & ~z).sum()), int((~a & z).sum()))
    change = curr["rate"] - prev["rate"]
    return change, noise, abs(change) >= INVESTIGATION_NO_CHANGE_PP and abs(change) > (noise or 0.0)


def metric_change(store: EvidenceStore, pair: PeriodPair, metric_id: str = "thirty_plus_rate",
                  scope: dict | None = None) -> Evidence:
    """The rate both months, its change in % points and relative %, and the
    change split into a numerator effect (more or fewer 30+ loans) and a base
    effect (more or fewer eligible loans); the two add up to the change."""
    metric = get_metric(metric_id)
    p = pair.scoped(scope)
    prev, curr = pair_counts(p, metric, "prev"), pair_counts(p, metric, "curr")
    for counts, month in ((prev, pair.prev_month), (curr, pair.curr_month)):
        if not counts["eligible"]:
            raise ToolError(f"no eligible loans in {month_word(month)} for this scope")
    n0, d0, n1, d1 = prev["numerator"], prev["eligible"], curr["numerator"], curr["eligible"]
    change = curr["rate"] - prev["rate"]
    numerator_effect = 100 * (n1 - n0) / d1
    base_effect = 100 * n0 * (1 / d1 - 1 / d0)
    summary = {
        "metric": metric.id, "label": metric.label, "version": metric.version,
        "prev_month": pair.prev_month, "curr_month": pair.curr_month, "scope": dict(scope or {}),
        "prev": prev, "curr": curr,
        "change_pp": change,
        "change_rel_pct": 100 * change / prev["rate"] if prev["rate"] else None,
        "all_running_change_pp": curr["rate_all_running"] - prev["rate_all_running"],
        "numerator_effect_pp": numerator_effect, "base_effect_pp": base_effect,
    }
    _, summary["noise_pp"], summary["material"] = material_change(p, metric)
    checks = [
        Check("effects_add_up", abs(numerator_effect + base_effect - change) <= INVESTIGATION_RECONCILE_TOL_PP,
              f"{numerator_effect:+.4f} + {base_effect:+.4f} vs {change:+.4f}"),
        Check("numerator_within_eligible", all(s["numerator"] <= s["eligible"] <= s["running"] for s in (prev, curr))),
    ]
    return store.add("metric_change", {"metric_id": metric_id, "scope": dict(scope or {})}, summary, checks=checks)


def _flow_flags(j: pd.DataFrame) -> pd.DataFrame:
    """Per loan, one boolean column per bridge line (30+ and eligible base).
    A loan not running in a month counts as absent that month."""
    here_p, here_c = j["prev_running"], j["curr_running"]
    both = here_p & here_c
    p30 = here_p & j["prev_bucket"].isin(THIRTY_PLUS_BUCKETS)
    c30 = here_c & j["curr_bucket"].isin(THIRTY_PLUS_BUCKETS)
    pe = here_p & j["prev_bucket"].isin(ELIGIBLE_BUCKETS)
    ce = here_c & j["curr_bucket"].isin(ELIGIBLE_BUCKETS)
    f = {
        "opening": p30, "closing": c30,
        "in_from_std": both & (j["prev_bucket"] == "STD") & c30,
        "in_from_1_30": both & (j["prev_bucket"] == "1-30 DPD") & c30,
        "in_from_no_emi": both & ~j["prev_bucket"].isin(ELIGIBLE_BUCKETS) & c30,
        "new_to_book": ~here_p & c30,
        "cured_to_std": both & p30 & (j["curr_bucket"] == "STD"),
        "cured_to_1_30": both & p30 & (j["curr_bucket"] == "1-30 DPD"),
        "became_no_emi": both & p30 & ~j["curr_bucket"].isin(ELIGIBLE_BUCKETS),
        "left_book": p30 & ~here_c,
        "base:opening": pe, "base:closing": ce,
        "base:new_to_book": ~here_p & ce,
        "base:in_from_no_emi": both & ~j["prev_bucket"].isin(ELIGIBLE_BUCKETS) & ce,
        "base:became_no_emi": both & pe & ~j["curr_bucket"].isin(ELIGIBLE_BUCKETS),
        "base:left_book": pe & ~here_c,
    }
    return pd.DataFrame(f).astype(int)


def _closes(row, prefix: str = "") -> bool:
    ins = _BASE_IN if prefix else _IN_LINES
    outs = _BASE_OUT if prefix else _OUT_LINES
    return row[f"{prefix}opening"] + sum(row[f"{prefix}{k}"] for k in ins) \
        - sum(row[f"{prefix}{k}"] for k in outs) == row[f"{prefix}closing"]


def flow_bridge(store: EvidenceStore, pair: PeriodPair, metric_id: str = "thirty_plus_rate",
                scope: dict | None = None, by: str | None = None) -> Evidence:
    """Last month's 30+ loans, plus loans that rolled in (from STD, 1-30 DPD,
    no EMI yet) and loans new to the book, minus cures, loans that went back
    to no EMI and loans that left the running book, equals this month's
    (checked exactly); the same for the eligible base. by: one row per
    segment, each loan under this month's label (last month's when it left
    the book)."""
    get_metric(metric_id)                   # v1 bridges the 30+ buckets; refuses an unknown metric
    p = pair.scoped(scope)
    j = p.joined()
    flags = _flow_flags(j)
    total = flags.sum()
    lines = {k: int(total[k]) for k in LINES}
    base_lines = {k: int(total[f"base:{k}"]) for k in BASE_LINES}
    left = j.loc[flags["left_book"].astype(bool), "prev_bucket"].value_counts()
    summary = {"metric": metric_id, "scope": dict(scope or {}), "lines": lines, "base_lines": base_lines,
               "left_book_last_bucket": {str(b): int(n) for b, n in left.items()}}
    checks = [Check("bridge_closes", _closes(total)), Check("base_bridge_closes", _closes(total, "base:"))]
    tables = {}
    if by:
        p.require(by)
        segment = j["Loan No"].map(p.loan_label(by))
        per = flags[list(LINES)].groupby(segment.to_numpy()).sum()
        per = per[(per != 0).any(axis=1)]
        tables["by"] = per.rename_axis("segment").reset_index()
        checks.append(Check("each_segment_closes", bool(per.apply(_closes, axis=1).all())))
        summary["by"] = by
    return store.add("flow_bridge", {"metric_id": metric_id, "scope": dict(scope or {}), "by": by},
                     summary, tables, checks)
