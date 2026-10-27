"""
data_health (spec 7.1): check the two files before explaining any change.
A halt stops the investigation and the data problem becomes the answer.
Statuses: pass, info (worth knowing), warn (can distort the answer), halt.
"""
from __future__ import annotations

import pandas as pd

from config import (
    INVESTIGATION_DPD_DISAGREE_WARN_PCT, INVESTIGATION_LABEL_DRIFT_WARN_PCT, INVESTIGATION_MISSING_UNIT_HALT_PCT,
    INVESTIGATION_POP_CHANGE_HALT_PCT,
)
from investigation.dimensions import DIMENSIONS
from investigation.evidence import Evidence, EvidenceStore
from investigation.metrics import get_metric
from investigation.panel import PeriodPair
from investigation.tools.change import month_word
from utils import count_agreed_after_month

_RANK = {"pass": 0, "info": 0, "warn": 1, "halt": 2}
_DAYS = "DelinquencyDays"


def _f(check: str, status: str, detail: str, units: list[str] | None = None) -> dict:
    out = {"check": check, "status": status, "detail": detail}
    if units is not None:
        out["units"] = units           # branches in one file only: prepare() compares like for like without them
    return out


def _names(items, limit: int = 8) -> str:
    items = sorted(map(str, items))
    return ", ".join(items[:limit]) + (f" and {len(items) - limit} more" if len(items) > limit else "")


def _months(pair: PeriodPair) -> dict:
    gap = (pd.Period(pair.curr_month, "M") - pd.Period(pair.prev_month, "M")).n
    if gap == 1:
        return _f("months_in_order", "pass", f"{month_word(pair.prev_month)} then {month_word(pair.curr_month)}")
    return _f("months_in_order", "warn", f"{month_word(pair.prev_month)} and {month_word(pair.curr_month)} are "
                                         f"{gap} months apart, not consecutive closings")


def _headers(pair: PeriodPair) -> list[dict]:
    renamed = {}
    for df in (pair.prev, pair.curr):
        renamed.update({k: v for k, v in (df.attrs.get("header_renames") or {}).items() if k.lower() != v.lower()})
    out = [_f("header_renames", "warn" if renamed else "pass",
              ("headers read under another name: " + _names(f"'{k}' as {v}" for k, v in renamed.items()))
              if renamed else "no renamed headers")]
    only = (set(pair.prev.columns) ^ set(pair.curr.columns)) - {"prev_bucket"}
    out.append(_f("columns_one_month_only", "warn" if only else "pass",
                  f"in one month's file only: {_names(only)}" if only else "same columns both months"))
    dups = {month_word(m): int(df.attrs.get("dropped_duplicate_loans") or 0)
            for df, m in ((pair.prev, pair.prev_month), (pair.curr, pair.curr_month))}
    dup_text = ", ".join(f"{n:,} in {m}" for m, n in dups.items() if n)
    out.append(_f("duplicates_dropped", "warn" if dup_text else "pass",
                  f"duplicate loan rows dropped at upload: {dup_text}" if dup_text else "no duplicate loans"))
    return out


def _units_new(pair: PeriodPair, new: set[str], curr_eligible: int) -> dict:
    """A branch only in this month's file is either new, or was missing from
    last month's file: its loans were agreed by last month's close but aren't
    in last month's file (real Sept 2026: HINGL, WASHM, 1,657 loans back to
    2022, 1.8% of the book; like for like the change was +0.71, not +0.78 %
    points, and every split by branch was distorted by them). Missing loans
    over INVESTIGATION_MISSING_UNIT_HALT_PCT of this month's eligible loans halt."""
    if not new:
        return _f("units_new", "pass", "no new branches", [])
    rows = pair.curr[unit_of(pair.curr).isin(new)]
    not_last_month = ~rows["Loan No"].isin(set(pair.prev["Loan No"]))
    agreed = pd.to_datetime(rows["Ag_Date"], errors="coerce") if "Ag_Date" in rows.columns \
        else pd.Series(pd.NaT, index=rows.index)
    old = not_last_month & (agreed <= pd.Period(pair.prev_month, "M").end_time)
    row_unit = unit_of(rows)
    by_unit = old.groupby(row_unit).sum()
    missing = {u: int(n) for u, n in by_unit.items() if n >= 0.5 * int((row_unit == u).sum())}
    if not missing:
        return _f("units_new", "info", f"branches new this month: {_names(new)}", [])
    share = 100 * sum(missing.values()) / max(curr_eligible, 1)
    status = "halt" if share > INVESTIGATION_MISSING_UNIT_HALT_PCT else "warn"
    detail = (f"missing from last month's file: {', '.join(f'{u} ({n:,} loans agreed by then)' for u, n in sorted(missing.items()))}, "
              f"{share:.1f}% of this month's loans. Compare like for like by leaving them out, or add them to last month's file")
    new_only = new - set(missing)
    if new_only:
        detail += f". New this month: {_names(new_only)}"
    return _f("units_new", status, detail, sorted(missing))


def unit_of(df: pd.DataFrame) -> pd.Series:
    """Each loan's branch as its month's file has it, trimmed and upper-cased:
    the one rule for "which branch is in which file"."""
    return df["Unit"].astype(str).str.strip().str.upper()


def _units(pair: PeriodPair, prev_eligible: pd.Series, curr_eligible: int) -> list[dict]:
    if "Unit" not in pair.prev.columns or "Unit" not in pair.curr.columns:
        return [_f("units_missing", "info", "not checked: no Unit column")]
    p_unit, c_unit = unit_of(pair.prev), unit_of(pair.curr)
    before, now = set(p_unit), set(c_unit)
    out = [_units_new(pair, now - before, curr_eligible)]
    gone = before - now
    if not gone:
        return out + [_f("units_missing", "pass", "every branch from last month is here", [])]
    # A branch whose loans are still in the book under another branch was
    # renamed or merged, not lost (review: it used to be "left out, 0 loans").
    curr_unit = dict(zip(pair.curr["Loan No"], c_unit))
    moved, missing = {}, []
    for u in sorted(gone):
        now_under = pair.prev.loc[p_unit == u, "Loan No"].map(curr_unit).dropna()
        if len(now_under) >= 0.5 * int((p_unit == u).sum()):
            moved[u] = now_under.value_counts().index[0]
        else:
            missing.append(u)
    moved_text = ", ".join(f"{u}'s loans are now under {v}" for u, v in moved.items())
    if not missing:
        return out + [_f("units_missing", "info", f"branch renamed or merged: {moved_text}", [])]
    share = pair.prev.loc[prev_eligible].groupby(p_unit[prev_eligible]).size() / max(int(prev_eligible.sum()), 1) * 100
    big = [u for u in missing if share.get(u, 0) > INVESTIGATION_MISSING_UNIT_HALT_PCT]
    detail = "missing this month: " + ", ".join(f"{u} ({share.get(u, 0):.1f}% of last month's loans)" for u in missing)
    if moved:
        detail += f". Renamed or merged: {moved_text}"
    return out + [_f("units_missing", "halt" if big else "warn", detail, missing)]


def _population(pair: PeriodPair, n0: int, n1: int) -> dict:
    pct = 100 * (n1 - n0) / n0 if n0 else float("inf")
    status = "halt" if abs(pct) > INVESTIGATION_POP_CHANGE_HALT_PCT else "pass"
    return _f("population_change", status, f"eligible loans {n0:,} -> {n1:,} ({pct:+.1f}%)")


def _days(pair: PeriodPair, metric) -> dict:
    if _DAYS not in pair.prev.columns or _DAYS not in pair.curr.columns:
        return _f("days_column_agreement", "info", f"not checked: no {_DAYS} column")
    worst, parts = 0.0, []
    for df, m in ((pair.prev, pair.prev_month), (pair.curr, pair.curr_month)):
        elig = metric.eligible(df)
        days = pd.to_numeric(df[_DAYS], errors="coerce")
        known = elig & days.notna()
        off = int(((days > 30) != metric.numerator(df))[known].sum())
        pct = 100 * off / max(int(elig.sum()), 1)
        worst = max(worst, pct)
        parts.append(f"{month_word(m)} {off:,} loans ({pct:.2f}%)")
    status = "warn" if worst > INVESTIGATION_DPD_DISAGREE_WARN_PCT else "pass"
    return _f("days_column_agreement", status, "days column and EMI buckets disagree on 30+: " + "; ".join(parts))


def _executives(pair: PeriodPair) -> dict:
    if "MNT NAME" not in pair.prev.columns or "MNT NAME" not in pair.curr.columns:
        return _f("executives_changed", "info", "not checked: no MNT NAME column")
    m = pair.prev[["Loan No", "MNT NAME"]].merge(pair.curr[["Loan No", "MNT NAME"]], on="Loan No")
    changed = int((m["MNT NAME_x"].astype(str).str.strip().str.upper()
                   != m["MNT NAME_y"].astype(str).str.strip().str.upper()).sum())
    return _f("executives_changed", "info" if changed else "pass",
              f"{changed:,} loans changed executive; they count under this month's executive" if changed
              else "no loans changed executive")


def drifted_dimensions(pair: PeriodPair) -> dict[str, float]:
    """{dimension id: % of the same loans relabelled} above the warning line.
    Executive is left out: a reassigned loan follows its new owner by design."""
    out = {}
    for dim in DIMENSIONS.values():
        if dim.kind == "executive" or pair.unavailable(dim.id):
            continue
        pct = pair.label_drift_pct(dim.id)
        if pct is not None and pct > INVESTIGATION_LABEL_DRIFT_WARN_PCT:
            out[dim.id] = pct
    return out


def _labels_changed(pair: PeriodPair) -> dict:
    drift = drifted_dimensions(pair)
    if not drift:
        return _f("labels_changed", "pass", "dimensions keep their labels for the same loans")
    shown = ", ".join(f"{DIMENSIONS[d].label} {pct:.1f}%" for d, pct in sorted(drift.items(), key=lambda x: -x[1]))
    return _f("labels_changed", "warn", f"labels changed for the same loans between the files: {shown}. "
                                        "These can't be compared month on month and are left out of the scan")


def _agreed_after(pair: PeriodPair) -> dict:
    counts = {month_word(m): count_agreed_after_month(df, m) for df, m in
              ((pair.prev, pair.prev_month), (pair.curr, pair.curr_month))}
    text = ", ".join(f"{n:,} in {m}" for m, n in counts.items() if n)
    return _f("agreed_after_month", "warn" if text else "pass",
              f"agreement date after the month (likely day/month swaps): {text}" if text else "no future agreement dates")


def data_health(store: EvidenceStore, pair: PeriodPair, scope: dict | None = None) -> Evidence:
    """Findings on the two files; summary status is the worst of them."""
    metric = get_metric("thirty_plus_rate")
    p = pair.scoped(scope)
    prev_eligible, curr_eligible = metric.eligible(p.prev), int(metric.eligible(p.curr).sum())
    findings = [
        _months(p), *_headers(p), *_units(p, prev_eligible, curr_eligible),
        _population(p, int(prev_eligible.sum()), curr_eligible),
        _days(p, metric), _labels_changed(p), _executives(p), _agreed_after(p),
    ]
    worst = max(findings, key=lambda f: _RANK[f["status"]])["status"]
    status = {"info": "pass"}.get(worst, worst)
    return store.add("data_health", {"scope": dict(scope or {})}, {"status": status, "findings": findings})
