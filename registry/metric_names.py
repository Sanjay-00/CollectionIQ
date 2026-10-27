"""Metric names: each metric's one name, its other spellings, and which way is
good, written down once (code review item 5, 2026-10-27).

Branch, region and view tables spell rates without a space ("NPA%"); the
executive scorecard (analysis/executive_scorecard.py) with one ("NPA %",
"Strike Rate %"); the AI may write either. These facts were copied into 9
tables across 7 files, and a spelling missing from one copy gave empty
tables, wrong sorts and refused questions. Every table callers use is now
built from METRICS below, and tests/test_metric_names.py checks the rest.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Metric:
    name: str                       # as branch, region and view tables spell it
    higher_is_better: bool
    aliases: tuple[str, ...] = ()
    executive: str | None = None    # the executive scorecard's column, when it has one
    rollup: str | None = None       # portfolio row of a view: "mean" (a rate) or "sum" (a count)


_ALL = [
    Metric("Collection%",   True,  ("Collection %",), "Collection %", "mean"),
    Metric("NPA%",          False, ("NPA %",), "NPA %", "mean"),
    Metric("NPA% (SOH)",    False, ("NPA % (SOH)",)),
    Metric("Delinquency%",  False, ("Delinquency %",), "Delinquency %"),
    Metric("Strike%",       True,  ("Strike Rate %", "Strike Rate%", "Strike %"), "Strike Rate %", "mean"),
    Metric("Hard Bucket%",  False, ("Hard Bucket %",), "Hard Bucket %", "mean"),
    Metric("SMA-2%",        False, ("SMA-2 %",), "SMA-2 %", "mean"),
    Metric("Roll Fwd%",     False, ("Roll Fwd %",), rollup="mean"),
    Metric("Roll Bwd%",     True,  ("Roll Bwd %",), rollup="mean"),
    Metric("Δ NPA%",        False, rollup="mean"),
    Metric("Δ SMA-2%",      False, rollup="mean"),
    Metric("Concern Score", False, rollup="mean"),
    Metric("Slipped",       False, rollup="sum"),
    Metric("Rescued",       True,  rollup="sum"),
    Metric("Net Recovery",  True,  rollup="sum"),
    Metric("STD → Delinquent | %", False),          # new defaulters (roll analysis)
    # New business: more is always the good direction here.
    Metric("Accounts This Month", True, rollup="sum"),
    Metric("Funded (Cr)",         True, rollup="sum"),
    Metric("Accounts MoM %",      True, rollup="mean"),
    Metric("Funded MoM %",        True, rollup="mean"),
]

METRICS: dict[str, Metric] = {m.name: m for m in _ALL}
_BY_SPELLING: dict[str, Metric] = {s: m for m in _ALL for s in (m.name, *m.aliases)}


def canonical(spelling) -> str | None:
    """The metric's one name for any spelling; None when it isn't a known metric."""
    m = _BY_SPELLING.get(str(spelling).strip()) if spelling is not None else None
    return m.name if m else None


def higher_is_better(spelling) -> bool | None:
    m = _BY_SPELLING.get(str(spelling).strip()) if spelling is not None else None
    return m.higher_is_better if m else None


def executive_column(spelling: str) -> str:
    """The executive scorecard's column for a metric; unchanged when it has none."""
    m = _BY_SPELLING.get(str(spelling).strip())
    return m.executive if m and m.executive else str(spelling).strip()


def spellings(spelling: str) -> list[str]:
    """The name first, then every other spelling."""
    m = _BY_SPELLING[str(spelling).strip()]
    return [m.name, *m.aliases]
