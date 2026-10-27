"""
The shape every investigation returns (the Phase 3 baseline and the Phase 4
agent alike), and the scorer that grades it against a case whose answer was
written before any run (spec section 12). Labels compare without case. A
driver citing a result that failed its own checks fails the case: a right
answer from a broken calculation is not a pass.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from investigation.evidence import EvidenceStore, ToolError


@dataclass
class Driver:
    dimension: str
    segment: str
    evidence_id: str


@dataclass
class InvestigationResult:
    # concluded | no_material_change | halted_data | inconclusive (a real change no
    # testable slice explains, e.g. its cause sits in a column the files lack)
    status: str
    drivers: list[Driver] = field(default_factory=list)
    mechanism: str | None = None             # mix | rate | flow | base
    deep: list[Driver] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)       # data_health checks not passed (info / warn / halt)
    untestable: list[str] = field(default_factory=list)  # dimensions that couldn't be tested
    excluded_units: list[str] = field(default_factory=list)
    tool_calls: int = 0
    model_calls: int = 0
    tokens: int = 0
    seconds: float = 0.0


@dataclass
class Expected:
    status: str | tuple[str, ...]            # a tuple when more than one answer is right (spec 12.2 note)
    drivers: list[tuple[str, str]] = field(default_factory=list)          # every one must be found, at any depth
    any_driver_of: list[tuple[str, str]] = field(default_factory=list)    # at least one must be found
    forbidden: list[tuple[str, str]] = field(default_factory=list)        # none may be named
    mechanism: str | None = None
    deep: list[tuple[str, str]] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)
    untestable: list[str] = field(default_factory=list)
    excluded_units: list[str] = field(default_factory=list)
    no_drivers: bool = False                 # nothing to explain: naming any driver is a made-up story


@dataclass
class Score:
    case_id: str
    checks: dict[str, bool]

    @property
    def passed(self) -> bool:
        return all(self.checks.values())


def _key(dimension: str, segment: str) -> tuple[str, str]:
    return str(dimension).strip().lower(), str(segment).strip().lower()


def _named(drivers: list[Driver]) -> set[tuple[str, str]]:
    return {_key(d.dimension, d.segment) for d in drivers}


def _lower(values) -> set[str]:
    return {str(v).strip().lower() for v in values}


def _shows(store: EvidenceStore, d: Driver) -> bool:
    """The cited result exists, passed its own checks, and is the breakdown
    by that dimension that has that segment: citing any other result for a
    driver is a guess dressed as evidence."""
    try:
        ev = store.get(d.evidence_id)
    except ToolError:
        return False
    if not ev.checks_passed or ev.tool not in ("mix_rate", "flow_bridge"):
        return False
    if str(ev.summary.get("dimension") or ev.summary.get("by") or "").lower() != d.dimension.strip().lower():
        return False
    table = ev.tables.get("segments", ev.tables.get("by"))
    return table is not None and d.segment.strip().lower() in {str(s).lower() for s in table["segment"]}


def score(case_id: str, expected: Expected, result: InvestigationResult, store: EvidenceStore) -> Score:
    """One check per field the case states; fields it leaves empty aren't judged.
    An expected driver counts when it's named at any depth (case 13: region,
    then branch, then executive, is a right answer)."""
    named, deep = _named(result.drivers), _named(result.deep)
    allowed = (expected.status,) if isinstance(expected.status, str) else tuple(expected.status)
    checks = {"status": result.status in allowed}
    if expected.drivers:
        checks["drivers"] = all(_key(*d) in named | deep for d in expected.drivers)
    if expected.no_drivers:
        checks["no_drivers"] = not result.drivers and not result.deep
    if expected.any_driver_of:
        checks["any_driver_of"] = any(_key(*d) in named for d in expected.any_driver_of)
    checks["forbidden"] = not any(_key(*d) in named | deep for d in expected.forbidden)
    if expected.mechanism:
        checks["mechanism"] = result.mechanism == expected.mechanism
    if expected.deep:
        checks["deep"] = all(_key(*d) in deep for d in expected.deep)
    if expected.flags:
        checks["flags"] = _lower(expected.flags) <= _lower(result.flags)
    if expected.untestable:
        checks["untestable"] = _lower(expected.untestable) <= _lower(result.untestable)
    if expected.excluded_units:
        checks["excluded_units"] = _lower(expected.excluded_units) <= _lower(result.excluded_units)
    checks["evidence_valid"] = all(_shows(store, d) for d in result.drivers + result.deep)
    return Score(case_id, checks)
