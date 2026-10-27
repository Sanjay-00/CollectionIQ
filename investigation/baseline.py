"""
The fixed-script investigation (spec 8): the tools in a set order, no model
call. It is a complete answer on its own and the benchmark the Phase 4 agent
must beat. Rules follow spec 12.2's notes: a driver is a segment beyond its
fair share of the change and noise, pushing the same way as the total, and
accounting for at least INVESTIGATION_DRIVER_MIN_SHARE_PCT of it.
"""
from __future__ import annotations

import time
from typing import Callable

import pandas as pd

from config import INVESTIGATION_CONCENTRATION_MIN_PCT, INVESTIGATION_DRIVER_MIN_SHARE_PCT, INVESTIGATION_NO_CHANGE_PP
from investigation.evaluation import Driver, InvestigationResult
from investigation.evidence import EvidenceStore, ToolError
from investigation.panel import PeriodPair, make_pair
from investigation.prepare import prepare
from investigation.tools import controlled_decompose, drill, flow_bridge, metric_change, scan_dimensions

# Geography, coarse to fine. Two geographies are nested, not rivals: a branch
# effect always "persists" inside its own region (review: a vacuous rival test).
GEO = ("zone", "region", "branch", "executive")


def are_rivals(a: str, b: str) -> bool:
    return not (a in GEO and b in GEO)


def choose_lead(lead: dict, rival: dict, persists: Callable[[dict], bool | None]) -> dict:
    """persists(x): does x's top segment stand out inside the other's groups?
    The lead stays unless it vanishes inside the rival and the rival holds up
    inside the lead (True / False / None = untestable)."""
    if persists(lead) is False and persists(rival) is True:
        return rival
    return lead


def mechanism_of(t: dict) -> str:
    """mix when the share terms (mix, new, gone) outweigh the rate term: a
    segment new this month has no rate term at all."""
    share = t["mix_pp"] + t.get("new_pp", 0.0) + t.get("gone_pp", 0.0)
    return "mix" if abs(share) > abs(t["rate_pp"]) else "rate"


def deep_allowed(inner_total: float, total: float, first: dict) -> bool:
    """A deep driver only explains a driver whose own loans moved the way the
    whole change did (review: a mix driver whose rate fell gave an executive
    "inside it" pulling against the rise)."""
    return inner_total * total > 0 and first["rate_pp"] * total > 0


def _qualifying(entry: dict, total: float, material: bool) -> list[dict]:
    """A ranked dimension's segments that may be named as drivers."""
    out = []
    for t in entry["top"]:
        if not t["beyond_share"]:
            continue
        if material and 100 * t["contribution_pp"] / total < INVESTIGATION_DRIVER_MIN_SHARE_PCT:
            continue
        if not material and abs(t["contribution_pp"]) < INVESTIGATION_NO_CHANGE_PP:
            continue
        out.append(t)
    return out


def _candidates(ranked: list[dict], total: float) -> list[dict]:
    return [r for r in ranked if (r.get("concentration_pct") or 0.0) >= INVESTIGATION_CONCENTRATION_MIN_PCT
            and _qualifying(r, total, True)]


def _inside(pair: PeriodPair, fine: str, f_seg: str, coarse: str, c_seg: str) -> bool:
    """Do most of the fine segment's loans sit inside the coarse segment?"""
    f = pair.loan_label(fine).astype(str).str.upper() == f_seg.upper()
    c = pair.loan_label(coarse).astype(str).str.upper() == c_seg.upper()
    return bool(f.sum()) and (f & c).sum() >= 0.8 * f.sum()


def _finest_geography(pair: PeriodPair, lead: dict, candidates: list[dict], total: float) -> dict:
    """When a finer geography's top segment sits inside the lead's and holds
    most of its change, the finer one leads (region R1 vs branch BR3 inside it)."""
    if lead["dimension"] not in GEO:
        return lead
    lead_top = _qualifying(lead, total, True)[0]
    for cand in candidates:
        if cand["dimension"] in GEO and GEO.index(cand["dimension"]) > GEO.index(lead["dimension"]):
            top = _qualifying(cand, total, True)[0]
            if top["contribution_pp"] >= 0.5 * lead_top["contribution_pp"] and \
                    _inside(pair, cand["dimension"], top["segment"], lead["dimension"], lead_top["segment"]):
                return _finest_geography(pair, cand, candidates, total)
    return lead


def _persists_fn(store: EvidenceStore, pair: PeriodPair, total: float, scope: dict | None, other: dict,
                 lead: dict) -> Callable[[dict], bool | None]:
    def persists(who: dict) -> bool | None:
        control = other if who is lead else lead
        segment = _qualifying(who, total, True)[0]["segment"]
        try:
            return controlled_decompose(store, pair, who["dimension"], segment, control["dimension"],
                                        scope=scope).summary["persists"]
        except ToolError as e:                         # recorded and reported, never silent
            store.add("rival_untestable", {"dimension": who["dimension"], "segment": segment,
                                           "control": control["dimension"], "scope": dict(scope or {})},
                      {"reason": str(e)})
            return None
    return persists


def run_baseline(curr: pd.DataFrame, prev: pd.DataFrame, curr_month: str, prev_month: str,
                 scope: dict | None = None, pair: PeriodPair | None = None) -> tuple[InvestigationResult, EvidenceStore]:
    """pair: these two months already paired (make_pair), to share its work
    across several investigations of the same upload (the review runner)."""
    started = time.time()
    store = EvidenceStore()
    prep = prepare(store, pair if pair is not None else make_pair(curr, prev, curr_month, prev_month), scope)
    excluded = [x["unit"] for x in prep.adjustment.summary["excluded"]] if prep.adjustment else []
    result = InvestigationResult(status="halted_data", flags=prep.flags, excluded_units=excluded)
    if prep.status != "ok":
        return _done(result, store, started)

    pair = prep.pair
    change = metric_change(store, pair, scope=scope).summary
    bridge = flow_bridge(store, pair, scope=scope).summary["lines"]
    scan = scan_dimensions(store, pair, scope=scope).summary
    result.untestable = [u["dimension"] for u in scan["unavailable"]]
    total, ranked = change["change_pp"], scan["ranked"]

    if not change["material"]:
        result.status = "no_material_change"
        if ranked:                                     # offsetting moves under a flat total (spec 12.2 case 9)
            result.drivers = [Driver(ranked[0]["dimension"], t["segment"], ranked[0]["evidence_id"])
                              for t in _qualifying(ranked[0], total, False)]
        return _done(result, store, started)

    if abs(change["base_effect_pp"]) > abs(change["numerator_effect_pp"]):
        result.status, result.mechanism = "concluded", "base"
        return _done(result, store, started)

    candidates = _candidates(ranked, total)
    if not candidates:                                 # nothing concentrates: the bridge tells the kind of change
        ins = bridge["in_from_std"] + bridge["in_from_1_30"]
        cures = bridge["cured_to_std"] + bridge["cured_to_1_30"]
        result.mechanism = "flow"
        result.status = "concluded" if (ins - cures) * total > 0 else "inconclusive"
        return _done(result, store, started)

    lead = _finest_geography(pair, candidates[0], candidates, total)
    rival = next((c for c in candidates if c is not lead and are_rivals(lead["dimension"], c["dimension"])), None)
    if rival is not None:
        lead = choose_lead(lead, rival, _persists_fn(store, pair, total, scope, rival, lead))

    picks = _qualifying(lead, total, True)
    result.status = "concluded"
    result.drivers = [Driver(lead["dimension"], t["segment"], lead["evidence_id"]) for t in picks]
    first = picks[0]
    result.mechanism = mechanism_of(first)

    if first["rate_change_pp"] is not None:            # a segment new this month has nothing to drill into
        try:
            inner = drill(store, pair, lead["dimension"], first["segment"], scope=scope).summary
        except ToolError:
            inner = None
        if inner and inner["material"] and deep_allowed(inner["total_change_pp"], total, first):
            deep_candidates = _candidates(inner["ranked"], inner["total_change_pp"])
            if deep_candidates:
                top = deep_candidates[0]
                result.deep = [Driver(top["dimension"], t["segment"], top["evidence_id"])
                               for t in _qualifying(top, inner["total_change_pp"], True)[:1]]
    return _done(result, store, started)


def _done(result: InvestigationResult, store: EvidenceStore, started: float) -> tuple[InvestigationResult, EvidenceStore]:
    result.tool_calls = len(store.all())
    result.seconds = time.time() - started
    return result, store
