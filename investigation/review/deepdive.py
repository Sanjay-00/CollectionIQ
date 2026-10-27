"""
Where inside a listed region the gap sits (spec 7.1, the background run).
One recipe per list, each reusing an engine that already exists:

  collection_shortfall  the shortfall split by opening bucket (adds up exactly
                        to the region's), and each branch of the region against
                        the rest of the portfolio (like for like, not additive)
  excess_30plus         the same two splits by agreement vintage and by branch
  worsening             the Deep Investigation baseline, scoped to the region

Descriptive, never causal. Each deep-dive is an Evidence record in the
review's store, with a self-check that its parts reconcile.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from investigation.baseline import run_baseline
from investigation.dimensions import get_dimension
from investigation.evaluation import InvestigationResult
from investigation.evidence import Check, EvidenceStore, ToolError
from investigation.panel import PeriodPair
from investigation.review.prioritise import (
    EXCESS, SHORTFALL, WORSE, Candidate, ReviewResult, excess_frame, excess_table, shortfall_frame, shortfall_table,
)

TOP_PARTS = 2           # buckets / vintages and branches named per deep-dive


@dataclass
class DeepDive:
    candidate_id: str
    list: str
    region: str
    evidence_id: str
    parts: list = field(default_factory=list)       # [(label, amount)] adding up to the region's figure, largest first
    branches: list = field(default_factory=list)    # [(branch, amount)] each like for like, largest first
    unit: str = "Rs"
    investigation: InvestigationResult | None = None   # worsening only
    investigation_store: EvidenceStore | None = None
    note: str = ""                                     # why nothing more could be said


def _branches_of(pair: PeriodPair, region: str) -> list[str]:
    """Branches whose loans this month are mostly in the region."""
    df = pd.DataFrame({"b": pair.labels("branch", "curr"), "r": pair.labels("region", "curr")})
    home = df.groupby("b")["r"].agg(lambda s: s.mode().iat[0])
    return sorted(home.index[home == region])


def _top(items: list[tuple[str, float]]) -> list[tuple[str, float]]:
    return sorted((x for x in items if x[1] > 0), key=lambda x: (-x[1], x[0]))[:TOP_PARTS]


def _shortfall(review: ReviewResult, pair: PeriodPair, c: Candidate) -> DeepDive:
    strata = review.store.get(c.evidence_id).tables["strata"]
    parts = [(str(r["s"]), float(r["D"] * r["rest_rate"] - r["C"])) for r in strata.to_dict("records")]
    work, _ = shortfall_frame(pair, pair.labels("branch", "curr"))
    mine = set(_branches_of(pair, c.region))
    table = shortfall_table(work, "g", "s", "d", "c")
    branches = [(r["region"], float(r["shortfall"])) for r in table.to_dict("records") if r["region"] in mine]
    return _record(review, c, parts, branches, "shortfall by opening bucket, and each branch against the rest")


def _excess(review: ReviewResult, pair: PeriodPair, c: Candidate) -> DeepDive:
    strata = review.store.get(c.evidence_id).tables["strata"]
    parts = [(str(r["s"]), float(r["H30"] - r["H"] * r["rest_soh_rate"])) for r in strata.to_dict("records")]
    work = excess_frame(pair, pair.labels("branch", "curr"))
    mine = set(_branches_of(pair, c.region))
    table = excess_table(work, "g", "s", "y", "h")
    branches = [(r["region"], float(r["excess_soh"])) for r in table.to_dict("records") if r["region"] in mine]
    return _record(review, c, parts, branches, "excess 30+ SOH by agreement vintage, and each branch against the rest")


def _record(review: ReviewResult, c: Candidate, parts, branches, what: str) -> DeepDive:
    total = sum(v for _, v in parts)
    ev = review.store.add(f"deep_dive_{c.list}", {"region": c.region, "candidate": c.id},
                          {"what": what, "parts": parts, "branches": branches},
                          checks=[Check("parts_add_up", abs(total - c.impact["amount"]) <= 1.0,
                                        f"{total:,.2f} vs {c.impact['amount']:,.2f}")])
    return DeepDive(c.id, c.list, c.region, ev.result_id, _top(parts), _top(branches))


def _worsening(review: ReviewResult, pair: PeriodPair, c: Candidate) -> DeepDive:
    result, store = run_baseline(pair.curr, pair.prev, pair.curr_month, pair.prev_month,
                                 scope={"region": c.region}, pair=pair)
    parts = []
    for d in result.drivers:
        top = next((t for t in store.get(d.evidence_id).summary["top"] if t["segment"] == d.segment), None)
        if top is not None:
            label = get_dimension(d.dimension).label
            parts.append((f"{label[0].lower()}{label[1:]} {d.segment}", float(top["rate_change_pp"])))
    ev = review.store.add("deep_dive_worsening", {"region": c.region, "candidate": c.id},
                          {"status": result.status, "mechanism": result.mechanism, "parts": parts,
                           "investigation_tool_calls": result.tool_calls})
    note = {"inconclusive": "no single segment inside the region explains the rise",
            "no_material_change": "inside the region the change is within normal month-to-month noise",
            "halted_data": "the data checks stopped the investigation"}.get(result.status, "")
    if result.status == "concluded" and not parts:
        # No segment concentrates it: say how the loans moved (the flow
        # bridge), or that the base changed, instead of saying nothing.
        if result.mechanism == "base":
            note = "the rise is mostly fewer loans with an EMI due, not more 30+ loans"
        else:
            lines = next((e.summary["lines"] for e in store.all() if e.tool == "flow_bridge"), None)
            if lines:
                ins = lines["in_from_std"] + lines["in_from_1_30"]
                cures = lines["cured_to_std"] + lines["cured_to_1_30"]
                note = (f"no single segment stands out: across the region {ins:,} loans rolled into 30+ "
                        f"and {cures:,} cured")
    return DeepDive(c.id, WORSE, c.region, ev.result_id, parts, [], "pts", result, store, note)


_RECIPES = {SHORTFALL: _shortfall, EXCESS: _excess, WORSE: _worsening}


def deep_dive(review: ReviewResult, pair: PeriodPair, candidate: Candidate) -> DeepDive:
    """The list's recipe for one listed candidate. A tool that can't answer
    comes back as a DeepDive with a note, never as a silent zero."""
    try:
        return _RECIPES[candidate.list](review, pair, candidate)
    except ToolError as e:
        ev = review.store.add(f"deep_dive_{candidate.list}", {"region": candidate.region}, {"refused": str(e)})
        return DeepDive(candidate.id, candidate.list, candidate.region, ev.result_id, note=str(e))
