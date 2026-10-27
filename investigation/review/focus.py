"""
Focus cards for the Investigator's briefing-and-workspace layout: one card per
region that stands out in the review, with every finding about it, the lens
to open it with, and where the leader is with it (from the decision log).

Order, so the first card is where to start: the AI's pick (when the checked
analyst summary made one), then regions on more lists, then larger amounts.
Pure: no Streamlit, so the grouping and order are tested on their own.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from investigation.review.brief import LIST_NAMES, Brief, _rs

OPEN, LOOKED_INTO, HANDLED = "Open", "Looked into", "Handled"


@dataclass
class Focus:
    region: str
    findings: list                       # brief.Finding, in the brief's list order
    lists: list                          # list names, same order
    lens: str                            # candidate id to open first
    ai_pick: bool = False
    status: str = OPEN
    chips: list = field(default_factory=list)    # [(list label, amount text)]


def _amount(brief: Brief, candidate_id: str) -> tuple[float, str]:
    name = candidate_id.split(":", 1)[0]
    c = next((c for c in brief.candidates.get(name, []) if c.id == candidate_id), None)
    if c is None:
        return 0.0, ""
    if c.impact["unit"] == "Rs":
        return float(c.impact["amount"]), _rs(c.impact["amount"])
    return 0.0, f"{c.impact['amount']:,.0f} loans"


def focuses(brief: Brief, decisions: list) -> list[Focus]:
    """One Focus per region in the brief's findings. decisions: this set of
    files' log entries (DecisionLog.for_data)."""
    a = getattr(brief, "analysis", None)
    pick = a.recommended_option.split(":", 1)[1] if a is not None and a.status == "ok" and a.recommended_option else None
    dived = {d.candidate_id for d in brief.deep_dives}
    status = {}
    for d in decisions:
        if d.data_version != brief.data_version or not d.region:
            continue
        if d.segment == "close":
            status[d.region] = HANDLED
        elif status.get(d.region) != HANDLED:
            status[d.region] = LOOKED_INTO
    by_region: dict[str, list] = {}
    for f in brief.findings:
        by_region.setdefault(f.region, []).append(f)
    cards = []
    for region, found in by_region.items():
        ids = [f.candidate_id for f in found]
        lens = pick if pick in ids else next((i for i in ids if i in dived), ids[0])
        amounts = [_amount(brief, i) for i in ids]
        cards.append(Focus(region, found, [f.list for f in found], lens, pick in ids, status.get(region, OPEN),
                           [(LIST_NAMES[f.list], text) for f, (_, text) in zip(found, amounts)]))
    rs = {c.region: max((_amount(brief, f.candidate_id)[0] for f in c.findings), default=0.0) for c in cards}
    return sorted(cards, key=lambda c: (not c.ai_pick, -len(c.lists), -rs[c.region], c.region))
