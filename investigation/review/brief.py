"""
The review brief from fixed templates (no model call): a headline, one
finding per listed region and list with where it sits inside the region,
regions on several lists, data notes, caveats, and the next steps a leader
can pick. Every number is formatted from a candidate or evidence field.
Descriptive wording only: "sits in", "largest part", never a cause.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from investigation.evidence import Evidence, EvidenceStore
from investigation.review.deepdive import DeepDive
from investigation.review.prioritise import DISCLAIMERS, EXCESS, LISTS, SHORTFALL, WORSE, Candidate, ReviewResult
from investigation.tools.change import month_word
from utils import fmt_value

LIST_NAMES = {SHORTFALL: "collection shortfall", EXCESS: "excess 30+ exposure", WORSE: "30+ getting worse"}
_PART_INTRO = {SHORTFALL: "By opening bucket, the largest parts are", EXCESS: "By agreement period, the largest parts are"}
_BRANCH_INTRO = {SHORTFALL: "Largest branch shortfalls", EXCESS: "Largest branch excess"}


@dataclass
class Finding:
    list: str
    region: str
    candidate_id: str
    text: str
    evidence_ids: list = field(default_factory=list)


@dataclass
class Brief:
    status: str                    # ok | halted_data | too_narrow
    headline: str
    findings: list = field(default_factory=list)
    overlaps: list = field(default_factory=list)
    data_notes: list = field(default_factory=list)
    caveats: list = field(default_factory=list)
    options: list = field(default_factory=list)     # [{"id", "label"}], acted on by the next step
    scorecard: pd.DataFrame | None = None
    candidates: dict = field(default_factory=dict)  # list name -> every assessed candidate
    store: EvidenceStore | None = None
    deep_dives: list = field(default_factory=list)
    data_version: str = ""
    period: dict = field(default_factory=dict)
    analysis: object = None        # analyst.Analysis: the checked AI summary, when one was written
    book: str = ""                  # outcomes.book_key: the same book from month to month


def _rs(v: float) -> str:
    """Rupees with the sign in front: -₹21,848, not ₹-21,848."""
    v = float(v)
    return ("-" if v < 0 else "") + fmt_value(abs(v), "money")


def _join(items: list[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def _parts(d: DeepDive, fmt) -> str:
    out = []
    if d.parts and d.list in _PART_INTRO:
        out.append(f"{_PART_INTRO[d.list]} {_join([f'{label} ({fmt(v)})' for label, v in d.parts])}.")
    if d.branches and d.list in _BRANCH_INTRO:
        out.append(f"{_BRANCH_INTRO[d.list]}: {_join([f'{b} ({fmt(v)})' for b, v in d.branches])}.")
    return " ".join(out)


def _shortfall_text(c: Candidate, d: DeepDive | None) -> str:
    text = (f"{c.region} collected {_rs(c.impact['amount'])} less than a like-for-like book this month: "
            f"Collection% {c.value:.2f}% against {c.benchmark:.2f}% for loans in the same opening buckets elsewhere.")
    if c.raw_impact > 0 and c.mix_part is not None and abs(c.mix_part) >= 0.2 * c.raw_impact:
        if c.mix_part > 0:
            text += (f" Of the raw gap of {_rs(c.raw_impact)}, {_rs(c.mix_part)} is book mix "
                     "(more overdue loans), not weaker collection.")
        else:
            text += f" Its book is easier than average, so the raw gap understates it by {_rs(-c.mix_part)}."
    return text + (f" {_parts(d, _rs)}" if d and (d.parts or d.branches) else "")


def _excess_text(c: Candidate, d: DeepDive | None, ev: Evidence) -> str:
    text = (f"{c.region} has {_rs(c.impact['amount'])} more 30+ exposure than a like-for-like book: 30+ "
            f"{c.value:.2f}% against {c.benchmark:.2f}% for loans of the same age elsewhere "
            f"({ev.summary['excess_loans']:,.0f} loans more).")
    return text + (f" {_parts(d, _rs)}" if d and (d.parts or d.branches) else "")


def _worse_text(c: Candidate, d: DeepDive | None, ev: Evidence) -> str:
    own = ev.summary["change_pp"]
    rest = own - c.gap_pts
    text = (f"{c.region}'s 30+ rate moved {own:+.2f} points to {c.value:.2f}% while the rest of the portfolio "
            f"moved {rest:+.2f} (about {c.impact['amount']:,.0f} more 30+ loans than that pace).")
    if d and d.parts:
        text += f" Inside {c.region} it sits in {_join([f'{label} ({v:+.2f} points there)' for label, v in d.parts])}."
    elif d and d.note:
        text += f" Inside {c.region}, {d.note}."
    return text


def build_brief(review: ReviewResult, deep_dives: list[DeepDive], health: Evidence | None) -> Brief:
    dives = {d.candidate_id: d for d in deep_dives}
    period = next((c.period for v in review.assessed.values() for c in v), {})
    regions = len(review.scorecard) if review.scorecard is not None else 0
    listed = sorted({c.region for v in review.lists.values() for c in v})
    when = f"{month_word(period['curr'])} vs {month_word(period['prev'])}" if period else "This month"
    headline = (f"{when}, {regions} regions reviewed: "
                + (f"{len(listed)} stand{'s' if len(listed) == 1 else ''} out ({_join(listed)})." if listed
                   else "none stands out beyond normal month-to-month noise."))

    findings, options, caveats = [], [], []
    for name in LISTS:
        for c in review.lists[name]:
            d, ev = dives.get(c.id), review.store.get(c.evidence_id)
            text = {SHORTFALL: lambda: _shortfall_text(c, d), EXCESS: lambda: _excess_text(c, d, ev),
                    WORSE: lambda: _worse_text(c, d, ev)}[name]()
            findings.append(Finding(name, c.region, c.id, text, [c.evidence_id] + ([d.evidence_id] if d else [])))
            if d is not None:      # the regions looked into; the rest are a typed question away
                options.append({"id": f"investigate:{c.id}", "label": f"Look deeper at {c.region}: {LIST_NAMES[name]}"})
            for x in c.caveats:
                x = x if x in DISCLAIMERS else f"{c.region}: {x}"
                if x not in caveats:
                    caveats.append(x)
    options += [{"id": "scorecard", "label": "Show the full scorecard"}, {"id": "stop", "label": "Stop here"}]
    overlaps = [f"{r} is on {len(v)} lists ({_join([LIST_NAMES[n] for n in v])})."
                for r, v in sorted(review.on_lists.items())]

    notes = [f["detail"] for f in (health.summary["findings"] if health else []) if f["status"] != "pass"]
    if review.too_small:
        notes.append(f"Too small to rank (under the minimum eligible loans): {_join(review.too_small)}.")
    caveats += [x for x in review.caveats if x not in caveats]
    return Brief("ok", headline, findings, overlaps, notes, caveats, options, review.scorecard,
                 review.assessed, review.store, deep_dives, review.data_version, period)


def halted_brief(health: Evidence, period: dict) -> Brief:
    """The data checks stopped the run: the data problem is the answer."""
    notes = [f["detail"] for f in health.summary["findings"] if f["status"] == "halt"]
    headline = "The two files can't be compared yet" + (f": {notes[0]}" if notes else ".")
    return Brief("halted_data", headline, data_notes=notes, period=period,
                 options=[{"id": "stop", "label": "Stop here"}])
