"""
Outcome tracking (build step 5): this month's review says whether the
regions chosen in last month's review moved beyond normal variation.

Last month's figures come from the decision log (Decision.baseline: the
candidate's gap and its standard error when it was chosen), because only
this month and last month are loaded and last month's own collection gap
needed the month before. This month's come from this review's candidates.

  collection shortfall, excess 30+ (levels): the gap now against the gap
      then, with a band of INVESTIGATION_NOISE_SE x the combined error
  30+ getting worse (a change): whether this month the region moves better
      than the rest of the portfolio, beyond its own band

Descriptive only. A region that improved after it was chosen is reported as
improved; nothing here says the choice caused it, and the page says so.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

from config import INVESTIGATION_NOISE_SE
from investigation.dimensions import _text
from investigation.review.brief import LIST_NAMES, Brief, _rs
from investigation.review.prioritise import WORSE as WORSE_LIST, Candidate
from investigation.tools.change import month_word

IMPROVED, WORSE, NO_CLEAR_CHANGE, NOT_COMPARABLE = "improved", "worse", "no clear change", "not comparable"
OUTCOME_NOTE = "These compare figures before and after a choice; they don't show what caused the change."


@dataclass
class Outcome:
    region: str
    list_name: str
    status: str
    text: str
    then: dict                    # Decision.baseline: last month's figures
    now: Candidate | None         # this month's candidate for the same region and list
    change_pts: float | None
    band_pts: float | None
    handled: bool
    decided_at: str


def book_key(pair) -> str:
    """The same book from month to month: its business unit(s) and regions.
    A sidebar region filter is a different book, so a filtered review is
    never compared with a whole one."""
    regions = sorted(map(str, pair.labels("region", "curr").unique()))
    bu = sorted(map(str, _text(pair.curr["BU"]).unique())) if "BU" in pair.curr.columns else []
    return hashlib.sha256(("BU:" + ",".join(bu) + "|R:" + ",".join(regions)).encode()).hexdigest()[:12]


def _status(change: float, band: float) -> str:
    if change < -band:
        return IMPROVED
    if change > band:
        return WORSE
    return NO_CLEAR_CHANGE


_STATUS_TEXT = {IMPROVED: "That improved beyond normal month-to-month variation.",
                WORSE: "That is worse, beyond normal month-to-month variation.",
                NO_CLEAR_CHANGE: "No clear change: within normal month-to-month variation."}
_WORSE_STATUS_TEXT = {IMPROVED: "This month it moved better than the rest, beyond normal variation.",
                      WORSE: "It is still moving worse than the rest, beyond normal variation.",
                      NO_CLEAR_CHANGE: "This month it moved in line with the rest, within normal variation."}


def _outcome(d, now: Candidate | None, handled: bool, now_month: str) -> Outcome:
    then, list_name, region = d.baseline, d.list_name, d.region
    m1, m2 = month_word(then["month"]), month_word(now_month)
    head = f"{region}, {LIST_NAMES[list_name]} (chosen in the {m1} review)"
    tail = " It was marked handled." if handled else ""
    if now is None or any("fewer than" in r for r in now.reasons_not_listed):
        why = "too small to rank this month" if now is not None else "not in this month's files"
        return Outcome(region, list_name, NOT_COMPARABLE, f"{head}: can't be compared, {why}.{tail}",
                       then, now, None, None, handled, d.at)
    if list_name == WORSE_LIST:
        change, band = now.gap_pts, INVESTIGATION_NOISE_SE * (now.se_pts or 0.0)
        status = _status(change, band)
        text = (f"{head}: it moved {then['gap_pts']:+.2f} points more than the rest in {m1}, "
                f"{now.gap_pts:+.2f} in {m2}. {_WORSE_STATUS_TEXT[status]}{tail}")
    else:
        change = now.gap_pts - then["gap_pts"]
        band = INVESTIGATION_NOISE_SE * math.hypot(now.se_pts or 0.0, then["se_pts"])
        status = _status(change, band)
        text = (f"{head}: gap {then['gap_pts']:.2f} points in {m1} ({_rs(then['impact'])}), "
                f"{now.gap_pts:.2f} points in {m2} ({_rs(now.impact['amount'])}). {_STATUS_TEXT[status]}{tail}")
    return Outcome(region, list_name, status, text, then, now, round(change, 4), round(band, 4), handled, d.at)


def track_outcomes(brief: Brief, decisions: list) -> list[Outcome]:
    """One outcome per region and list chosen in last month's review of this
    book (the first choice counts; choosing it again adds nothing)."""
    last_month, this_month = brief.period.get("prev"), brief.period.get("curr")
    chosen, handled = {}, set()
    for d in sorted(decisions, key=lambda d: d.at):
        if d.book != brief.book or not d.book or d.period.get("curr") != last_month:
            continue
        if d.segment == "review" and d.option_id.startswith("investigate:") and d.baseline:
            chosen.setdefault((d.list_name, d.region), d)
        elif d.segment == "close":
            handled.add((d.list_name, d.region))
    out = []
    for (list_name, region), d in chosen.items():
        now = next((c for c in brief.candidates.get(list_name, []) if c.region == region), None)
        out.append(_outcome(d, now, (list_name, region) in handled, this_month))
    return out
