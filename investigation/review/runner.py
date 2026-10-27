"""
The background review (spec 7.1, Phase 2 of the build order): on a new
upload, check the data, score every region, look inside the top candidate of
each list, and write the template brief. Read-only, so it needs no approval
and can start by itself. Every figure is computed in code; when the caller
passes an analyst (the app does), one checked AI call writes the summary at
the end (investigation/review/analyst.py).

run_review is the whole run, synchronous (tests, scripts). ReviewRunner runs
it on one worker thread, keyed by what the caller says identifies the data
(the app's data version and sidebar filter), so a rerun or a second tab never
starts it twice. The worker never touches Streamlit: the page reads a Job's
steps and brief while it runs. Every step shares one paired dataset.
"""
from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable

import pandas as pd

from config import REVIEW_MAX_DEEP_DIVES, REVIEW_MAX_JOBS
from investigation.evidence import EvidenceStore
from investigation.panel import make_pair
from investigation.prepare import prepare
from investigation.review.brief import Brief, LIST_NAMES, build_brief, halted_brief
from investigation.review.deepdive import deep_dive
from investigation.review.outcomes import book_key
from investigation.review.prioritise import LISTS, review_regions
from utils import PREV_CARRYOVER_COLS

logger = logging.getLogger(__name__)


def _picks(review) -> list:
    """The top candidate of each list, in list order, at most REVIEW_MAX_DEEP_DIVES."""
    return [review.lists[name][0] for name in LISTS if review.lists[name]][:REVIEW_MAX_DEEP_DIVES]


def run_review(curr: pd.DataFrame, prev: pd.DataFrame, curr_month: str, prev_month: str,
               progress: Callable[[str], None] | None = None,
               analyst: Callable[[Brief], object] | None = None) -> Brief:
    """The whole background review. progress(text) is told each step as it
    ends. analyst(brief): the checked AI summary (analyst.write_analysis); off
    by default, so tests and scripts stay code only."""
    say = progress or (lambda _text: None)
    # The app's month join carries last month's figures on each loan
    # (prev_bucket, prev_SOH, ...). They aren't columns of the file, so the
    # pair (and its data check) sees this month as uploaded.
    carried = [c for c in ("prev_bucket", *PREV_CARRYOVER_COLS.values()) if c in curr.columns]
    pair = make_pair(curr.drop(columns=carried), prev, curr_month, prev_month)
    say(f"Paired the two months: {len(pair.curr):,} loans this month, {len(pair.prev):,} last month")

    prep = prepare(EvidenceStore(), pair)
    period = {"prev": pair.prev_month, "curr": pair.curr_month}
    if prep.status != "ok":
        say("Checked the data: the files can't be compared")
        return halted_brief(prep.files, period)
    say("Checked the data" + (f": {len(prep.flags)} notes" if prep.flags else ": no problems"))

    regions = sorted(pair.labels("region", "curr").unique())
    if len(regions) < 2:                 # a sidebar region filter: nothing to compare a region with
        shown = regions[0] if regions else "No region"
        return Brief("too_narrow", f"{shown} is the only region in view: clear the region filter to compare regions.",
                     period=period, options=[{"id": "stop", "label": "Stop here"}])

    review = review_regions(curr, prev, curr_month, prev_month, pair=pair)
    say(f"Scored {len(review.scorecard)} regions on {len(LISTS)} lists")

    dives = []
    for c in _picks(review):
        dives.append(deep_dive(review, pair, c))
        say(f"Looked inside {c.region} ({LIST_NAMES[c.list]})")
    brief = build_brief(review, dives, prep.files)
    brief.book = book_key(pair)
    say("Wrote the brief")
    if analyst is not None and brief.findings:
        brief.analysis = analyst(brief)
        say({"ok": "Wrote the analyst summary (AI, checked against the figures)",
             "fallback": "The AI summary didn't pass its checks: the plain brief stands",
             }.get(brief.analysis.status, "Skipped the AI summary"))
    return brief


@dataclass
class Job:
    key: str
    status: str = "running"            # running | done | failed
    steps: list = field(default_factory=list)       # [(text, seconds since start)]
    brief: Brief | None = None
    error: str = ""
    started: float = field(default_factory=time.time)
    finished: float | None = None

    @property
    def seconds(self) -> float:
        return (self.finished or time.time()) - self.started


class ReviewRunner:
    """One worker thread; the last REVIEW_MAX_JOBS jobs kept."""

    def __init__(self, max_jobs: int = REVIEW_MAX_JOBS, analyst: Callable[[Brief], object] | None = None) -> None:
        self._analyst = analyst
        self._jobs: OrderedDict[str, Job] = OrderedDict()
        self._lock = threading.Lock()
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="portfolio-review")
        self._max = max_jobs

    def get(self, key: str) -> Job | None:
        with self._lock:
            return self._jobs.get(key)

    def submit(self, key: str, curr: pd.DataFrame, prev: pd.DataFrame, curr_month: str, prev_month: str) -> Job:
        """The job for this key, started if there isn't one. A failed job is
        retried on the next submit; a running or finished one is returned as is."""
        with self._lock:
            job = self._jobs.get(key)
            if job is not None and job.status != "failed":
                self._jobs.move_to_end(key)
                return job
            job = self._jobs[key] = Job(key)
            while len(self._jobs) > self._max:
                oldest = next((k for k, j in self._jobs.items() if j.status != "running" and k != key), None)
                if oldest is None:
                    break
                del self._jobs[oldest]
        self._pool.submit(self._run, job, curr, prev, curr_month, prev_month)
        return job

    def _run(self, job: Job, curr, prev, curr_month, prev_month) -> None:
        try:
            job.brief = run_review(curr, prev, curr_month, prev_month,
                                   progress=lambda text: job.steps.append((text, round(job.seconds, 1))),
                                   analyst=self._analyst)
            job.status = "done"
        except Exception as e:                       # shown to the user, never a silent stop
            logger.exception("Portfolio review failed")
            job.error, job.status = f"{type(e).__name__}: {e}", "failed"
        finally:
            job.finished = time.time()
