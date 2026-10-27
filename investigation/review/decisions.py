"""
The decision log (spec 7.3, 7.4): every choice a leader makes from the
portfolio review, the AI's recommendation next to it, and the evidence it
rested on. Kept for audit now and for checking next month whether the focus
worked (outcome tracking, later).

Durable without a database: entries are appended to a local JSON-lines file
(REVIEW_DECISION_LOG), and the whole log can be downloaded and uploaded
again, so it survives a Streamlit Cloud restart in the leader's own hands.
Only ids, region and branch names and the reviewer's short note are stored,
never a loan row. Entries are filtered by data version, so a log only shows
decisions taken on the same two files.

A repeated click, a rerun or a retry never adds a second entry: each entry
has a key (data version, review run, step, option), and appending an
existing key returns the first entry.
"""
from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

SEGMENTS = ("review", "next_step", "action", "close")
_NOTE_MAX = 300
_TEXT_MAX = 120


@dataclass
class Decision:
    data_version: str
    run: str                    # which review this came from (a hash, never the session id itself)
    segment: str                # review | next_step | action | close
    option_id: str              # e.g. investigate:collection_shortfall:NASHIK, priority_branch:MALEGAON, stop
    region: str = ""
    list_name: str = ""          # which focus list: collection_shortfall | excess_30plus | worsening
    branch: str = ""
    period: dict = field(default_factory=dict)
    ai_recommended: str | None = None
    followed_ai: bool | None = None
    evidence_ids: list = field(default_factory=list)
    approver: str = "this browser session"
    note: str = ""
    book: str = ""              # the same book across months (outcomes.book_key), unlike data_version
    baseline: dict = field(default_factory=dict)   # the candidate's figures when chosen (Candidate.baseline)
    at: str = field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S"))
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    @property
    def key(self) -> str:
        return hashlib.sha256("|".join((self.data_version, self.run, self.segment, self.option_id)).encode()).hexdigest()[:16]


_BASELINE_NUMBERS = {"gap_pts", "se_pts", "value", "benchmark", "impact"}
_BASELINE_TEXT = {"month", "unit"}


def _valid_baseline(b) -> bool:
    """Known keys only: numbers for the figures, short text for month and unit."""
    if not isinstance(b, dict) or not set(b) <= _BASELINE_NUMBERS | _BASELINE_TEXT:
        return False
    return (all(isinstance(b[k], (int, float)) and not isinstance(b[k], bool) for k in _BASELINE_NUMBERS & set(b))
            and all(isinstance(b[k], str) and len(b[k]) <= 10 for k in _BASELINE_TEXT & set(b)))


def _valid(raw: dict) -> Decision | None:
    """A Decision from an untrusted dict (an uploaded log), or None: known
    fields only, short strings only, a known step."""
    if not isinstance(raw, dict):
        return None
    try:
        d = Decision(**{k: raw[k] for k in Decision.__dataclass_fields__ if k in raw})
    except TypeError:
        return None
    texts = (d.data_version, d.run, d.segment, d.option_id, d.region, d.list_name, d.branch, d.approver, d.at, d.id)
    if d.segment not in SEGMENTS or not all(isinstance(t, str) and len(t) <= _TEXT_MAX for t in texts):
        return None
    if not isinstance(d.note, str) or len(d.note) > _NOTE_MAX or not isinstance(d.evidence_ids, list):
        return None
    if d.ai_recommended is not None and not isinstance(d.ai_recommended, str):
        return None
    if not isinstance(d.book, str) or len(d.book) > _TEXT_MAX or not _valid_baseline(d.baseline):
        return None
    d.evidence_ids = [str(e)[:_TEXT_MAX] for e in d.evidence_ids][:20]
    d.period = {k: str(v)[:10] for k, v in (d.period or {}).items() if k in ("prev", "curr")}
    return d


class DecisionLog:
    """Thread-safe, append-only, optionally backed by a JSON-lines file."""

    def __init__(self, path: str | Path | None = None) -> None:
        self._path = Path(path) if path else None
        self._lock = threading.Lock()
        self._entries: dict[str, Decision] = {}          # key -> first entry
        if self._path and self._path.exists():
            self._merge_lines(self._path.read_text(encoding="utf-8").splitlines(), persist=False)

    def append(self, **fields) -> tuple[Decision, bool]:
        """(the entry, whether it is new). An existing key returns the first entry."""
        d = Decision(**fields)
        if d.segment not in SEGMENTS:
            raise ValueError(f"unknown step '{d.segment}'")
        d.note = d.note[:_NOTE_MAX]
        with self._lock:
            if d.key in self._entries:
                return self._entries[d.key], False
            self._entries[d.key] = d
            self._write([d])
        return d, True

    def for_data(self, data_version: str) -> list[Decision]:
        """Decisions taken on these two files, oldest first."""
        with self._lock:
            return sorted((d for d in self._entries.values() if d.data_version == data_version), key=lambda d: d.at)

    def entries_for_book(self, book: str) -> list[Decision]:
        """Every decision on this book, any month, oldest first (outcome tracking)."""
        with self._lock:
            return sorted((d for d in self._entries.values() if book and d.book == book), key=lambda d: d.at)

    def export_jsonl(self, data_version: str | None = None, book: str | None = None) -> bytes:
        """The log as JSON lines, narrowed to one set of files (data_version)
        or one book across months (book): the server's file can hold other
        sessions' files, which a download must never carry."""
        with self._lock:
            return "".join(json.dumps(asdict(d), ensure_ascii=False) + "\n" for d in self._entries.values()
                           if (data_version is None or d.data_version == data_version)
                           and (book is None or d.book == book)).encode()

    def import_jsonl(self, data: bytes) -> tuple[int, int]:
        """Merge an uploaded log: (added, skipped). Bad lines are skipped, never raised."""
        lines = data.decode("utf-8", errors="replace").splitlines()
        with self._lock:
            return self._merge_lines(lines, persist=True)

    def _merge_lines(self, lines: list[str], persist: bool) -> tuple[int, int]:
        added, skipped, new = 0, 0, []
        for line in lines:
            if not line.strip():
                continue
            try:
                d = _valid(json.loads(line))
            except (json.JSONDecodeError, ValueError):
                d = None
            if d is None or d.key in self._entries:
                skipped += 1
                continue
            self._entries[d.key] = d
            new.append(d)
            added += 1
        if persist:
            self._write(new)
        return added, skipped

    def _write(self, entries: list[Decision]) -> None:
        """Append to the file; a write problem is logged, never raised (the
        in-memory log and the download still hold the entries)."""
        if not self._path or not entries:
            return
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("a", encoding="utf-8") as f:
                for d in entries:
                    f.write(json.dumps(asdict(d), ensure_ascii=False) + "\n")
        except OSError:
            logger.exception("Couldn't write the decision log to %s", self._path)
