"""AI Query plan cache (code review item 3, 2026-10-27).

The planner's one Gemini call is about 90% of an AI Query's time, and its
plan (IR-1) depends only on the question and the planner's system prompt,
not on the rows of the file. So a plan that answered a question is kept and
reused for the same question, in any session, with no AI call.

The key is the full system prompt plus the question, so anything that changes
what the planner would see (reporting months, whether last month is loaded,
the registry vocabulary, the model) is automatically a different key; nothing
needs invalidating by hand. graph.run_query decides what goes in: only plans
whose whole query worked, never errors or clarifications, and a reused plan
that fails is dropped.

In memory and process wide (shared by every browser session), bounded, oldest
out first, and thread safe: Streamlit serves each session on its own thread."""
import copy
import hashlib
import re
import threading
from collections import OrderedDict

from config import GEMINI_MODEL, PLAN_CACHE_MAX, PLANNER_TEMPERATURE

_SPACES = re.compile(r"\s+")


def planner_cache_key(query: str, snapshot_dates: dict | None, allow_clarification: bool) -> str:
    """Spacing is ignored; case is not (a branch or executive name in a filter
    can depend on it, and a cache must never change an answer)."""
    from agents.logical_planner import _build_full_system_prompt
    prompt = _build_full_system_prompt(snapshot_dates, allow_clarification)
    question = _SPACES.sub(" ", query).strip()
    raw = "\x00".join([GEMINI_MODEL, str(PLANNER_TEMPERATURE), prompt, question])
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class PlanCache:
    def __init__(self, max_entries: int = PLAN_CACHE_MAX):
        self._max = max_entries
        self._entries: OrderedDict[str, dict] = OrderedDict()
        self._lock = threading.Lock()
        self._hits = self._misses = 0

    def get(self, key: str) -> dict | None:
        """A copy, so the caller can't change the kept plan."""
        with self._lock:
            plan = self._entries.get(key)
            if plan is None:
                self._misses += 1
                return None
            self._entries.move_to_end(key)
            self._hits += 1
            return copy.deepcopy(plan)

    def put(self, key: str, plan: dict) -> None:
        with self._lock:
            self._entries[key] = copy.deepcopy(plan)
            self._entries.move_to_end(key)
            while len(self._entries) > self._max:
                self._entries.popitem(last=False)

    def drop(self, key: str) -> None:
        with self._lock:
            self._entries.pop(key, None)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self._hits = self._misses = 0

    def stats(self) -> dict:
        with self._lock:
            return {"hits": self._hits, "misses": self._misses, "entries": len(self._entries)}

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)


PLAN_CACHE = PlanCache()
