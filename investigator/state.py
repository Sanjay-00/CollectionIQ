"""
Session-scoped entity memory for the Investigator drill-down assistant.

Replaces a linear "queue of steps" state model that was tried first and
broke immediately on real usage: asking about a branch, drilling down, and
then asking about an unrelated executive mid-thread had no clean home in a
single ordered list without inventing an "active investigation" vs. "side
lookup" special case. There is no such distinction here -- EntityMemory is
just a dict of entities (branch/executive/customer), each holding its own
last-computed result, and reference resolution ("him," "that branch")
always resolves to whichever entity has the most recent touched_at. A new,
unrelated question just adds a new entry; it cannot corrupt an existing one.

NOTE (forward-looking, no action needed today): this is designed to live in
st.session_state because the current phase is Streamlit-only, which already
gives it session-scoped persistence across turns -- no LangGraph checkpointer
is used (see the Investigator plan doc for the full reasoning: a checkpointer
earns its place on cross-process durability, resumable runs, or a caller
outside the Streamlit process, none of which apply yet). If this feature
ever moves behind a non-Streamlit caller (e.g. FastAPI, per the portability
already confirmed for analysis/agents/compiler/registry), EntityMemory needs
to become a LangGraph-checkpointed, thread_id-keyed object instead of a
plain in-process dict -- flagged now so this interface isn't shaped in a way
that's painful to peel off later.
"""
from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass
class EntityRecord:
    entity_type: str
    entity_value: Any
    result_df: pd.DataFrame
    display_cols: list[str] = field(default_factory=list)
    metric_focus: str | None = None
    narrative: str = ""
    # Which STEP_REGISTRY function produced result_df -- lets a renderer
    # decide what "Analyse further" means for this entity without guessing
    # from metric_focus (a metric NAME, not a step type -- conflating the two
    # was a real bug caught in review before this shipped).
    step_type: str = ""
    # The EXACT step params that produced result_df -- lets a later "refine"
    # turn (e.g. "same as this, but metric=X") splice stored params
    # together deterministically instead of asking the LLM to re-derive
    # scope_col/scope_value/etc from a text description every time, which
    # is exactly the re-derivation step that produced a real misroute (see
    # investigator/llm.py's STAY-AT-GRAIN rule docstring for the incident).
    params: dict = field(default_factory=dict)
    touched_at: float = field(default_factory=time.time)
    # Monotonic tie-breaker for recency ordering -- wall-clock time.time()
    # has real-world resolution too coarse to guarantee strict ordering
    # between two touch() calls microseconds apart (confirmed: two rapid
    # touches produced an identical touched_at and most_recent() picked the
    # WRONG one via max()'s first-seen tiebreak). touched_at is kept for
    # display/debugging; ordering always uses this sequence number instead.
    _seq: int = field(default=0, compare=False)


class EntityMemory:
    """dict-like store keyed by (entity_type, entity_value)."""

    def __init__(self) -> None:
        self._entities: dict[tuple[str, Any], EntityRecord] = {}
        self._seq_counter = itertools.count()

    def touch(
        self,
        entity_type: str,
        entity_value: Any,
        result_df: pd.DataFrame,
        display_cols: list[str] | None = None,
        metric_focus: str | None = None,
        narrative: str = "",
        step_type: str = "",
        params: dict | None = None,
    ) -> EntityRecord:
        """Add a new entity, or refresh an existing one to be the most
        recently touched -- either way, other entities are untouched."""
        record = EntityRecord(
            entity_type=entity_type,
            entity_value=entity_value,
            result_df=result_df,
            display_cols=display_cols or [],
            metric_focus=metric_focus,
            narrative=narrative,
            step_type=step_type,
            params=params or {},
            _seq=next(self._seq_counter),
        )
        self._entities[self._key(entity_type, entity_value)] = record
        return record

    def get(self, entity_type: str, entity_value: Any) -> EntityRecord | None:
        return self._entities.get(self._key(entity_type, entity_value))

    def most_recent(self) -> EntityRecord | None:
        if not self._entities:
            return None
        return max(self._entities.values(), key=lambda r: r._seq)

    def all(self) -> list[EntityRecord]:
        """Every entity, most-recently-touched first."""
        return sorted(self._entities.values(), key=lambda r: r._seq, reverse=True)

    def clear(self) -> None:
        self._entities.clear()

    @staticmethod
    def _key(entity_type: str, entity_value: Any) -> tuple[str, Any]:
        if isinstance(entity_value, list):
            entity_value = tuple(entity_value)
        return (entity_type, entity_value)
