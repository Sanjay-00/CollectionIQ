"""Session-size caps for the Investigator: remembered result tables per
thread (EntityMemory) and number of threads per browser session. Both live
in st.session_state for the whole session, so unbounded they grow forever."""
import pandas as pd
import streamlit as st

from config import INVESTIGATOR_MAX_THREADS
from investigator.state import EntityMemory
from ui.tabs.investigator import _get_threads, _new_thread


class TestEntityMemoryCap:
    def test_evicts_least_recently_touched_beyond_cap(self):
        memory = EntityMemory(max_entities=3)
        for name in ("A", "B", "C"):
            memory.touch("branch", name, result_df=pd.DataFrame())
        memory.touch("branch", "A", result_df=pd.DataFrame())  # A is now most recent
        memory.touch("branch", "D", result_df=pd.DataFrame())  # over cap -> evict B
        assert memory.get("branch", "B") is None
        assert {r.entity_value for r in memory.all()} == {"A", "C", "D"}

    def test_just_touched_entity_is_never_evicted(self):
        memory = EntityMemory(max_entities=1)
        memory.touch("branch", "A", result_df=pd.DataFrame())
        memory.touch("branch", "B", result_df=pd.DataFrame())
        assert memory.most_recent().entity_value == "B"
        assert len(memory.all()) == 1

    def test_retouching_existing_entity_does_not_grow_memory(self):
        memory = EntityMemory(max_entities=2)
        for _ in range(5):
            memory.touch("branch", "A", result_df=pd.DataFrame())
        assert len(memory.all()) == 1


class TestThreadCap:
    def setup_method(self):
        st.session_state.pop("investigator_threads", None)
        st.session_state.pop("investigator_active_thread", None)

    teardown_method = setup_method

    def test_thread_count_never_exceeds_cap(self):
        for _ in range(INVESTIGATOR_MAX_THREADS + 5):
            _new_thread()
        assert len(_get_threads()) == INVESTIGATOR_MAX_THREADS

    def test_oldest_thread_is_evicted_but_active_thread_is_kept(self):
        first = _new_thread()
        st.session_state["investigator_active_thread"] = first
        second = _new_thread()
        for _ in range(INVESTIGATOR_MAX_THREADS):
            _new_thread()
        threads = _get_threads()
        assert first in threads          # active: protected
        assert second not in threads     # oldest non-active: evicted
