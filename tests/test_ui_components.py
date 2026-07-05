"""ui/components.py::_dl_btn caching.

Regression: the cache used to key on (key, id(df), len(df)). id() is a memory
address that CPython can reuse once the original object is garbage-collected,
so two different same-length DataFrames could in theory collide on id() and
serve a stale download. Fixed by keying the cache on `key` alone and holding
a strong reference to the cached df, validated with `is` - holding that
reference is what prevents the id-reuse scenario from being possible at all.
"""
import warnings

import pandas as pd
import streamlit as st

from ui.components import _dl_btn, _load_and_concat
from test_utils import _build_upload

warnings.filterwarnings("ignore", message="Session state does not function")


class TestDlBtnCaching:
    def setup_method(self):
        st.session_state["_excel_bytes_cache"] = {}

    def test_same_object_is_a_cache_hit(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        _dl_btn(df, "f.xlsx", "k1")
        cached_after_first = st.session_state["_excel_bytes_cache"]["k1"]

        _dl_btn(df, "f.xlsx", "k1")
        cached_after_second = st.session_state["_excel_bytes_cache"]["k1"]

        assert cached_after_first[0] is df
        # Same bytes object reused, not recomputed, on the second call.
        assert cached_after_second[1] is cached_after_first[1]

    def test_different_object_same_length_is_not_served_stale(self):
        df1 = pd.DataFrame({"a": [1, 2, 3]})
        df2 = pd.DataFrame({"a": [9, 9, 9]})  # same length, different content
        _dl_btn(df1, "f.xlsx", "k1")
        bytes_for_df1 = st.session_state["_excel_bytes_cache"]["k1"][1]

        _dl_btn(df2, "f.xlsx", "k1")
        entry = st.session_state["_excel_bytes_cache"]["k1"]

        assert entry[0] is df2
        assert entry[1] != bytes_for_df1

    def test_one_entry_per_key_no_accumulation(self):
        df1 = pd.DataFrame({"a": [1]})
        df2 = pd.DataFrame({"a": [2]})
        _dl_btn(df1, "f.xlsx", "k1")
        _dl_btn(df2, "f.xlsx", "k1")
        assert list(st.session_state["_excel_bytes_cache"].keys()) == ["k1"]

    def test_distinct_keys_cache_independently(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        _dl_btn(df, "f1.xlsx", "k1")
        _dl_btn(df, "f2.xlsx", "k2")
        assert set(st.session_state["_excel_bytes_cache"].keys()) == {"k1", "k2"}


class TestLoadAndConcatDuplicateLoanCount:
    """_load_and_concat aggregates load_and_validate's per-file
    dropped_duplicate_loans count (df.attrs), plus any additional cross-file
    duplicate Loan Nos found when concatenating multiple region files -
    a second, previously equally-silent dedup step this function does itself.

    _load_and_concat calls the real @st.cache_data-wrapped load_and_validate
    (not .__wrapped__), whose hasher calls os.path.getmtime on the upload's
    .name - so unlike test_utils.py's in-memory BytesIO helper, these need a
    real file on disk."""

    def _real_upload(self, tmp_path, name, overrides):
        buf = _build_upload(overrides)
        path = tmp_path / name
        path.write_bytes(buf.getvalue())
        return open(path, "rb")  # real path on disk -> .name is already usable

    def test_single_file_no_duplicates(self, tmp_path):
        f = self._real_upload(tmp_path, "a.xlsx", {"Loan No": ["L1", "L2", "L3"]})
        result_df, errs = _load_and_concat(f)
        assert errs == []
        assert result_df.attrs.get("dropped_duplicate_loans", 0) == 0

    def test_single_file_with_duplicates_reports_count(self, tmp_path):
        f = self._real_upload(tmp_path, "a.xlsx", {"Loan No": ["L1", "L1", "L2"]})
        result_df, errs = _load_and_concat(f)
        assert errs == []
        assert result_df.attrs["dropped_duplicate_loans"] == 1

    def test_cross_file_duplicate_is_also_counted(self, tmp_path):
        f1 = self._real_upload(tmp_path, "a.xlsx", {"Loan No": ["L1", "L2", "L3"]})
        f2 = self._real_upload(tmp_path, "b.xlsx", {"Loan No": ["L3", "L4", "L5"]})  # L3 collides
        result_df, errs = _load_and_concat([f1, f2])
        assert errs == []
        assert len(result_df) == 5  # L3 kept once
        assert result_df.attrs["dropped_duplicate_loans"] == 1
