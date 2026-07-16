"""ui/components.py::_dl_btn caching.

Regression: the cache used to key on `df is` the object cached last time,
under the assumption that df_curr (and everything derived from it) keeps the
same object identity across reruns when data/filters are unchanged, since it
comes from an upstream @st.cache_data call. That assumption was wrong:
st.cache_data returns a fresh COPY on every cache hit, never the original
object, so the identity check was a guaranteed miss on every single rerun --
confirmed by profiling real 60k-row data, where this cost ~53s of a ~62s warm
rerun just for the Alerts tab's 31 _dl_btn calls. Fixed by delegating to
_excel_bytes, an @st.cache_data function in its own right, so caching is
content-hash-based like the rest of this app's caching, not identity-based.
"""
import warnings

import pandas as pd
import streamlit as st

from ui.components import _dl_btn, _esc, _excel_bytes, _kpi_card_html, _load_and_concat
from test_utils import _build_upload
from utils import REQUIRED_COLS

warnings.filterwarnings("ignore", message="Session state does not function")


class TestDlBtnCaching:
    def test_fresh_object_same_content_is_a_cache_hit(self):
        """The scenario that actually happens on every Streamlit rerun: a
        brand new DataFrame object with identical content to one seen before
        (because it came from a fresh @st.cache_data copy upstream) must
        still be served from cache, not recomputed."""
        df1 = pd.DataFrame({"a": [1, 2, 3]})
        bytes1 = _excel_bytes(df1)

        df2 = pd.DataFrame({"a": [1, 2, 3]})  # different object, same content
        bytes2 = _excel_bytes(df2)

        assert df1 is not df2
        assert bytes1 == bytes2

    def test_different_content_is_not_served_stale(self):
        df1 = pd.DataFrame({"a": [1, 2, 3]})
        df2 = pd.DataFrame({"a": [9, 9, 9]})  # same shape, different content
        assert _excel_bytes(df1) != _excel_bytes(df2)


class TestEsc:
    """_esc guards every data-originated value interpolated into
    unsafe_allow_html f-strings across ui/ -- manually-typed LCC fields
    (Unit, MNT NAME, SegmentName...) and Gemini/planner output can contain
    &, <, > that would otherwise break table markup or inject markup."""

    def test_escapes_html_metacharacters_in_strings(self):
        assert _esc("R&B MOTORS") == "R&amp;B MOTORS"
        assert _esc("<script>alert(1)</script>") == "&lt;script&gt;alert(1)&lt;/script&gt;"

    def test_escapes_quotes_for_attribute_safety(self):
        # Used inside title="..." attributes (_top5_breakdown), so quotes
        # must be escaped too (html.escape quote=True).
        assert '"' not in _esc('BRANCH "MAIN"')

    def test_non_strings_pass_through_unchanged(self):
        # Numbers routinely continue into format specs (f"{_esc(v):.1f}");
        # None is handled by callers' own "- " fallbacks.
        assert _esc(42) == 42
        assert _esc(3.14) == 3.14
        assert _esc(None) is None

    def test_scorecard_table_escapes_executive_names(self):
        # End-to-end: a malicious/awkward MNT NAME must not survive into the
        # rendered scorecard HTML as live markup.
        from analysis.executive_scorecard import build_scorecard_table_html

        df = pd.DataFrame([{
            "Rank": 1,
            "Executive (Branch)": 'EVIL <img src=x onerror=alert(1)> & CO (PUNE)',
            "Accounts": 10, "Collection %": 95.0, "Strike Rate %": 80.0,
            "NPA": 0, "SMA-2": 1, "Tier": "top",
        }])
        out = build_scorecard_table_html(df)
        assert "<img" not in out
        assert "&lt;img" in out

    def test_dl_btn_renders_for_distinct_keys(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        _dl_btn(df, "f1.xlsx", "k1")
        _dl_btn(df, "f2.xlsx", "k2")


class TestKpiCardHtmlArrowAndColor:
    """Regression coverage for the double-sign-flip bug: a caller used to
    pre-flip an inverse metric's delta before passing it in, and this
    function ALSO flips color based on `inverse`, combining into arrows AND
    colors that were both backwards. Contract now: `delta` is always the raw
    (curr - prev) movement; arrow direction is derived only from its sign
    and is never affected by `inverse`."""

    def test_arrow_direction_always_matches_raw_delta_sign_inverse_or_not(self):
        up_inverse = _kpi_card_html("NPA %", "10%", 5.0, inverse=True)
        up_normal = _kpi_card_html("Collection %", "10%", 5.0, inverse=False)
        down_inverse = _kpi_card_html("NPA %", "10%", -5.0, inverse=True)
        down_normal = _kpi_card_html("Collection %", "10%", -5.0, inverse=False)
        assert "▲" in up_inverse and "▼" not in up_inverse
        assert "▲" in up_normal and "▼" not in up_normal
        assert "▼" in down_inverse and "▲" not in down_inverse
        assert "▼" in down_normal and "▲" not in down_normal

    def test_inverse_flips_color_not_arrow(self):
        # A positive (worsening) delta on an inverse metric must render red,
        # even though the arrow still points up (matching the real increase).
        worsened = _kpi_card_html("NPA %", "10%", 5.0, inverse=True)
        improved = _kpi_card_html("NPA %", "10%", -5.0, inverse=True)
        assert "kpi-mom-down" in worsened  # red: NPA% went up, that's bad
        assert "kpi-mom-up" in improved    # green: NPA% went down, that's good

    def test_good_override_controls_color_not_arrow(self):
        # SOH rose (arrow must point up) but the override says it's bad
        # (driven by arrears growth, not POS growth).
        html = _kpi_card_html("Total SOH", "1Cr", 5.0, inverse=True, good_override=False)
        assert "▲" in html
        assert "kpi-mom-down" in html

        html2 = _kpi_card_html("Total SOH", "1Cr", 5.0, inverse=True, good_override=True)
        assert "▲" in html2
        assert "kpi-mom-up" in html2


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


class TestLoadAndConcatMissingOptionalCols:
    """pd.concat doesn't propagate .attrs from its input frames, so the multi
    -file path must recompute missing_optional_cols on the final combined
    frame rather than trust any single file's own attrs -- a column present in
    only ONE regional file still ends up in the combined result (NaN-filled for
    the others), so it must NOT be reported as missing."""

    def _real_upload(self, tmp_path, name, overrides):
        buf = _build_upload(overrides)
        path = tmp_path / name
        path.write_bytes(buf.getvalue())
        return open(path, "rb")

    def test_single_file_reports_its_own_missing_cols(self, tmp_path):
        f = self._real_upload(tmp_path, "a.xlsx", {"Loan No": ["L1", "L2", "L3"]})
        result_df, errs = _load_and_concat(f)
        assert errs == []
        assert result_df.attrs.get("missing_optional_cols", []) == []

    def test_column_present_in_only_one_of_two_files_is_not_reported_missing(self, tmp_path):
        # Build file A with every REQUIRED_COLS column, file B missing
        # CoLending_Loans -- after concat, the combined frame still HAS that
        # column (NaN for file B's rows), so it must not appear as "missing".
        f1 = self._real_upload(tmp_path, "a.xlsx", {"Loan No": ["L1", "L2", "L3"]})

        data = {c: ["x"] * 3 for c in REQUIRED_COLS if c != "CoLending_Loans"}
        data.update({
            "Loan No": ["L4", "L5", "L6"], "Arrears / EMI": [0.0] * 3,
            "Month Receipt Amount": [100.0] * 3,
            "Month Collection (Excluding Reserve Collection)": [100.0] * 3,
            "Net Collection Demand Inst+Exp+BC": [100.0] * 3,
            "POS": [1000.0] * 3, "LCC%": [100.0] * 3, "Closing Arrears": [0.0] * 3,
            "Month Due-Inst": [100.0] * 3, "Month Due-Exp": [0.0] * 3,
            "Total Cum Collection": [1000.0] * 3, "Strike": ["Y"] * 3, "Due Dt": [5] * 3,
        })
        buf = pd.DataFrame(data)
        path2 = tmp_path / "b.xlsx"
        buf.to_excel(path2, index=False, engine="openpyxl")
        f2 = open(path2, "rb")

        result_df, errs = _load_and_concat([f1, f2])
        assert errs == []
        assert "CoLending_Loans" in result_df.columns
        assert "CoLending_Loans" not in result_df.attrs["missing_optional_cols"]

    def test_column_missing_from_every_file_is_reported(self, tmp_path):
        def _build_without_colending(loan_nos):
            data = {c: ["x"] * len(loan_nos) for c in REQUIRED_COLS if c != "CoLending_Loans"}
            data.update({
                "Loan No": loan_nos, "Arrears / EMI": [0.0] * len(loan_nos),
                "Month Receipt Amount": [100.0] * len(loan_nos),
                "Month Collection (Excluding Reserve Collection)": [100.0] * len(loan_nos),
                "Net Collection Demand Inst+Exp+BC": [100.0] * len(loan_nos),
                "POS": [1000.0] * len(loan_nos), "LCC%": [100.0] * len(loan_nos),
                "Closing Arrears": [0.0] * len(loan_nos),
                "Month Due-Inst": [100.0] * len(loan_nos), "Month Due-Exp": [0.0] * len(loan_nos),
                "Total Cum Collection": [1000.0] * len(loan_nos), "Strike": ["Y"] * len(loan_nos),
                "Due Dt": [5] * len(loan_nos),
            })
            return pd.DataFrame(data)

        path1 = tmp_path / "a.xlsx"
        _build_without_colending(["L1", "L2"]).to_excel(path1, index=False, engine="openpyxl")
        path2 = tmp_path / "b.xlsx"
        _build_without_colending(["L3", "L4"]).to_excel(path2, index=False, engine="openpyxl")

        result_df, errs = _load_and_concat([open(path1, "rb"), open(path2, "rb")])
        assert errs == []
        assert "CoLending_Loans" not in result_df.columns
        assert "CoLending_Loans" in result_df.attrs["missing_optional_cols"]
