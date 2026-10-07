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

from ui.components import _dl_btn, _esc, _excel_bytes, _kpi_card_html, _load_and_concat, query_confidence_tier, _confidence_badge_html
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

    def test_executive_table_escapes_executive_names(self):
        # End-to-end: a malicious/awkward MNT NAME must not survive into the
        # rendered executive table (Portfolio Intelligence) as live markup.
        from ui.tabs.portfolio_intelligence import units_table_html

        df = pd.DataFrame([{
            "Executive": 'EVIL <img src=x onerror=alert(1)> & CO', "Branch": "PUNE",
            "Accounts": 10, "Collection%": 95.0, "Strike%": 80.0,
        }])
        out = units_table_html(df, "Executive", extra=["Branch"])
        assert "<img" not in out
        assert "&lt;img" in out

    def test_dl_btn_renders_for_distinct_keys(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        _dl_btn(df, "f1.xlsx", "k1")
        _dl_btn(df, "f2.xlsx", "k2")


class TestFileFingerprintMemo:
    """app.py fingerprints the uploads on every rerun; hashing a large file
    each click was pure latency. Memoized per Streamlit file_id + size."""

    class _Upload:
        def __init__(self, data: bytes, file_id: str):
            self.data, self.file_id, self.size, self.reads = data, file_id, len(data), 0

        def getvalue(self):
            self.reads += 1
            return self.data

    def setup_method(self):
        st.session_state.pop("_file_fingerprint_memo", None)

    teardown_method = setup_method

    def test_same_upload_is_hashed_once(self):
        from ui.components import _file_fingerprint
        f = self._Upload(b"lcc-bytes", "id-1")
        assert _file_fingerprint(f) == _file_fingerprint(f)
        assert f.reads == 1

    def test_new_upload_is_hashed_and_differs(self):
        from ui.components import _file_fingerprint
        a, b = self._Upload(b"month-a", "id-a"), self._Upload(b"month-b", "id-b")
        assert _file_fingerprint(a) != _file_fingerprint(b)
        assert (a.reads, b.reads) == (1, 1)

    def test_objects_without_file_id_are_always_hashed(self):
        from io import BytesIO
        from ui.components import _file_fingerprint
        f = BytesIO(b"plain")
        assert _file_fingerprint(f) == _file_fingerprint(BytesIO(b"plain"))


class TestDlBtnFullSource:
    """Curated on screen, full raw + computed columns in the download."""

    def _capture(self, monkeypatch):
        """The button gets a function (built on click); "clicking" runs it."""
        seen = {}

        def fake_excel_bytes(df):
            seen["df"] = df
            return b"x"

        def fake_button(label, data=None, **_):
            assert callable(data), "downloads must be built on click, not on every page load"
            assert "df" not in seen            # nothing built before the click
            assert data() == b"x"

        monkeypatch.setattr("ui.components._excel_bytes", fake_excel_bytes)
        monkeypatch.setattr("ui.components.st.download_button", fake_button)
        return seen

    def test_download_carries_raw_and_computed_columns(self, monkeypatch):
        seen = self._capture(monkeypatch)
        source = pd.DataFrame({"Loan No": ["L1", "L2"], "CHANNEL": ["A", "B"], "POS": [1.0, 2.0]})
        curated = pd.DataFrame({"Loan No": ["L2"], "Tenure Completed %": [75.0]})
        _dl_btn(curated, "g.xlsx", "k_full", full_source=source)
        out = seen["df"]
        assert list(out.columns) == ["Loan No", "CHANNEL", "POS", "Tenure Completed %"]
        assert out.iloc[0]["CHANNEL"] == "B"

    def test_without_full_source_download_is_unchanged(self, monkeypatch):
        seen = self._capture(monkeypatch)
        curated = pd.DataFrame({"Loan No": ["L1"], "SOH": [1.0]})
        _dl_btn(curated, "g.xlsx", "k_plain")
        assert list(seen["df"].columns) == ["Loan No", "SOH"]


class TestQueryConfidenceTier:
    """query_confidence_tier reads signals already present in graph.py's final
    QueryState (view_render/priority_mode/repair_attempts) to classify how
    much validation an AI Query answer received -- fast-path view (same code
    as the dashboard) ranks above a compiler plan that needed a repair pass.
    Precedence matters: a view match or priority mode short-circuits before
    repair_attempts is even considered, since neither of those paths compiles
    an IR-1 in the first place."""

    def test_view_match_is_verified_regardless_of_other_fields(self):
        label, _, _ = query_confidence_tier({"view_render": "kpi_cards", "repair_attempts": 3})
        assert label == "Verified"

    def test_priority_mode_ranks_below_view_but_above_compiler_signals(self):
        label, _, _ = query_confidence_tier({"priority_mode": True, "repair_attempts": 2})
        assert label == "Priority Rules"

    def test_clean_compile_is_validated(self):
        label, _, _ = query_confidence_tier({"repair_attempts": 0})
        assert label == "Validated"

    def test_repaired_compile_is_self_corrected(self):
        label, _, _ = query_confidence_tier({"repair_attempts": 1})
        assert label == "Self-corrected"

    def test_missing_repair_attempts_key_defaults_to_validated(self):
        # A direct/legacy caller that never set repair_attempts must not be
        # misread as "needed a repair" -- absence means "no repair recorded",
        # same as the explicit 0 case.
        label, _, _ = query_confidence_tier({})
        assert label == "Validated"

    def test_badge_html_escapes_nothing_user_controlled_but_renders_label(self):
        html_out = _confidence_badge_html({"view_render": "kpi_cards"})
        assert "Verified" in html_out


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


def _in_memory(path):
    """An upload-like file object (bytes + .name) that needs no closing."""
    import io
    from pathlib import Path
    buf = io.BytesIO(Path(path).read_bytes())
    buf.name = str(path)
    return buf


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
        return _in_memory(path)  # real path on disk -> .name is already usable

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
        return _in_memory(path)

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
        f2 = _in_memory(path2)

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

        result_df, errs = _load_and_concat([_in_memory(path1), _in_memory(path2)])
        assert errs == []
        assert "CoLending_Loans" not in result_df.columns
        assert "CoLending_Loans" in result_df.attrs["missing_optional_cols"]


class TestMomChangeCells:
    """Month-on-month change in the shared table (Business tab). A Total row
    can carry "" for a % it can't sum; that once crashed the whole Business
    tab, so it must render as a blank, and a rise is green (more business is
    good), a fall red."""

    def _html(self, rows, total=None):
        from ui.components import html_table
        return html_table(pd.DataFrame(rows), [
            {"key": "Branch"}, {"key": "Funded (Cr)", "fmt": "cr"},
            {"key": "Accounts MoM %", "fmt": "pct_change", "good_if_up": True}], total=total)

    def test_blank_and_missing_values_render_as_a_dash(self):
        html = self._html([{"Branch": "A", "Funded (Cr)": 0.11, "Accounts MoM %": None}],
                          total={"Branch": "Total", "Funded (Cr)": 0.11, "Accounts MoM %": ""})
        assert html.count('color:#9ca3af;">-</span>') == 2

    def test_rise_is_green_fall_is_red(self):
        html = self._html([{"Branch": "A", "Funded (Cr)": 1.0, "Accounts MoM %": 9.49},
                           {"Branch": "B", "Funded (Cr)": 1.0, "Accounts MoM %": -3.2}])
        assert "▲ 9.5%" in html and "▼ 3.2%" in html
        a, b = html.split("▲ 9.5%")[0].rsplit("<td", 1)[1], html.split("▼ 3.2%")[0].rsplit("<td", 1)[1]
        assert "#16a34a" in a and "#dc2626" in b

    def test_amounts_show_two_decimals(self):
        html = self._html([{"Branch": "A", "Funded (Cr)": 0.11, "Accounts MoM %": 1.0}])
        assert "₹0.11 Cr" in html and "0.110000" not in html


def test_every_table_on_screen_goes_through_safe_df():
    # _safe_df strips the meaningless 00:00:00 from LCC dates (via _dateonly);
    # a table shown without it prints "2023-09-23 00:00:00".
    import re
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    raw = []
    for f in [root / "app.py", *sorted((root / "ui").rglob("*.py"))]:
        text = f.read_text(encoding="utf-8-sig")
        for m in re.finditer(r"^\s*st\.dataframe\(\s*([A-Za-z_][\w.]*)", text, re.MULTILINE):
            arg = m.group(1)
            if arg not in ("_safe_df", "styled_df"):   # styled_df applies _safe_df
                raw.append(f"{f.relative_to(root)}: {m.group(0)}")
    assert raw == []


def test_safe_df_drops_midnight_time_from_dates():
    import pandas as pd
    from ui.components import _safe_df
    df = pd.DataFrame({"Ag_Date": pd.to_datetime(["2023-09-23"]).astype("datetime64[us]")})
    assert str(_safe_df(df)["Ag_Date"].iloc[0]) == "2023-09-23"


def test_percent_and_count_share_one_cell():
    from ui.components import html_table, pct_count
    assert pct_count(11.94, 37) == '11.9% <span style="color:#6b7280;font-weight:400;">(37)</span>'
    assert "-" in pct_count(None, 3)
    df = pd.DataFrame({"Branch": ["A"], "NPA%": [12.5], "NPA": [5]})
    html = html_table(df, [{"key": "Branch"}, {"key": "NPA%", "label": "NPA", "fmt": "pct_count", "count": "NPA"}])
    assert "12.5% <span" in html and "(5)" in html and html.count("</th>") == 2   # one column, not two


def test_percent_by_soh_and_its_crore_share_one_cell():
    from ui.components import html_table, pct_amount
    assert pct_amount(34.62, 121.4) == '34.6% <span style="color:#6b7280;font-weight:400;">(₹121.40 Cr)</span>'
    assert "-" in pct_amount(None, 1.0)
    df = pd.DataFrame({"Branch": ["A"], "NPA% (SOH)": [8.0], "NPA SOH (Cr)": [1.5]})
    html = html_table(df, [{"key": "Branch"},
                           {"key": "NPA% (SOH)", "fmt": "pct_cr", "amount": "NPA SOH (Cr)"}])
    assert "8.0% <span" in html and "(₹1.50 Cr)" in html and html.count("</th>") == 2


def test_top_x_in_each_branch(monkeypatch):
    import ui.components as comp
    df = pd.DataFrame({"Branch": ["A"] * 5 + ["B"] * 5, "SOH": list(range(10, 0, -1))})
    picks = {"pi_t_n": 2, "pi_t_b": "All"}
    monkeypatch.setattr(comp.st, "columns", lambda spec: [_Fake(picks)] * 3)
    monkeypatch.setattr(comp.st, "caption", lambda *a, **k: None)
    out = comp.list_controls("pi_t", df, "Branch")
    assert out.groupby("Branch").size().to_dict() == {"A": 2, "B": 2}      # top 2 from EACH branch
    assert out["SOH"].tolist() == [10, 9, 5, 4]


class _Fake:
    """Stands in for a Streamlit column: selectbox returns a preset, the
    'each branch' checkbox is ticked."""
    def __init__(self, picks):
        self.picks = picks

    def selectbox(self, label, options, index=0, key=None):
        return self.picks.get(key, options[index])

    def checkbox(self, label, key=None, disabled=False):
        return True

    def markdown(self, *a, **k):
        return None


def test_dashboard_cards_move_percent_metrics_in_points():
    # A % metric's card shows the change in points, like every other tab and
    # the report (59.00% from 58.36% is "0.64 pts", not the relative "1.10%").
    from ui.tabs.dashboard import _change
    metrics = {"Delinquency %": (59.0, 1.1), "Count": (8000, 0.38)}
    prev = {"Delinquency %": (58.36, None), "Count": (7970, None)}
    assert _change("Delinquency %", metrics, prev) == (0.64, " pts")
    assert _change("Delinquency %", metrics, None) == (None, " pts")       # no last month: "no prev data"
    assert _change("Count", metrics, prev) == (0.38, "%")                   # counts keep the % change


def test_segments_group_by_loan_attributes():
    # Payment mode, NACH, legal stage and security type are Segments groupings;
    # a blank legal stage is "Not in legal" and NACH Y/N read as words.
    from ui.tabs.portfolio_intelligence import _cached_segments
    df = pd.DataFrame({"Loan No": ["A", "B", "C"], "Unit": ["X"] * 3, "Arrears / EMI": [0.0, 4.0, 1.0],
                       "curr_bucket": ["STD", "NPA", "1-30 DPD"], "SOH": [1.0, 2.0, 3.0],
                       "LGL_DESCRIPTION": [None, "Sec 138 Notice", ""], "NACHStatus": ["Y", "N", "y"]})
    legal = _cached_segments.__wrapped__(df, 0, "", "LGL_DESCRIPTION", "Together", "All")
    assert set(legal["Name"]) == {"Not in legal", "Sec 138 Notice"}
    assert int(legal.set_index("Name").loc["Not in legal", "Accounts"]) == 2
    nach = _cached_segments.__wrapped__(df, 0, "", "NACHStatus", "Together", "All")
    assert set(nach["Name"]) == {"Registered", "Not registered"}
