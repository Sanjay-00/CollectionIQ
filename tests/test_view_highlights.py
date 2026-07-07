"""Fix B: query-aware highlight KPI cards and ranking on the fast-path view layer.

_build_highlights/_apply_sort_by (graph.py) are the pure-pandas half of the
design -- the Logical Planner only ever names WHICH column + direction, never
a value; these functions do the actual idxmax/idxmin/sort_values work. These
tests exercise them directly with synthetic DataFrames, no LLM/Gemini call.
"""
import pandas as pd

from graph import _build_highlights, _apply_sort_by, _build_portfolio_kpis, _normalize_col
from registry.views import VIEWS


def _region_df():
    return pd.DataFrame([
        {"Region": "CS NAGAR", "NPA%": 16.93, "Collection%": 86.83},
        {"Region": "AKOLA",    "NPA%": 14.77, "Collection%": 91.21},
        {"Region": "LATUR",    "NPA%": 11.31, "Collection%": 92.49},
    ])


class TestBuildHighlights:
    def test_max_and_min_on_region_scorecard(self):
        spec = VIEWS["region_scorecard"]
        requests = [
            {"column": "NPA%", "agg": "max"},
            {"column": "Collection%", "agg": "min"},
        ]
        out = _build_highlights(spec, _region_df(), requests)
        assert len(out) == 2

        npa = next(h for h in out if h["label"] == "Highest NPA%")
        assert npa["entity"] == "CS NAGAR"
        assert npa["value"] == "16.93%"
        assert npa["bad"] is True  # high NPA% is bad

        coll = next(h for h in out if h["label"] == "Lowest Collection%")
        assert coll["entity"] == "CS NAGAR"
        assert coll["value"] == "86.83%"
        assert coll["bad"] is True  # low Collection% is bad

    def test_min_on_high_bad_metric_is_a_good_highlight(self):
        # Asking for the LOWEST NPA% (the best-performing region) should not be
        # flagged red -- it's the opposite of a concerning signal.
        spec = VIEWS["region_scorecard"]
        out = _build_highlights(spec, _region_df(), [{"column": "NPA%", "agg": "min"}])
        assert out[0]["entity"] == "LATUR"
        assert out[0]["bad"] is False

    def test_unknown_column_is_silently_dropped(self):
        spec = VIEWS["region_scorecard"]
        out = _build_highlights(spec, _region_df(), [
            {"column": "NotARealColumn", "agg": "max"},
            {"column": "NPA%", "agg": "max"},
        ])
        assert len(out) == 1
        assert out[0]["label"] == "Highest NPA%"

    def test_column_not_in_views_metrics_allowlist_is_dropped(self):
        # "Region" is a real column in the DataFrame but not a declared
        # highlightable metric for this view -- must not be usable as one.
        spec = VIEWS["region_scorecard"]
        out = _build_highlights(spec, _region_df(), [{"column": "Region", "agg": "max"}])
        assert out == []

    def test_no_requests_returns_empty_list(self):
        spec = VIEWS["region_scorecard"]
        assert _build_highlights(spec, _region_df(), []) == []

    def test_view_without_label_col_returns_empty_list(self):
        spec = VIEWS["top_delinquent_accounts"]  # no "metrics"/"label_col" declared
        out = _build_highlights(spec, _region_df(), [{"column": "NPA%", "agg": "max"}])
        assert out == []

    def test_empty_result_df_returns_empty_list(self):
        spec = VIEWS["region_scorecard"]
        out = _build_highlights(spec, pd.DataFrame(), [{"column": "NPA%", "agg": "max"}])
        assert out == []


class TestApplySortBy:
    def test_sorts_descending_by_default_direction(self):
        out = _apply_sort_by(_region_df(), {"column": "NPA%", "dir": "desc"})
        assert out["Region"].tolist() == ["CS NAGAR", "AKOLA", "LATUR"]

    def test_sorts_ascending(self):
        out = _apply_sort_by(_region_df(), {"column": "Collection%", "dir": "asc"})
        assert out["Region"].tolist() == ["CS NAGAR", "AKOLA", "LATUR"]

    def test_no_sort_by_is_a_no_op(self):
        df = _region_df()
        out = _apply_sort_by(df, None)
        assert out["Region"].tolist() == df["Region"].tolist()

    def test_unknown_column_is_a_no_op(self):
        df = _region_df()
        out = _apply_sort_by(df, {"column": "NotARealColumn", "dir": "desc"})
        assert out["Region"].tolist() == df["Region"].tolist()

    def test_fuzzy_matches_cross_view_column_spelling(self):
        # Real observed failure mode: the planner named "Strike Rate %"
        # (executive_scorecard's spelling) while region_scorecard's own
        # column is "Strike%" -- must still resolve and actually sort.
        df = pd.DataFrame([
            {"Region": "CS NAGAR", "Strike%": 40.0},
            {"Region": "AKOLA",    "Strike%": 70.0},
            {"Region": "LATUR",    "Strike%": 90.0},
        ])
        out = _apply_sort_by(df, {"column": "Strike Rate %", "dir": "desc"})
        assert out["Region"].tolist() == ["LATUR", "AKOLA", "CS NAGAR"]

    def test_fuzzy_match_ignores_case_and_spacing(self):
        out = _apply_sort_by(_region_df(), {"column": "collection %", "dir": "asc"})
        assert out["Region"].tolist() == ["CS NAGAR", "AKOLA", "LATUR"]

    def test_fuzzy_match_resolves_exact_normalized_equivalents(self):
        # "NPA" and "NPA%" normalize to the identical string ("npa") -- an
        # exact normalized match, not a risky partial/substring guess, so
        # this must resolve regardless of the substring-match length gate.
        df = pd.DataFrame([{"Region": "A", "NPA%": 1.0}, {"Region": "B", "NPA%": 2.0}])
        out = _apply_sort_by(df, {"column": "NPA", "dir": "desc"})
        assert out["Region"].tolist() == ["B", "A"]

    def test_fuzzy_match_does_not_false_positive_on_short_fragments(self):
        # A short (<4 char) fragment that is only a SUBSTRING of a column name
        # (not an exact normalized match) is too ambiguous to safely resolve --
        # must fall back to a no-op rather than guessing.
        df = pd.DataFrame([{"Region": "A", "NPA%": 1.0}, {"Region": "B", "NPA%": 2.0}])
        out = _apply_sort_by(df, {"column": "PA", "dir": "desc"})
        assert out["Region"].tolist() == df["Region"].tolist()


class TestBuildPortfolioKpis:
    def test_percent_columns_averaged(self):
        spec = VIEWS["region_scorecard"]
        out = _build_portfolio_kpis(spec, _region_df())
        by_label = {k["label"]: k["value"] for k in out}
        assert by_label["Avg NPA%"] == f"{(16.93 + 14.77 + 11.31) / 3:.2f}%"
        assert by_label["Avg Collection%"] == f"{(86.83 + 91.21 + 92.49) / 3:.2f}%"

    def test_count_like_columns_are_summed_not_averaged(self):
        spec = VIEWS["executive_recovery"]
        df = pd.DataFrame([
            {"Executive": "A", "Rescued": 5, "Slipped": 2, "Net Recovery": 3},
            {"Executive": "B", "Rescued": 3, "Slipped": 1, "Net Recovery": 2},
        ])
        out = _build_portfolio_kpis(spec, df)
        by_label = {k["label"]: k["value"] for k in out}
        assert by_label["Total Rescued"] == "8"
        assert by_label["Total Slipped"] == "3"
        assert by_label["Total Net Recovery"] == "5"

    def test_view_without_metrics_returns_empty_list(self):
        spec = VIEWS["top_delinquent_accounts"]
        assert _build_portfolio_kpis(spec, _region_df()) == []

    def test_empty_result_df_returns_empty_list(self):
        spec = VIEWS["region_scorecard"]
        assert _build_portfolio_kpis(spec, pd.DataFrame()) == []

    def test_missing_column_is_skipped_not_an_error(self):
        # region_scorecard declares more metrics (Strike%, Roll Fwd%, ...) than
        # this minimal df has -- must skip absent ones, not raise.
        spec = VIEWS["region_scorecard"]
        out = _build_portfolio_kpis(spec, _region_df())
        labels = {k["label"] for k in out}
        assert "Avg NPA%" in labels
        assert "Avg Strike%" not in labels  # not present in _region_df()


class TestNormalizeColDeltaCollision:
    def test_delta_prefix_does_not_collide_with_plain_column(self):
        # Regression: _normalize_col used to strip "Δ" (delta) as punctuation,
        # so "NPA%", "NPA", and "Δ NPA%" all normalized to the identical "npa" --
        # a near-miss sort_by request like "NPA %" could then silently resolve to
        # the WRONG column ("Δ NPA%") via dict-key collision in _resolve_column_fuzzy.
        assert _normalize_col("NPA") == "npa"
        assert _normalize_col("NPA%") == "npa"
        assert _normalize_col("Δ NPA%") != "npa"
        assert _normalize_col("Δ NPA%") == _normalize_col("ΔNPA%")  # still normalizes spacing/case

    def test_apply_sort_by_resolves_near_miss_to_plain_column_not_delta(self):
        df = pd.DataFrame({
            "Region": ["C", "A", "B"],
            "NPA%": [1.0, 5.0, 9.0],
            "Δ NPA%": [7.0, 2.0, -3.0],
        })
        out = _apply_sort_by(df, {"column": "NPA %", "dir": "desc"})
        assert out["Region"].tolist() == ["B", "A", "C"]  # sorted by NPA%, not the delta


class TestApplySortByRankRecompute:
    def test_stale_rank_column_is_recomputed_to_match_new_order(self):
        # branch_quadrant ships its own "Rank" (1..N by Concern Score). Re-sorting
        # by a DIFFERENT column must not leave the OLD rank numbers attached to
        # rows now in a different position.
        df = pd.DataFrame({
            "Rank": [1, 2, 3],
            "Branch": ["X", "Y", "Z"],
            "Collection%": [80.0, 95.0, 90.0],
        })
        out = _apply_sort_by(df, {"column": "Collection%", "dir": "desc"})
        assert out["Branch"].tolist() == ["Y", "Z", "X"]
        assert out["Rank"].tolist() == [1, 2, 3]  # recomputed to match new order

    def test_no_rank_column_is_unaffected(self):
        df = pd.DataFrame({"Region": ["A", "B"], "NPA%": [1.0, 2.0]})
        out = _apply_sort_by(df, {"column": "NPA%", "dir": "desc"})
        assert "Rank" not in out.columns
