"""One registry of metric names (code review item 5, 2026-10-27). The same
facts (how a metric is spelled, whether higher is better, its executive
scorecard column) were written out in 9 tables across 7 files, and the
spellings drifting apart caused empty tables, wrong sorts and, found while
doing this, executive questions about "Strike%" being refused."""
import pytest

from registry import metric_names as mn


class TestRegistry:
    @pytest.mark.parametrize("spelling, name", [
        ("NPA%", "NPA%"), ("NPA %", "NPA%"), (" NPA % ", "NPA%"),
        ("Strike%", "Strike%"), ("Strike Rate %", "Strike%"), ("Strike Rate%", "Strike%"), ("Strike %", "Strike%"),
        ("Collection %", "Collection%"), ("Hard Bucket %", "Hard Bucket%"), ("Delinquency %", "Delinquency%"),
        ("SMA-2 %", "SMA-2%"), ("Roll Fwd %", "Roll Fwd%"), ("Concern Score", "Concern Score"),
    ])
    def test_every_spelling_has_one_name(self, spelling, name):
        assert mn.canonical(spelling) == name

    def test_an_unknown_name_is_none_not_a_guess(self):
        assert mn.canonical("Profit %") is None and mn.higher_is_better("Profit %") is None
        assert mn.canonical(None) is None

    def test_direction(self):
        assert mn.higher_is_better("Collection %") is True and mn.higher_is_better("Strike Rate %") is True
        assert mn.higher_is_better("NPA %") is False and mn.higher_is_better("Slipped") is False

    def test_executive_column(self):
        assert mn.executive_column("Strike%") == "Strike Rate %"
        assert mn.executive_column("NPA %") == "NPA %"
        assert mn.executive_column("Accounts") == "Accounts"        # no executive spelling: unchanged

    def test_spellings_lists_the_name_first_then_the_others(self):
        assert mn.spellings("NPA%") == ["NPA%", "NPA %"]
        assert mn.spellings("Strike Rate %")[0] == "Strike%"

    def test_no_spelling_belongs_to_two_metrics(self):
        seen = {}
        for m in mn.METRICS.values():
            for s in (m.name, *m.aliases):
                assert s not in seen, f"{s} is listed under {seen[s]} and {m.name}"
                seen[s] = m.name


class TestEveryTableAgrees:
    """The tables callers use are built from the registry; these keep any
    table that still lists names by hand from drifting."""
    def test_investigator_directions(self):
        from investigator.steps import METRIC_DIRECTION
        for spelling, hib in METRIC_DIRECTION.items():
            assert mn.higher_is_better(spelling) is hib, spelling

    def test_view_directions_and_rollups(self):
        from registry.views import _METRIC_AGG, _METRIC_DIRECTION
        for spelling, d in _METRIC_DIRECTION.items():
            assert mn.higher_is_better(spelling) is (d == "low_bad"), spelling
        for spelling, agg in _METRIC_AGG.items():
            assert mn.METRICS[mn.canonical(spelling)].rollup == agg, spelling

    def test_league_table_directions(self):
        from analysis.summary import LEAGUE_METRICS
        for label, (col, higher_is_worse) in LEAGUE_METRICS.items():
            assert mn.higher_is_better(col) is (not higher_is_worse), label

    def test_drill_filters_cover_both_spellings(self):
        from investigator.steps import METRIC_DIRECTION, METRIC_DRILL_FILTER
        assert set(METRIC_DRILL_FILTER) == set(METRIC_DIRECTION) - {"SMA-2%", "SMA-2 %"}
        for spelling, f in METRIC_DRILL_FILTER.items():
            assert METRIC_DRILL_FILTER[mn.canonical(spelling)] == f, spelling

    def test_excel_colours_agree(self):
        """The Excel export colours columns by name pattern; where a pattern
        matches a registry metric it must point the same way."""
        from ui.components import _XL_HIGHER_IS_BETTER, _XL_HIGHER_IS_WORSE
        for m in mn.METRICS.values():
            for s in (m.name, *m.aliases):
                up = s.upper()
                if any(p in up for p in _XL_HIGHER_IS_WORSE):
                    assert m.higher_is_better is False, s
                elif any(p in up for p in _XL_HIGHER_IS_BETTER):
                    assert m.higher_is_better is True, s


class TestFoundWhileDoingThis:
    @pytest.mark.parametrize("step", ["underperformer_quartile", "bottom_n_per_group", "executive_roster"])
    def test_executive_steps_accept_strike_in_either_spelling(self, step):
        """"Strike%".replace(" %", "%") never becomes "Strike Rate%", so the
        guardrail refused the very spelling the router is told to use."""
        from investigator.guardrails import _metric_producible
        assert _metric_producible(step, "Strike%") and _metric_producible(step, "Strike Rate %")

    def test_region_tables_still_refuse_strike(self):
        """dimension_breakdown's aggregator has no Strike% column: refusing is right."""
        from investigator.guardrails import _metric_producible
        assert not _metric_producible("dimension_breakdown", "Strike%")
        assert not _metric_producible("dimension_breakdown", "Strike Rate %")
