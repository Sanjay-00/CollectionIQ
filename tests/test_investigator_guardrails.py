import pandas as pd

from helpers import make_df
from investigator.state import EntityMemory
from investigator.guardrails import validate_and_correct


class TestNoMetricRequested:
    def test_turn_with_no_metric_param_passes_through_unchanged(self):
        turn = {"step_type": "dimension_breakdown", "params": {"dimension": "region"}}
        result = validate_and_correct(turn, EntityMemory())
        assert result == turn


class TestMetricIsProducibleByTheChosenStep:
    def test_dimension_breakdown_with_a_producible_metric_passes_through(self):
        turn = {"step_type": "dimension_breakdown", "params": {"dimension": "branch", "metric": "NPA%"}}
        result = validate_and_correct(turn, EntityMemory())
        assert result == turn

    def test_underperformer_quartile_with_a_producible_metric_passes_through(self):
        turn = {"step_type": "underperformer_quartile", "params": {"metric": "Strike Rate %"}}
        result = validate_and_correct(turn, EntityMemory())
        assert result == turn


class TestMetricNotProducibleAutoCorrectsFromStoredParams:
    """The exact regression this guardrail exists for: "why strike rate is
    less" right after an executive-grain table routed to
    dimension_breakdown(region), which can never produce a "Strike Rate %"
    column at all. Detected AFTER parse_turn, deterministically, using the
    known producible-column set per step type -- never relying on the
    prompt alone to prevent this a second time."""

    def _executive_grain_memory(self):
        mem = EntityMemory()
        df = pd.DataFrame({
            "MNT NAME": ["RAHUL", "AJAY"], "Unit": ["KMGON", "KMGON"],
            "Collection %": [71.8, 79.0], "Strike Rate %": [62.6, 67.0],
        })
        mem.touch(
            "executive_group", "underperformer_quartile:Collection %:KMGON",
            result_df=df, metric_focus="Collection %", step_type="underperformer_quartile",
            params={"metric": "Collection %", "scope_col": "branch", "scope_value": "KMGON"},
        )
        return mem

    def test_corrects_to_the_executive_grain_entity_s_own_step_and_scope(self):
        turn = {
            "step_type": "dimension_breakdown",
            "params": {"dimension": "region", "metric": "Strike Rate %", "higher_is_better": False},
        }
        result = validate_and_correct(turn, self._executive_grain_memory())
        assert result["step_type"] == "underperformer_quartile"
        assert result["params"]["metric"] == "Strike Rate %"
        assert result["params"]["scope_col"] == "branch"
        assert result["params"]["scope_value"] == "KMGON"
        assert not result["needs_clarification"]

    def test_no_suitable_fallback_degrades_to_clarification_not_a_silent_wrong_answer(self):
        turn = {
            "step_type": "dimension_breakdown",
            "params": {"dimension": "region", "metric": "Strike Rate %"},
        }
        result = validate_and_correct(turn, EntityMemory())
        assert result["step_type"] is None
        assert result["needs_clarification"] is True
        assert "Strike Rate %" in result["clarification_question"]

    def test_fallback_entity_without_the_requested_metric_also_degrades_to_clarification(self):
        mem = EntityMemory()
        df = pd.DataFrame({"MNT NAME": ["RAHUL"], "Unit": ["KMGON"], "Collection %": [71.8]})
        mem.touch(
            "executive_group", "underperformer_quartile:Collection %:KMGON",
            result_df=df, step_type="underperformer_quartile",
            params={"metric": "Collection %", "scope_col": "branch", "scope_value": "KMGON"},
        )
        turn = {
            "step_type": "dimension_breakdown",
            "params": {"dimension": "region", "metric": "Some Unlisted Metric"},
        }
        result = validate_and_correct(turn, mem)
        assert result["step_type"] is None
        assert result["needs_clarification"] is True


class TestBottomNPerGroupWithExecutiveGroupColRedirects:
    """A plain "bottom N executives [by METRIC]" question with no group
    named (no "per branch"/"per zone") has been caught live routing to
    bottom_n_per_group with group_col="executive" -- invalid, since
    resolve_dimension has no "executive" alias and executive is what's
    being ranked, never what's grouped by. This is a flat ranked list, the
    exact shape executive_roster(worst_first=True) exists for -- redirect
    there instead of executing a step known to be broken for this input."""

    def test_redirects_to_executive_roster_worst_first(self):
        turn = {
            "step_type": "bottom_n_per_group",
            "params": {"metric": "Collection %", "n": 2, "group_col": "executive", "higher_is_better": None},
        }
        result = validate_and_correct(turn, EntityMemory())
        assert result["step_type"] == "executive_roster"
        assert result["params"]["metric"] == "Collection %"
        assert result["params"]["n"] == 2
        assert result["params"]["worst_first"] is True
        assert not result["needs_clarification"]

    def test_valid_group_col_passes_through_unchanged(self):
        turn = {
            "step_type": "bottom_n_per_group",
            "params": {"metric": "Collection %", "n": 2, "group_col": "branch", "higher_is_better": None},
        }
        result = validate_and_correct(turn, EntityMemory())
        assert result == turn


class TestAmbiguousExecutiveNameAsksForClarification:
    """The exact bug found while auditing this pipeline: entity_summary's
    name-only (contains) executive match, when it resolves to more than one
    DISTINCT person, used to silently return compute_executive_scorecard's
    iloc[0] -- best Collection% first, per that function's own sort -- with
    zero signal that a second person even matched. Concretely: two
    executives both named "Rahul," one genuinely struggling (low
    Collection%) and one doing fine (high Collection%) -- asking "why is
    Rahul struggling" used to silently return the WELL-PERFORMING Rahul's
    numbers every time, since the struggling one always loses the iloc[0]
    tiebreak. This is a real reputational-risk bug for a tool whose answers
    can affect how a real person's performance is judged -- must ask which
    Rahul was meant instead of guessing."""

    def _two_rahuls_df(self):
        rows = (
            [{"MNT NAME": "RAHUL SHARMA", "Unit": "PUNE",   "Strike": "Y"}] * 5
            + [{"MNT NAME": "RAHUL VERMA",  "Unit": "NAGPUR", "Strike": "N"}] * 5
        )
        return make_df(rows)

    def test_ambiguous_name_asks_instead_of_silently_picking_the_better_performer(self):
        turn = {
            "step_type": "entity_summary",
            "params": {"entity_type": "executive", "entity_value": ["Rahul", None]},
        }
        result = validate_and_correct(turn, EntityMemory(), self._two_rahuls_df())
        assert result["needs_clarification"] is True
        assert result["step_type"] is None
        # Both distinct people named in the clarification, not just one.
        assert "RAHUL SHARMA" in result["clarification_question"]
        assert "RAHUL VERMA" in result["clarification_question"]
        assert "PUNE" in result["clarification_question"]
        assert "NAGPUR" in result["clarification_question"]

    def test_unambiguous_name_passes_through_unchanged(self):
        turn = {
            "step_type": "entity_summary",
            "params": {"entity_type": "executive", "entity_value": ["Sharma", None]},
        }
        result = validate_and_correct(turn, EntityMemory(), self._two_rahuls_df())
        assert result == turn

    def test_unit_that_actually_disambiguates_passes_through_unchanged(self):
        turn = {
            "step_type": "entity_summary",
            "params": {"entity_type": "executive", "entity_value": ["Rahul", "PUNE"]},
        }
        result = validate_and_correct(turn, EntityMemory(), self._two_rahuls_df())
        assert result == turn

    def test_no_df_curr_available_skips_the_check(self):
        # A turn validated with no data available (e.g. before df_curr is
        # ready) must not crash -- this check is a best-effort safety net,
        # not a hard requirement, and _executive_row's own iloc[0] fallback
        # still applies if this genuinely can't be checked here.
        turn = {
            "step_type": "entity_summary",
            "params": {"entity_type": "executive", "entity_value": ["Rahul", None]},
        }
        result = validate_and_correct(turn, EntityMemory())
        assert result == turn
