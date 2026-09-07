import pandas as pd

from investigator.state import EntityMemory


class TestEntityMemoryBasics:
    def test_touch_then_get_round_trips(self):
        mem = EntityMemory()
        mem.touch("branch", "Mahad", result_df=pd.DataFrame({"a": [1]}), metric_focus="NPA%")
        record = mem.get("branch", "Mahad")
        assert record is not None
        assert record.entity_type == "branch"
        assert record.entity_value == "Mahad"
        assert record.metric_focus == "NPA%"

    def test_unknown_entity_returns_none(self):
        mem = EntityMemory()
        assert mem.get("branch", "Nowhere") is None

    def test_second_unrelated_entity_does_not_disturb_the_first(self):
        # The scenario that motivated entity memory over a linear queue:
        # asking about a second, unrelated executive mid-thread must not
        # corrupt or replace the first entity's state.
        mem = EntityMemory()
        mem.touch("branch", "Mahad", result_df=pd.DataFrame({"a": [1]}))
        mem.touch("executive", ("Rahul", "Pune Unit"), result_df=pd.DataFrame({"b": [2]}))
        assert mem.get("branch", "Mahad") is not None
        assert mem.get("executive", ("Rahul", "Pune Unit")) is not None
        assert mem.get("branch", "Mahad").result_df["a"].iloc[0] == 1


class TestStepTypeRoundTrips:
    def test_step_type_defaults_to_empty_string(self):
        mem = EntityMemory()
        record = mem.touch("branch", "Mahad", result_df=pd.DataFrame())
        assert record.step_type == ""

    def test_step_type_is_stored_and_retrievable(self):
        mem = EntityMemory()
        mem.touch("branch", "Mahad", result_df=pd.DataFrame(), step_type="entity_summary")
        assert mem.get("branch", "Mahad").step_type == "entity_summary"


class TestParamsRoundTrip:
    """params stores the EXACT step params that produced result_df -- lets a
    later turn say "same as this entity, but metric=X" without the LLM
    having to re-derive scope_col/scope_value/etc from a text description
    (the params-storage refactor: a follow-up regenerating structured
    parameters from prose is exactly what produced the strike-rate/region
    misroute bug; splicing stored params deterministically removes that
    re-derivation step entirely for a "refine" turn)."""

    def test_params_defaults_to_empty_dict(self):
        mem = EntityMemory()
        record = mem.touch("branch", "Mahad", result_df=pd.DataFrame())
        assert record.params == {}

    def test_params_is_stored_and_retrievable(self):
        mem = EntityMemory()
        mem.touch(
            "executive_group", "underperformer_quartile:Collection %:KMGON",
            result_df=pd.DataFrame(),
            params={"metric": "Collection %", "scope_col": "branch", "scope_value": "KMGON"},
        )
        record = mem.get("executive_group", "underperformer_quartile:Collection %:KMGON")
        assert record.params == {"metric": "Collection %", "scope_col": "branch", "scope_value": "KMGON"}


class TestMostRecent:
    def test_most_recently_touched_entity_wins(self):
        mem = EntityMemory()
        mem.touch("branch", "Mahad", result_df=pd.DataFrame())
        mem.touch("executive", ("Rahul", "Pune Unit"), result_df=pd.DataFrame())
        recent = mem.most_recent()
        assert recent.entity_type == "executive"
        assert recent.entity_value == ("Rahul", "Pune Unit")

    def test_re_touching_an_older_entity_makes_it_most_recent_again(self):
        mem = EntityMemory()
        mem.touch("branch", "Mahad", result_df=pd.DataFrame())
        mem.touch("executive", ("Rahul", "Pune Unit"), result_df=pd.DataFrame())
        mem.touch("branch", "Mahad", result_df=pd.DataFrame({"x": [1]}), metric_focus="Hard Bucket%")
        recent = mem.most_recent()
        assert recent.entity_type == "branch"
        assert recent.entity_value == "Mahad"
        assert recent.metric_focus == "Hard Bucket%"

    def test_empty_memory_most_recent_is_none(self):
        assert EntityMemory().most_recent() is None


class TestAllAndClear:
    def test_all_returns_most_recent_first(self):
        mem = EntityMemory()
        mem.touch("branch", "Mahad", result_df=pd.DataFrame())
        mem.touch("executive", ("Rahul", "Pune Unit"), result_df=pd.DataFrame())
        entities = mem.all()
        assert [e.entity_type for e in entities] == ["executive", "branch"]

    def test_clear_empties_the_memory(self):
        mem = EntityMemory()
        mem.touch("branch", "Mahad", result_df=pd.DataFrame())
        mem.clear()
        assert mem.most_recent() is None
        assert mem.all() == []
