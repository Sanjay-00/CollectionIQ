from investigator.suggestions import suggest_mechanism_steps


class TestSuggestMechanismSteps:
    """Deterministic "check mechanism before location" recommendations --
    a plain Python lookup table, never an LLM improvising which step to
    suggest (see investigator/suggestions.py's own module docstring for
    why). Given the worst-moved metric on an entity_summary result, returns
    which of roll_rate_summary/vintage_summary/demand_summary to suggest
    checking BEFORE drilling into executives, and why -- or an empty list
    when the metric has no registered mechanism angle."""

    def test_npa_pct_suggests_roll_rate_and_vintage(self):
        result = suggest_mechanism_steps("NPA%")
        step_types = [s["step_type"] for s in result]
        assert step_types == ["roll_rate_summary", "vintage_summary"]

    def test_hard_bucket_pct_suggests_roll_rate_and_vintage(self):
        result = suggest_mechanism_steps("Hard Bucket%")
        step_types = [s["step_type"] for s in result]
        assert step_types == ["roll_rate_summary", "vintage_summary"]

    def test_collection_pct_suggests_demand_breakdown(self):
        result = suggest_mechanism_steps("Collection%")
        step_types = [s["step_type"] for s in result]
        assert step_types == ["demand_summary"]

    def test_every_suggestion_carries_a_label_and_a_reason(self):
        for metric in ("NPA%", "Hard Bucket%", "Collection%"):
            for suggestion in suggest_mechanism_steps(metric):
                assert suggestion["label"]
                assert suggestion["reason"]

    def test_metric_with_no_mechanism_angle_returns_empty(self):
        assert suggest_mechanism_steps("Strike%") == []
        assert suggest_mechanism_steps("SOH (Cr)") == []
        assert suggest_mechanism_steps("Accounts") == []

    def test_none_or_empty_metric_returns_empty(self):
        assert suggest_mechanism_steps(None) == []
        assert suggest_mechanism_steps("") == []
