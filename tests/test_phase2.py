"""Phase 2 tests  -  IR-1 planner plumbing.

The catalog generation and IR-1 normalisation are pure and tested directly,
no live Gemini call needed.
"""
from agents.logical_planner import build_catalog, _normalize_ir1


class TestCatalog:
    def test_catalog_lists_registry_names(self):
        cat = build_catalog()
        assert "colending_at_risk" in cat
        assert "exposure" in cat
        assert "branch" in cat
        # The catalog must NOT leak raw condition definitions (the compiler owns those).
        assert "CoLending_Loans" not in cat


class TestNormalizeIR1:
    def test_empty_gets_safe_defaults(self):
        ir = _normalize_ir1({})
        assert ir["intent"] == "loan_table"   # safer default: never auto-aggregate
        assert ir["filters"] == [] and ir["dimensions"] == [] and ir["measures"] == []
        assert ir["limit"] is None

    def test_passthrough_preserves_fields(self):
        ir = _normalize_ir1({"intent": "loan_table", "filters": [{"concept": "npa"}], "limit": 5})
        assert ir["intent"] == "loan_table"
        assert ir["filters"] == [{"concept": "npa"}]
        assert ir["limit"] == 5

    def test_show_all_columns_defaults_to_false(self):
        assert _normalize_ir1({})["show_all_columns"] is False

    def test_show_all_columns_passthrough_true(self):
        ir = _normalize_ir1({"show_all_columns": True, "display_columns": ["Loan No"]})
        assert ir["show_all_columns"] is True
        # display_columns is preserved too -- the compiler decides which wins,
        # not the normalizer.
        assert ir["display_columns"] == ["Loan No"]

    def test_view_highlight_metrics_are_coerced(self):
        ir = _normalize_ir1({"view": {
            "name": "region_scorecard",
            "highlight_metrics": [{"column": "NPA%", "agg": "max"}, {"column": "Collection%", "agg": "min"}],
        }})
        assert ir["view"]["highlight_metrics"] == [
            {"column": "NPA%", "agg": "max"}, {"column": "Collection%", "agg": "min"},
        ]

    def test_view_highlight_metrics_drops_malformed_entries(self):
        ir = _normalize_ir1({"view": {
            "name": "region_scorecard",
            "highlight_metrics": [
                {"column": "NPA%", "agg": "max"},      # valid
                {"column": "NPA%"},                     # missing agg
                {"column": "NPA%", "agg": "average"},   # invalid agg
                "not a dict",                           # wrong type
            ],
        }})
        assert ir["view"]["highlight_metrics"] == [{"column": "NPA%", "agg": "max"}]

    def test_view_highlight_metrics_defaults_to_empty(self):
        ir = _normalize_ir1({"view": {"name": "region_scorecard"}})
        assert ir["view"]["highlight_metrics"] == []
        assert _normalize_ir1({})["view"] is None

    def test_view_sort_by_is_coerced(self):
        ir = _normalize_ir1({"view": {"name": "region_scorecard", "sort_by": {"column": "Collection%", "dir": "asc"}}})
        assert ir["view"]["sort_by"] == {"column": "Collection%", "dir": "asc"}

    def test_view_sort_by_dir_defaults_to_desc(self):
        ir = _normalize_ir1({"view": {"name": "region_scorecard", "sort_by": {"column": "Collection%"}}})
        assert ir["view"]["sort_by"] == {"column": "Collection%", "dir": "desc"}

    def test_view_sort_by_invalid_dir_falls_back_to_desc(self):
        ir = _normalize_ir1({"view": {"name": "region_scorecard", "sort_by": {"column": "Collection%", "dir": "sideways"}}})
        assert ir["view"]["sort_by"]["dir"] == "desc"

    def test_view_sort_by_missing_column_is_none(self):
        ir = _normalize_ir1({"view": {"name": "region_scorecard", "sort_by": {"dir": "asc"}}})
        assert ir["view"]["sort_by"] is None
        ir2 = _normalize_ir1({"view": {"name": "region_scorecard"}})
        assert ir2["view"]["sort_by"] is None
