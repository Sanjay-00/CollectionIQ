"""Phase 2 tests  -  IR-1 planner plumbing + shadow harness.

No live Gemini: the planner's LLM call is replaced by an injected stub that
returns a hand-written IR-1, so the new path (IR-1 -> compiler -> engine) and the
comparator are exercised deterministically. The catalog generation and IR-1
normalisation are pure and tested directly.
"""
import pandas as pd

from agents.logical_planner import build_catalog, _normalize_ir1
from compiler.shadow import run_new_path, compare_results, shadow_evaluate


def _df():
    # Same shape as the compiler test: Branch A true at-risk exposure = 100, B = 300.
    return pd.DataFrame({
        "Loan No":         [1, 2, 3, 4, 5],
        "Unit":            ["A", "A", "A", "B", "B"],
        "MNT NAME":        ["x", "x", "y", "z", "z"],
        "CoLending_Loans": ["Y", "Y", "N", "Y", "N"],
        "Arrears / EMI":   [2, 0, 5, 3, 0],
        "SOH":             [100.0, 50.0, 200.0, 300.0, 10.0],
        "RegionName":      ["PUNE", "PUNE", "PUNE", "MUM", "MUM"],
    })


def _colending_per_branch_ir1(query, snapshot_dates=None):
    """Stub planner: the IR-1 a correct model would emit for
    'co-lending at-risk exposure per branch'."""
    return {
        "intent": "aggregation",
        "filters": [{"concept": "colending_at_risk"}],
        "dimensions": ["branch"],
        "measures": [{"metric": "exposure"}],
        "order_by": [{"by": "exposure", "dir": "desc"}],
        "limit": None,
    }


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


class TestRunNewPath:
    def test_injected_planner_produces_correct_result(self):
        out, ir1, err = run_new_path(_df(), "co-lending at-risk per branch",
                                     planner=_colending_per_branch_ir1)
        assert err == ""
        assert ir1["filters"] == [{"concept": "colending_at_risk"}]
        assert out[out["Unit"] == "A"].iloc[0]["exposure"] == 100.0
        assert out[out["Unit"] == "B"].iloc[0]["exposure"] == 300.0

    def test_planner_exception_is_captured(self):
        def boom(q, s=None):
            raise RuntimeError("api down")
        out, ir1, err = run_new_path(_df(), "x", planner=boom)
        assert out is None and "planner failed" in err

    def test_unknown_concept_surfaces_compile_error(self):
        out, ir1, err = run_new_path(
            _df(), "x", planner=lambda q, s=None: {"filters": [{"concept": "ghost"}]}
        )
        assert out is None and "unknown concept 'ghost'" in err


class TestCompareResults:
    def test_match_ignores_rank_and_schema_names(self):
        legacy = pd.DataFrame({"Unit": ["A", "B"], "exposure": [100.0, 300.0]})
        # New path frame has a Rank column and a differently-named metric col.
        new = pd.DataFrame({"Rank": [1, 2], "Unit": ["B", "A"], "soh_sum": [300.0, 100.0]})
        v = compare_results(legacy, new)
        assert v["status"] == "match"
        assert v["row_count_match"] and v["values_match"]

    def test_value_divergence_is_mismatch(self):
        legacy = pd.DataFrame({"Unit": ["A", "B"], "exposure": [100.0, 300.0]})
        new = pd.DataFrame({"Rank": [1, 2], "Unit": ["A", "B"], "exposure": [150.0, 300.0]})
        v = compare_results(legacy, new)
        assert v["status"] == "mismatch"
        assert v["values_match"] is False

    def test_new_none_is_new_error(self):
        legacy = pd.DataFrame({"Unit": ["A"], "exposure": [100.0]})
        assert compare_results(legacy, None)["status"] == "new_error"


class TestShadowEvaluate:
    def test_end_to_end_match_with_stub_planner(self):
        df = _df()
        # Legacy result computed independently (manual co-lending-at-risk per branch).
        legacy = pd.DataFrame({"Unit": ["A", "B"], "exposure": [100.0, 300.0]})
        verdict = shadow_evaluate(df, "co-lending at-risk per branch", legacy,
                                  planner=_colending_per_branch_ir1)
        assert verdict["status"] == "match"
        assert verdict["ir1"]["dimensions"] == ["branch"]

    def test_never_raises_on_bad_planner(self):
        verdict = shadow_evaluate(_df(), "x", pd.DataFrame(),
                                  planner=lambda q, s=None: (_ for _ in ()).throw(ValueError("x")))
        assert verdict["status"] == "new_error"
