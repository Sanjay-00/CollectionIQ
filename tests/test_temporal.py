"""Temporal model tests  -  Step 3 of IR-2.

The compiler maps a declarative `time.compare` block to concrete prev_* column
aggregations (from PREV_CARRYOVER_COLS), so the LLM never authors raw prev_*
column names. Today's `prev_*` snapshot-comparison hack becomes a MODELED SPECIAL
CASE of the time block, not a parallel code path.

All tests are deterministic (no live Gemini). They cover:
- TIME_MODEL structure (the semantic model for time)
- snapshot compare: compiler adds prev_* aggregation alongside the current one
- change compare: compiler also derives the delta column
- unsupported period refs: loud error (not silent wrong plan)
- measures with no prev mapping: silently skipped (graceful, not fatal)
- backward compatibility: IR-1 without time block is unchanged
- _normalize_ir1 passes the time block through
- full roundtrip: execute a plan with a DataFrame that has prev_SOH
"""
import pytest
import pandas as pd

from registry.semantic_model import TIME_MODEL
from utils import PREV_CARRYOVER_COLS
from compiler.core import compile_logical, _resolve_time_compare
from agents.logical_planner import _normalize_ir1
from agents.plan_executor import execute_plan


# ── Helpers ────────────────────────────────────────────────────────────────────

def _df_with_prev():
    """Minimal DataFrame with SOH and prev_SOH, grouped by two branches."""
    return pd.DataFrame([
        {"Unit": "A", "SOH": 500.0, "prev_SOH": 600.0,
         "POS": 450.0, "prev_POS": 550.0},
        {"Unit": "A", "SOH": 300.0, "prev_SOH": 350.0,
         "POS": 250.0, "prev_POS": 300.0},
        {"Unit": "B", "SOH": 200.0, "prev_SOH": 150.0,
         "POS": 180.0, "prev_POS": 130.0},
    ])


# ── TIME_MODEL structure ───────────────────────────────────────────────────────

class TestTimeModelStructure:
    def test_required_fields_present(self):
        for field in ("axis_column", "period_grain", "row_semantics",
                      "supported_periods", "flow_columns"):
            assert field in TIME_MODEL, f"TIME_MODEL missing '{field}'"

    def test_period_labels(self):
        assert "prev" in TIME_MODEL["supported_periods"]
        assert "curr" in TIME_MODEL["supported_periods"]

    def test_row_semantics_is_snapshot(self):
        # Data is a point-in-time stock; summing across months is wrong.
        assert TIME_MODEL["row_semantics"] == "snapshot"

    def test_flow_columns_are_a_set(self):
        # frozenset or set  -  iterable, not a list (order doesn't matter).
        assert hasattr(TIME_MODEL["flow_columns"], "__contains__")
        assert "Month Receipt Amount" in TIME_MODEL["flow_columns"]


# ── _resolve_time_compare unit tests ─────────────────────────────────────────

class TestResolveTimeCompare:
    def test_snapshot_adds_prev_agg(self):
        ir_measures = [{"metric": "exposure", "alias": "soh"}]
        extra_aggs, extra_derives = _resolve_time_compare(
            {"compare": {"type": "snapshot", "from": "prev", "to": "curr"}},
            ir_measures, [],
        )
        aliases = [a["alias"] for a in extra_aggs]
        assert "prev_soh" in aliases
        assert extra_derives == []   # snapshot: no change derive

    def test_change_adds_prev_agg_and_derive(self):
        ir_measures = [{"metric": "exposure", "alias": "soh"}]
        extra_aggs, extra_derives = _resolve_time_compare(
            {"compare": {"type": "change", "from": "prev", "to": "curr"}},
            ir_measures, [],
        )
        assert any(a["alias"] == "prev_soh" for a in extra_aggs)
        assert any(d["column"] == "soh_change" for d in extra_derives)
        change = next(d for d in extra_derives if d["column"] == "soh_change")
        assert change["expr"] == "soh - prev_soh"

    def test_unsupported_period_ref_errors(self):
        errs: list = []
        _resolve_time_compare(
            {"compare": {"type": "change", "from": "q1", "to": "q2"}},
            [{"metric": "exposure", "alias": "soh"}], errs,
        )
        assert errs, "Should emit an error for unsupported period refs"
        assert any("from:'prev'" in e for e in errs)

    def test_column_without_prev_mapping_skipped(self):
        # "Loan Amount" is not in PREV_CARRYOVER_COLS → no prev agg, no error.
        ir_measures = [{"column": "Loan Amount", "agg": "sum", "alias": "loan_amt"}]
        errs: list = []
        extra_aggs, extra_derives = _resolve_time_compare(
            {"compare": {"type": "snapshot", "from": "prev", "to": "curr"}},
            ir_measures, errs,
        )
        assert extra_aggs == []
        assert errs == []   # not a fatal error  -  just no prev data for that column

    def test_multiple_measures_only_mapped_columns_get_prev(self):
        ir_measures = [
            {"metric": "exposure", "alias": "soh"},         # SOH -> prev_SOH (mapped)
            {"column": "Loan Amount", "agg": "sum", "alias": "lamt"},  # not mapped
        ]
        extra_aggs, _ = _resolve_time_compare(
            {"compare": {"type": "snapshot", "from": "prev", "to": "curr"}},
            ir_measures, [],
        )
        aliases = {a["alias"] for a in extra_aggs}
        assert "prev_soh" in aliases
        assert "prev_lamt" not in aliases  # skipped gracefully


# ── compile_logical integration ───────────────────────────────────────────────

class TestCompileLogicalWithTime:
    def test_snapshot_plan_contains_both_current_and_prev_agg(self):
        df = _df_with_prev()
        ir = {
            "dimensions": ["branch"],
            "measures": [{"metric": "exposure", "alias": "soh"}],
            "time": {"grain": "month", "compare": {"type": "snapshot", "from": "prev", "to": "curr"}},
        }
        plan, errs = compile_logical(ir, df.columns)
        assert errs == [], errs
        ga = next(s for s in plan if s["op"] == "group_aggregate")
        aliases = [a["alias"] for a in ga["aggregations"]]
        assert "soh" in aliases      # current-period column
        assert "prev_soh" in aliases  # prev-period column injected by compiler

    def test_change_plan_has_derive_step(self):
        df = _df_with_prev()
        ir = {
            "dimensions": ["branch"],
            "measures": [{"metric": "exposure", "alias": "soh"}],
            "time": {"grain": "month", "compare": {"type": "change", "from": "prev", "to": "curr"}},
        }
        plan, errs = compile_logical(ir, df.columns)
        assert errs == [], errs
        derives = [s for s in plan if s["op"] == "derive"]
        assert any(d.get("column") == "soh_change" for d in derives)

    def test_ir_without_time_block_unchanged(self):
        """IR-1 without a time block must produce the same plan as before Step 3."""
        df = _df_with_prev()
        ir_no_time = {"dimensions": ["branch"], "measures": [{"metric": "exposure", "alias": "soh"}]}
        plan, errs = compile_logical(ir_no_time, df.columns)
        assert errs == [], errs
        ga = next(s for s in plan if s["op"] == "group_aggregate")
        aliases = [a["alias"] for a in ga["aggregations"]]
        assert "soh" in aliases
        assert "prev_soh" not in aliases  # no time block → no prev injection

    def test_unsupported_period_propagates_to_compile_errors(self):
        df = _df_with_prev()
        ir = {
            "dimensions": ["branch"],
            "measures": [{"metric": "exposure", "alias": "soh"}],
            "time": {"compare": {"type": "change", "from": "fy25", "to": "fy26"}},
        }
        _, errs = compile_logical(ir, df.columns)
        assert any("from:'prev'" in e for e in errs)


# ── Full roundtrip (compile + execute) ────────────────────────────────────────

class TestTemporalRoundtrip:
    def test_snapshot_roundtrip_produces_correct_values(self):
        """Branch A: current SOH=800, prev=950. Branch B: current=200, prev=150."""
        df = _df_with_prev()
        ir = {
            "dimensions": ["branch"],
            "measures": [{"metric": "exposure", "alias": "soh"}],
            "time": {"grain": "month", "compare": {"type": "snapshot", "from": "prev", "to": "curr"}},
        }
        plan, errs = compile_logical(ir, df.columns)
        assert errs == [], errs
        out, err = execute_plan(df, plan)
        assert err == ""
        row_a = out[out["Unit"] == "A"].iloc[0]
        assert row_a["soh"] == 800.0        # 500 + 300
        assert row_a["prev_soh"] == 950.0   # 600 + 350

        row_b = out[out["Unit"] == "B"].iloc[0]
        assert row_b["soh"] == 200.0
        assert row_b["prev_soh"] == 150.0

    def test_change_roundtrip_correct_delta(self):
        """soh_change = soh - prev_soh. A: 800-950=-150. B: 200-150=+50."""
        df = _df_with_prev()
        ir = {
            "dimensions": ["branch"],
            "measures": [{"metric": "exposure", "alias": "soh"}],
            "time": {"grain": "month", "compare": {"type": "change", "from": "prev", "to": "curr"}},
        }
        plan, errs = compile_logical(ir, df.columns)
        assert errs == [], errs
        out, err = execute_plan(df, plan)
        assert err == ""
        row_a = out[out["Unit"] == "A"].iloc[0]
        assert row_a["soh_change"] == pytest.approx(-150.0)  # 800 - 950

        row_b = out[out["Unit"] == "B"].iloc[0]
        assert row_b["soh_change"] == pytest.approx(50.0)   # 200 - 150

    def test_multiple_measures_snapshot(self):
        """Two measures (exposure + pos) both get prev columns."""
        df = _df_with_prev()
        ir = {
            "dimensions": ["branch"],
            "measures": [
                {"metric": "exposure", "alias": "soh"},
                {"metric": "pos", "alias": "pos"},
            ],
            "time": {"grain": "month", "compare": {"type": "snapshot", "from": "prev", "to": "curr"}},
        }
        plan, errs = compile_logical(ir, df.columns)
        assert errs == [], errs
        ga = next(s for s in plan if s["op"] == "group_aggregate")
        aliases = {a["alias"] for a in ga["aggregations"]}
        assert {"soh", "prev_soh", "pos", "prev_pos"}.issubset(aliases)

    def test_lcc_pct_ratio_without_prev_mapping_skipped_gracefully(self):
        """lcc_pct's numerator/denominator aren't in PREV_CARRYOVER_COLS  -  no prev ratio,
        no error. The current-period lcc_pct is still computed correctly."""
        df = pd.DataFrame([
            {"Unit": "A", "Cum Coll (Inst+Exp)": 90.0, "Cum Due-Inst": 50.0, "Cum Due-Exp": 50.0},
        ])
        ir = {
            "dimensions": ["branch"],
            "measures": [{"metric": "lcc_pct", "alias": "lcc"}],
            "time": {"grain": "month", "compare": {"type": "snapshot", "from": "prev", "to": "curr"}},
        }
        plan, errs = compile_logical(ir, df.columns)
        assert errs == [], errs   # no fatal error
        derives = [s for s in plan if s["op"] == "derive"]
        assert any(d.get("column") == "lcc" for d in derives)   # current lcc present
        assert not any("prev_lcc" in str(d) for d in derives)    # prev skipped gracefully


# ── _normalize_ir1 ─────────────────────────────────────────────────────────────

class TestNormalizeIr1:
    def test_time_block_passed_through(self):
        time_block = {"grain": "month", "compare": {"type": "change", "from": "prev", "to": "curr"}}
        result = _normalize_ir1({"time": time_block})
        assert result["time"] == time_block

    def test_missing_time_block_is_none(self):
        result = _normalize_ir1({})
        assert result["time"] is None

    def test_existing_ir1_fields_unchanged(self):
        raw = {"intent": "aggregation", "dimensions": ["branch"], "measures": []}
        result = _normalize_ir1(raw)
        assert result["intent"] == "aggregation"
        assert result["dimensions"] == ["branch"]
        assert result["time"] is None   # back-compat: absent → None


# ── Proof: the hack is subsumed ───────────────────────────────────────────────

class TestPrevHackIsSubsumed:
    def test_llm_never_needs_to_author_prev_column_name(self):
        """The key Step 3 principle: the compiler maps 'prev'->prev_SOH.
        The IR contains NO raw prev_* column name  -  only the 'prev' period label."""
        ir = {
            "dimensions": ["branch"],
            "measures": [{"metric": "exposure", "alias": "soh"}],
            "time": {"compare": {"type": "snapshot", "from": "prev", "to": "curr"}},
        }
        # No raw "prev_SOH" string anywhere in the IR.
        ir_str = str(ir)
        assert "prev_SOH" not in ir_str
        assert "prev_soh" not in ir_str

        # But the COMPILED PLAN does reference it (compiler injected it).
        df = _df_with_prev()
        plan, errs = compile_logical(ir, df.columns)
        assert errs == []
        plan_str = str(plan)
        assert "prev_SOH" in plan_str  # compiler's work, not the LLM's
