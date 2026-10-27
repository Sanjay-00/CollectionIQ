"""Multi-part questions in a region's workspace (2026-10-26): "is it a
segment or an executive?" asked about NASHIK ran one step (a NASHIK summary)
and answered neither part. In a workspace the router may now return up to
three steps, each run through the same checks as a single question."""
import json

import pytest

from investigator.llm import MAX_PARTS, _normalize_turn_response, build_turn_prompt
from investigator.state import EntityMemory

SEGMENT = {"step_type": "product_analysis", "params": {"axis": "segment", "scope_col": "region", "scope_value": "NASHIK"},
           "label": "By segment"}
EXECUTIVE = {"step_type": "underperformer_quartile",
             "params": {"metric": "Collection %", "scope_col": "region", "scope_value": "NASHIK"},
             "label": "By executive"}


def _raw(**fields):
    base = {"step_type": None, "params": {}, "resolved_entity": None, "needs_clarification": False,
            "clarification_question": ""}
    return json.dumps({**base, **fields})


class TestNormalize:
    def test_two_parts_come_back_as_parts(self):
        out = _normalize_turn_response(_raw(parts=[SEGMENT, EXECUTIVE]), allow_multi=True)
        assert [p["step_type"] for p in out["parts"]] == ["product_analysis", "underperformer_quartile"]
        assert [p["label"] for p in out["parts"]] == ["By segment", "By executive"]
        assert out["step_type"] == "product_analysis" and not out["needs_clarification"]

    def test_outside_a_workspace_parts_are_ignored(self):
        out = _normalize_turn_response(_raw(step_type="product_analysis", params=SEGMENT["params"],
                                            parts=[SEGMENT, EXECUTIVE]), allow_multi=False)
        assert "parts" not in out and out["step_type"] == "product_analysis"

    def test_unknown_and_handoff_steps_are_dropped_from_parts(self):
        bad = {"step_type": "delete_everything", "params": {}}
        handoff = {"step_type": "portfolio_query", "params": {"question": "list loans"}}
        out = _normalize_turn_response(_raw(parts=[SEGMENT, bad, handoff, EXECUTIVE]), allow_multi=True)
        assert [p["step_type"] for p in out["parts"]] == ["product_analysis", "underperformer_quartile"]

    def test_one_valid_part_is_a_plain_single_step(self):
        out = _normalize_turn_response(_raw(parts=[SEGMENT, {"step_type": "nope"}]), allow_multi=True)
        assert "parts" not in out and out["step_type"] == "product_analysis" and out["params"] == SEGMENT["params"]

    def test_at_most_three_parts(self):
        out = _normalize_turn_response(_raw(parts=[SEGMENT, EXECUTIVE, SEGMENT, EXECUTIVE, SEGMENT]),
                                       allow_multi=True)
        assert len(out["parts"]) == MAX_PARTS == 3

    def test_a_single_step_answer_is_unchanged(self):
        out = _normalize_turn_response(_raw(step_type="entity_summary",
                                            params={"entity_type": "region", "entity_value": "NASHIK"}),
                                       allow_multi=True)
        assert "parts" not in out and out["step_type"] == "entity_summary"


class TestPrompt:
    def test_the_workspace_prompt_explains_parts(self):
        prompt = build_turn_prompt("About the NASHIK region: is it a segment or an executive?", EntityMemory(),
                                   allow_multi=True)
        assert "MULTI-PART QUESTIONS" in prompt and '"parts"' in prompt

    def test_the_general_prompt_does_not(self):
        assert "MULTI-PART QUESTIONS" not in build_turn_prompt("why is NPA high", EntityMemory())
