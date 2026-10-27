"""The planner's instructions moved from a 380-line f-string into
agents/prompts/planner_system.txt (code review item 6, 2026-10-27), filled
with str.format. Checked byte for byte against the old function at the move;
these keep the template safe to edit."""
import string

import pytest

import agents.logical_planner as lp

_FILLED = {"snapshot_block", "views_catalog", "catalog", "clarification_rule", "out_of_scope_precedence_note"}


def test_the_template_uses_only_the_placeholders_the_code_fills():
    """A single brace typed while editing (e.g. a JSON example) would be read
    as a placeholder and fail every question; a literal brace must be doubled."""
    names = {name for _, name, _, _ in string.Formatter().parse(lp._planner_template()) if name is not None}
    assert names == _FILLED


@pytest.mark.parametrize("dates", [None, {"curr": "2026-09-30"}, {"curr": "2026-09-30", "prev": "2026-08-31"}])
@pytest.mark.parametrize("allow", [True, False])
def test_every_variant_fills_completely(dates, allow):
    prompt = lp._build_full_system_prompt(dates, allow)
    assert prompt.startswith("You are the semantic layer of CollectionIQ")
    assert "{snapshot_block}" not in prompt and "{catalog}" not in prompt
    assert ('previous-period data IS loaded' in prompt) == bool(dates and dates.get("prev"))
    assert ("KNOWN DOMAIN-AMBIGUOUS TERMS" in prompt) == allow


def test_json_examples_keep_single_braces():
    assert '"view": {"name": "<view name>"' in lp._build_full_system_prompt(None, True)


def test_no_dashes_the_rules_forbid():
    """The prompt tells the model never to use an em or en dash; it must not use one itself."""
    assert "—" not in lp._planner_template() and "–" not in lp._planner_template()


def test_the_version_is_recorded_on_traces():
    assert isinstance(lp.PLANNER_PROMPT_VERSION, int) and lp.PLANNER_PROMPT_VERSION >= 1
    assert "PLANNER_PROMPT_VERSION" in open(lp.__file__, encoding="utf-8").read().split("def plan_logical")[0][-400:]
