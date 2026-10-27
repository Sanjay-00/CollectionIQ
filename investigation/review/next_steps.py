"""
What a leader can do after looking deeper at a region from the portfolio
review (spec 7.1: choice -> investigate -> act -> done). Each step reuses an
Investigator step that already exists:

  priority loans in the branch with the largest gap (when the deep-dive named
  one): priority_menu scoped to that branch, where "Draft email" builds a
  downloadable email for the branch manager (never sent)
  priority loans across the region: priority_menu scoped to the region
  mark it handled, or go back to the briefing

Pure: no Streamlit, no data access, so the options are tested on their own.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from investigation.review.brief import LIST_NAMES


@dataclass(frozen=True)
class NextStep:
    id: str
    label: str
    step_type: str | None = None          # an Investigator STEP_REGISTRY name, or None for close / back
    params: dict = field(default_factory=dict)


def next_steps(region: str, list_name: str, branch: str | None = None) -> list[NextStep]:
    steps = []
    if branch:
        steps.append(NextStep(f"priority_branch:{branch}", f"Priority loans in {branch} (largest branch gap)",
                              "priority_menu", {"scope_col": "branch", "scope_value": branch}))
    steps.append(NextStep(f"priority_region:{region}", f"Priority loans across {region}",
                          "priority_menu", {"scope_col": "region", "scope_value": region}))
    steps.append(NextStep(f"close:{list_name}:{region}", f"Mark {region} ({LIST_NAMES[list_name]}) as handled"))
    steps.append(NextStep("back", "Back to the briefing"))
    return steps


def largest_branch(brief, candidate_id: str) -> str | None:
    """The branch with the largest gap in this candidate's deep-dive, if any."""
    dive = next((d for d in (brief.deep_dives if brief else []) if d.candidate_id == candidate_id), None)
    return dive.branches[0][0] if dive and dive.branches else None
