"""
Branch -> manager email mapping for the "draft a priority email" feature
(investigator/email_draft.py). A plain Python dict, not YAML -- this
codebase has no existing YAML dependency (config.py's own convention is
flat Python constants, not a config file format), so this follows suit
rather than introducing pyyaml for one small mapping. Mirrors the SHAPE of
the sibling "Excel Automation" project's config/recipients.yaml (to/cc per
entity), just keyed by branch instead of region, and as a Python dict
instead of YAML.

Edit BRANCH_RECIPIENTS directly to add a branch's manager email(s). An
unmapped branch is not an error -- the email draft still builds, just with
a blank To field for the user to fill in themselves (see
email_draft.py::build_eml_bytes).
"""
from __future__ import annotations

BRANCH_RECIPIENTS: dict[str, dict[str, list[str]]] = {
    # "MAHAD": {"to": ["branch.manager@example.com"], "cc": []},
}


def get_branch_recipients(branch: str) -> tuple[list[str], list[str]]:
    """(to, cc) email lists for a branch name, case/whitespace-insensitive
    (branch names in the uploaded data are manually typed and inconsistently
    cased). Empty lists (never an error) when the branch isn't mapped yet."""
    if not branch:
        return [], []
    entry = BRANCH_RECIPIENTS.get(str(branch).strip().upper(), {})
    return list(entry.get("to") or []), list(entry.get("cc") or [])
