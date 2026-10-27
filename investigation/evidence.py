"""
Evidence: every Deep Investigation tool writes one record here and returns
it. A record carries a short summary (numbers and labels the model may read),
its tables (for the screen and the download) and the invariant checks the
tool ran on itself. A record whose checks failed can never back a claim.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


class ToolError(ValueError):
    """A tool can't answer this (bad input, missing data, nothing in scope).
    Raised with a plain message instead of returning an empty result that
    would read as a real zero."""


@dataclass(frozen=True)
class Check:
    name: str
    passed: bool
    detail: str = ""


@dataclass
class Evidence:
    result_id: str
    tool: str
    args: dict
    summary: dict
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    checks: list[Check] = field(default_factory=list)

    @property
    def checks_passed(self) -> bool:
        return all(c.passed for c in self.checks)


class EvidenceStore:
    """One investigation's results, in the order they were made (E1, E2, ...)."""

    def __init__(self) -> None:
        self._items: dict[str, Evidence] = {}

    def add(self, tool: str, args: dict, summary: dict, tables: dict | None = None,
            checks: list[Check] | None = None) -> Evidence:
        result_id = f"E{len(self._items) + 1}"
        ev = Evidence(result_id, tool, dict(args), summary, dict(tables or {}), list(checks or []))
        self._items[result_id] = ev
        return ev

    def get(self, result_id: str) -> Evidence:
        if result_id not in self._items:
            raise ToolError(f"there is no result {result_id} in this investigation")
        return self._items[result_id]

    def all(self) -> list[Evidence]:
        return list(self._items.values())
