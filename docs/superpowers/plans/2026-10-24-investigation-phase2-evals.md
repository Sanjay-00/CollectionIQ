# Deep Investigation Phase 2 (like for like, test books, cases, scorer) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (chosen: native) to implement this plan task-by-task.

**Goal:** Investigations compare like for like when a branch is in only one file, and a fixed set of planted-cause cases with a scorer exists, so Phase 3's baseline and Phase 4's agent can be graded.

**Architecture:** `investigation/prepare.py` runs `data_health`, leaves out branches present in one file only, and re-checks. `investigation/synthetic.py` (moved from `tests/investigation/books.py`) builds two-month books with planted causes. `investigation/evaluation.py` defines the result shape an investigation returns and scores it against a case's expected answer. `investigation/eval_cases.py` holds the cases; each is proven recoverable by running the Phase 1 tools directly.

**Tech Stack:** Python 3.12, pandas 3, pytest. Phase 1 tools unchanged except where stated.

**Spec:** `docs/superpowers/specs/2026-10-23-investigation-agent-design.md` (sections 7.1, 12, 15 phase 2) plus the user decision of 2026-10-24: compare like for like instead of halting on branch coverage gaps.

## Global Constraints

- Same as Phase 1 (user runs git; thresholds in `config.py`; no customer fields in summaries; "delinquent", "MoM", "% points", no em/en dashes in returned text; full suite green after every task).
- Like for like leaves out ONLY branches flagged `units_missing` or `units_new` "missing from last month's file", from both months, and always reports what it left out with loan counts. Every other halt still halts.
- Cases use synthetic data only; the expected answer is written in the case before any run.

## Review Focus

- A book where the left-out branches are the whole scope (scope = that branch): must refuse plainly, not return empty figures.
- Like for like changes the rate: the report must show both the full-file and the like-for-like change.
- A case's expected driver given as a label the tool spells differently ("tractor" vs "TRACTOR"): scoring compares case-insensitively.
- A result that names the right driver but cites an evidence id whose checks failed: scored as a failure.
- Hold-out cases must never be imported by prompt-tuning code (Phase 4); the list lives in one place.

---

### Task 1: Like for like (`investigation/prepare.py`)

- `data_health` findings `units_missing` and `units_new` gain `"units": [str, ...]` (the branches concerned; for `units_new` only the "missing from last month's file" ones).
- `PeriodPair.without_units(units: list[str]) -> PeriodPair` drops those branches (by each loan's `loan_label("branch")`) from both months; `attrs` kept.
- `@dataclass Prepared: pair: PeriodPair; status: str ("ok" | "halted_data"); health: Evidence; adjustment: Evidence | None`.
- `prepare(store, pair, scope=None) -> Prepared`: run `data_health`; if every halt is a branch-coverage halt, build `pair.without_units(...)`, run `data_health` again and record a `like_for_like` Evidence (`summary = {"excluded": [{"unit", "reason", "loans_prev", "loans_curr"}], "full_change_pp", "like_for_like_change_pp"}`); `status = "halted_data"` when any halt remains. Warn-level coverage gaps are also left out (they distort splits the same way).
- Tests (`tests/investigation/test_prepare.py`): missing-from-last-month branch → excluded, status ok, second health has no halt, both changes reported; branch missing this month → same; population jump → still `halted_data`, no adjustment; clean book → no adjustment; scope equal to an excluded branch → `ToolError`.

### Task 2: `investigation/synthetic.py`

- Move `tests/investigation/books.py` into `investigation/synthetic.py` unchanged in behaviour; `tests/investigation/books.py` becomes `from investigation.synthetic import *  # noqa` so Phase 1 tests keep working.
- New plants (keyword arguments of `make_book`):
  - `vintage_rate_up=(start, end, pts)`: loans agreed in [start, end] get pts % points more 30+ this month.
  - `product_by_vintage=(product, start)`: product's loans are agreed from `start` onward (confounds product with vintage).
  - `exec_rate_up=(branch, executive, pts)`: one executive's loans in one branch, pts % points more 30+.
  - `cure_block=f`: fraction f of loans that would cure this month stay 30+ (with `churn`).
  - `base_grow=k`: k loans that had no EMI due last month (NA) become STD this month (base rises, numerator flat).
  - `segment_swing=(dimension column, value, n, pts)`: a small segment of n loans with value, pts more 30+.
  - `duplicates=k`: k duplicate rows in this month's frame.
  - `attrs=dict`: attrs set on this month's frame (header renames, duplicates dropped).
- Tests (`tests/investigation/test_synthetic.py`): each plant's direct effect measured with `metric_change` / counts (one test per plant).

### Task 3: Result shape and scorer (`investigation/evaluation.py`)

- `@dataclass Driver: dimension: str; segment: str; evidence_id: str`.
- `@dataclass InvestigationResult: status: str ("concluded" | "no_material_change" | "halted_data"); drivers: list[Driver]; mechanism: str | None ("mix" | "rate" | "flow" | "base"); deep: list[Driver]; flags: list[str]; untestable: list[str]; excluded_units: list[str]; tool_calls: int = 0; model_calls: int = 0; tokens: int = 0; seconds: float = 0.0`.
- `@dataclass Expected: status: str; drivers: list[tuple[str, str]] = []; any_driver_of: list[tuple[str, str]] = []; forbidden: list[tuple[str, str]] = []; mechanism: str | None = None; deep: list[tuple[str, str]] = []; flags: list[str] = []; untestable: list[str] = []; excluded_units: list[str] = []`.
- `@dataclass Score: case_id: str; checks: dict[str, bool]; passed: bool` (passed = all checks).
- `score(case_id, expected, result, store) -> Score` with checks: `status`, `drivers` (every expected driver among result drivers, case-insensitive), `forbidden`, `mechanism` (when expected), `deep`, `flags` (each expected flag present), `untestable`, `excluded_units`, `evidence_valid` (every cited evidence id exists in the store and `checks_passed`).
- Tests (`tests/investigation/test_evaluation.py`): a perfect result passes; each wrong field fails exactly its own check; a driver citing a failed-check evidence fails `evidence_valid`; label case ignored.

### Task 4: Cases (`investigation/eval_cases.py`)

- `@dataclass Case: id: int; name: str; build: Callable[[], tuple[pd.DataFrame, pd.DataFrame]]; expected: Expected; holdout: bool = False`.
- `CASES: list[Case]` — spec 12.2 cases 1 to 16, adjusted by the like-for-like decision (case 3: missing branch → `excluded_units` = that branch, status concluded on the planted cause), plus case 17: branch missing from last month's file (the real Sept 2026 fault) with a planted product cause. Hold-out: 4, 9, 12, 15, 16 (`HOLDOUT_IDS`).
- Every case builds 20,000 to 40,000 loans with `churn=0.06` unless the case is about noise-free data, months 2026-08 / 2026-09.
- `recoverable(case) -> dict[str, bool]`: runs `prepare`, `metric_change`, `flow_bridge` and `scan_dimensions` directly and checks the plant is visible (expected flags raised; expected drivers notable in their dimension's `mix_rate`; status material / not material as expected; base / flow mechanism visible in `metric_change` / `flow_bridge`; untestable dimensions in `scan.unavailable`; deep driver notable in a `mix_rate` scoped to its parent).
- Tests (`tests/investigation/test_eval_cases.py`): parametrized over `CASES`, every `recoverable` check true; ids unique; hold-outs exactly 4, 9, 12, 15, 16.

### Task 5: Real-file run like for like

- `investigation/real_check.py` runs `prepare` first and prints the exclusion (units, loan counts, full and like-for-like change), then the tools on the prepared pair.
- Run on the real Aug → Sept files: expect HINGL and WASHM excluded, like-for-like change +0.71 % points (27.57% → 28.28%), status ok, every check passed.
