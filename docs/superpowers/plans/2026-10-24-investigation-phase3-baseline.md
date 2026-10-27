# Deep Investigation Phase 3 (rival test, drill, sample loans, fixed-script baseline) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans (native, chosen by the user).

**Goal:** A complete investigation with no model call: the fixed script runs the tools in a set order, returns an `InvestigationResult` and a plain report, is scored on all 19 cases, and runs on the real Aug -> Sept files for the user to review.

**Architecture:** Three tools join the Phase 1 set: `controlled_decompose` and `drill` reuse `mix_rate` / `scan_dimensions` inside a scope; `sample_loans` lists the loans behind a finding. `investigation/baseline.py` is the script. `investigation/report.py` turns a result and its evidence into the sections of spec 11. `thirty_plus_pct` joins AI Query's registry with a consistency test (spec 9.5 decision 3).

**Spec:** `docs/superpowers/specs/2026-10-23-investigation-agent-design.md` sections 7.6 to 7.8, 8, 9.5, 11, 12, 15 phase 3.

## Global Constraints
- As Phases 1 and 2 (user runs git; thresholds in config; no customer fields in summaries; wording rules; suite green after each task).
- The baseline never calls a model and never names a driver that the scorer's rules (spec 12.2 notes) would reject.
- `sample_loans` tables may hold customer rows (for the screen); its summary holds counts only.

## Review Focus
- A case where the top dimension's best segment is beyond its share but under 10% of the change: no driver, not the next segment down.
- The rival test when the second dimension is unavailable or has one value inside the scope.
- A drill whose scope leaves under the drill minimum: no deep driver, no error.
- A non-material change with offsetting moves (case 9) versus pure noise (case 6).
- A run on the real files where the like-for-like adjustment applied: the report states it before any finding.

---

### Task 1: `thirty_plus_pct` in the registry
- `registry/ontology.py` METRICS gains `thirty_plus_pct`: `count_ratio`, numerator `Arrears / EMI >= 1`, denominator `curr_bucket in ELIGIBLE_BUCKETS`, scale 100, grain loan.
- Test (`tests/test_metric_consistency.py` style, new `tests/investigation/test_registry_thirty_plus.py`): on a mixed book incl. NA and non-RUN loans... compiled through `compile_logical` + `execute_plan` grouped by region, every region's rate equals `side_counts(...)["rate"]` from `investigation.tools.change` on that region's RUN loans.

### Task 2: `controlled_decompose`
- `controlled_decompose(store, pair, dimension, segment, control, metric_id="thirty_plus_rate", scope=None) -> Evidence` (tool "controlled_decompose"): for each `control` level holding at least `INVESTIGATION_MIN_SEGMENT_DRILL` eligible loans of `segment`, run `mix_rate(dimension, scope={**scope, control: level}, min_size=INVESTIGATION_MIN_SEGMENT_DRILL)` and read the segment's row.
- `summary = {"dimension", "segment", "control", "levels": [{"level", "loans", "beyond_share", "rate_change_pp", "evidence_id"}], "tested_loans", "persisting_loans", "persists": bool}`; `persists` = the segment is beyond its share in levels holding at least half of the tested loans. No testable level -> `ToolError`.
- Tests: case 10's book: product TRACTOR persists within vintage; vintage 2025-Q3 does not persist within product. Unavailable control -> `ToolError`.

### Task 3: `drill`
- `scan_dimensions` gains `include_drill_only: bool = False` and `min_size: int | None = None` (passed to `mix_rate`).
- `drill(store, pair, dimension, segment, metric_id=..., scope=None) -> Evidence` (tool "drill"): `scan_dimensions` with scope `{dimension: segment}`, drill-only dimensions included, `min_size=INVESTIGATION_MIN_SEGMENT_DRILL`; the drilled dimension itself is left out. Summary as scan's plus `{"parent": [dimension, segment]}`.
- Tests: case 13's book: drill branch BR3 ranks executive first with "EXEC7 (BR3)" beyond its share; drilling a segment too small for any split returns an empty ranking, not an error.

### Task 4: `sample_loans`
- `sample_loans(store, pair, scope: dict, kind: str, n: int = 50) -> Evidence` (tool "sample_loans"); kind in `thirty_plus_now` (30+ this month), `rolled_in` (matched, not 30+ last month, 30+ now), `cured` (matched, 30+ last month, STD / 1-30 now); largest SOH first, at most `n` (cap 50).
- `tables["loans"]`: Loan No, branch, executive, last / this month's bucket, SOH and the customer name and mobile when present; `summary = {"kind", "scope", "count_all", "shown"}`.
- Tests: kinds pick the right loans on a hand book; n capped at 50; summary has no customer field.

### Task 5: Baseline (`investigation/baseline.py`)
- `run_baseline(curr: pd.DataFrame, prev: pd.DataFrame, curr_month: str, prev_month: str, scope: dict | None = None) -> tuple[InvestigationResult, EvidenceStore]`, steps:
  1. `prepare`; halted -> status `halted_data`, flags, nothing else.
  2. `metric_change`; `flow_bridge`; `scan_dimensions`.
  3. Not material: drivers = top-ranked dimension's segments beyond their share with |contribution| >= `INVESTIGATION_NO_CHANGE_PP` (offsetting moves); status `no_material_change`.
  4. Material and |base effect| > |numerator effect|: mechanism `base`, no drivers, `concluded`.
  5. Material, no dimension with concentration >= `INVESTIGATION_CONCENTRATION_MIN_PCT`: mechanism `flow`, no drivers, `concluded` when roll-ins exceed cures, else `inconclusive`.
  6. Otherwise lead = top dimension, its first segment beyond its share and >= `INVESTIGATION_DRIVER_MIN_SHARE_PCT` of the change (none -> step 5). Rival: when the second-ranked dimension also clears the concentration line, `controlled_decompose(lead within second)`; if it doesn't persist, `controlled_decompose(second's top segment within lead dimension)`, and if that persists the second becomes the lead. Drivers: the lead's qualifying segments (up to two). Mechanism: `mix` when |mix| > |rate| for the first driver, else `rate`. Deep: `drill(lead)`; its top dimension's first qualifying segment (same rules, against the scoped change).
  7. `flags = prepared.flags`; `untestable` = scan's unavailable dimensions; `excluded_units` from the adjustment; counts of tool calls and seconds.
- Tests: `tests/investigation/test_baseline.py` scores the baseline on every case with `evaluation.score`; cases it is expected to miss are `xfail(strict=True)` with the reason (decided from the first run, recorded in the ledger).

### Task 6: Report (`investigation/report.py`)
- `build_report(result, store) -> dict` with sections in spec 11 order: `headline`, `adjustment`, `what_changed` (rate both ways, bridge lines), `where` (drivers with contribution, mix / rate), `checks` (rival test outcome), `not_tested`, `data_notes`, `next_steps` (Action Lists names: "New defaulters" for roll-ins from STD, "1-30 → SMA-1+" for roll-ins from 1-30), and `markdown` joining them; wording rules apply; causal verbs never used.
- Tests: headline numbers equal the evidence; no causal verbs; the adjustment sentence comes before the findings; halted and no-change results get their own short reports.

### Task 7: Real files
- `real_check.py` runs the baseline and prints the report markdown; run on Aug -> Sept; record the result for the user's review.
