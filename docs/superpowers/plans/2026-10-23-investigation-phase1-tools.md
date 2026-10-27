# Deep Investigation Phase 1 (metric, data layer, five tools) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A tested `investigation/` package that measures the 30+ rate for two monthly closings and explains its change with five deterministic tools: `data_health`, `metric_change`, `flow_bridge`, `mix_rate`, `scan_dimensions`.

**Architecture:** Pure pandas functions over a `PeriodPair` (both months plus one outer join on Loan No). Every tool writes one `Evidence` record (summary dict, tables, invariant checks) to an `EvidenceStore` and returns it; a tool that cannot answer raises `ToolError`, never an empty result. No model calls, no UI in this phase.

**Tech Stack:** Python 3.12, pandas 3.0.3, pytest. Reuses `utils.bucket_from_arrears_emi` buckets (`curr_bucket`), `utils.prev_held_now`, `utils.loan_key`, `utils.count_agreed_after_month`.

**Spec:** `docs/superpowers/specs/2026-10-23-investigation-agent-design.md` (sections 3, 4, 5, 6, 7.1 to 7.5, 15 phase 1).

## Global Constraints

- The user runs every git write. A "Commit" step means: stop, list the files and the suggested message (5 to 15 words, no Co-Authored-By trailer, no tool names).
- Metric `thirty_plus_rate` v1: eligible = `Loan Status` == "RUN" (all rows when the column is absent) and `curr_bucket` in {STD, 1-30 DPD, SMA-1, SMA-2, NPA}; numerator = eligible and `curr_bucket` in {SMA-1, SMA-2, NPA}; rate = 100 × numerator / eligible (ratio of summed counts, never an average of %). Companion: numerator / running loans.
- A loan belongs to this month's executive (`utils.prev_held_now`). Region, branch and every other dimension use each month's own labels.
- An undefined rate (zero eligible) is `None`, never 0.
- Every threshold lives in `config.py` (Task 1 names them); no literal thresholds in `investigation/`.
- No customer names, mobile numbers or loan rows in any `Evidence.summary` (tables may hold loan-level rows only in Phase 3's `sample_loans`; none in this phase).
- Wording in any text a tool returns: "delinquent", "MoM", "% points"; no em or en dashes.
- New tests live in `tests/investigation/` (add `tests/investigation/__init__.py`); the full suite (`python -m pytest -q`) must stay green after every task.

## Review Focus

- A scope that exists in only one month (a branch new in Sept): `metric_change` must raise `ToolError` naming the month with no loans, not divide by zero (Task 4 test `test_scope_new_this_month_is_refused`).
- The same loan typed differently in the two files (`12345.0` vs `"12345"`, lower case): the join must match it via `utils.loan_key` (Task 3 test `test_loan_numbers_typed_differently_still_match`).
- Blank dimension values: kept as a "Not recorded" segment so every split stays exhaustive and reconciles (Task 2 test `test_blank_values_become_not_recorded`).
- Months passed the wrong way round or equal: `make_pair` raises (Task 3 test `test_months_must_be_in_order`).
- A flat total hiding offsetting segment moves: `scan_dimensions` must still rank by absolute segment contribution instead of dividing by a near-zero total (Task 7 test `test_flat_total_ranks_by_largest_segment_move`).

---

### Task 1: Config, metric registry, evidence store

**Files:**
- Modify: `config.py` (append an "Investigation" block)
- Create: `investigation/__init__.py` (empty), `investigation/metrics.py`, `investigation/evidence.py`
- Test: `tests/investigation/__init__.py`, `tests/investigation/test_metrics_evidence.py`

**Interfaces:**
- Produces (config):
  `INVESTIGATION_MIN_SEGMENT = 200`, `INVESTIGATION_MIN_SEGMENT_DRILL = 50`, `INVESTIGATION_NOISE_SE = 2.0`,
  `INVESTIGATION_NO_CHANGE_PP = 0.20`, `INVESTIGATION_RECONCILE_TOL_PP = 0.01`,
  `INVESTIGATION_DPD_DISAGREE_WARN_PCT = 2.0`, `INVESTIGATION_POP_CHANGE_HALT_PCT = 15.0`,
  `INVESTIGATION_MISSING_UNIT_HALT_PCT = 1.0`, `INVESTIGATION_TOP_SEGMENTS = 2`
- Produces (`investigation/metrics.py`):
  `ELIGIBLE_BUCKETS = ("STD", "1-30 DPD", "SMA-1", "SMA-2", "NPA")`, `THIRTY_PLUS_BUCKETS = ("SMA-1", "SMA-2", "NPA")`;
  `@dataclass(frozen=True) class MetricDef: id: str; label: str; version: int; running: Callable[[pd.DataFrame], pd.Series]; eligible: Callable[[pd.DataFrame], pd.Series]; numerator: Callable[[pd.DataFrame], pd.Series]`
  (each callable returns a boolean Series aligned to the frame; `numerator` is already restricted to eligible);
  `METRICS: dict[str, MetricDef]` with one entry, id `"thirty_plus_rate"`, label `"30+ (SMA-1 and worse: one or more EMIs overdue)"`, version 1;
  `get_metric(metric_id: str) -> MetricDef` raising `ToolError` for an unknown id (message lists known ids).
- Produces (`investigation/evidence.py`):
  `class ToolError(ValueError)`;
  `@dataclass(frozen=True) class Check: name: str; passed: bool; detail: str = ""`;
  `@dataclass class Evidence: result_id: str; tool: str; args: dict; summary: dict; tables: dict[str, pd.DataFrame]; checks: list[Check]` with property `checks_passed -> bool` (all checks passed; True when no checks);
  `class EvidenceStore` with `add(tool: str, args: dict, summary: dict, tables: dict | None = None, checks: list[Check] | None = None) -> Evidence`, `get(result_id: str) -> Evidence` (raises `ToolError` when unknown), `all() -> list[Evidence]` in insertion order. Ids are `"E1"`, `"E2"`, ... per store.

- [ ] **Step 1: Write the failing tests** in `tests/investigation/test_metrics_evidence.py`:
  - `test_thirty_plus_counts_one_or_more_emis_overdue`: frame with buckets `["STD", "1-30 DPD", "SMA-1", "SMA-2", "NPA", "NA"]`, all RUN → `eligible.sum() == 5`, `numerator.sum() == 3`, `running.sum() == 6`.
  - `test_non_running_loans_are_not_eligible`: an NPA row with `Loan Status` "CLOSED" → not running, not eligible, not numerator.
  - `test_unknown_metric_is_refused`: `get_metric("npa_rate")` raises `ToolError` with `"thirty_plus_rate"` in the message.
  - `test_store_numbers_results_in_order`: two `add` calls → ids `"E1"`, `"E2"`; `get("E2").tool` matches; `get("E9")` raises `ToolError`.
  - `test_one_failed_check_marks_the_result`: `checks=[Check("a", True), Check("b", False, "x")]` → `checks_passed is False`.

- [ ] **Step 2: Run** `python -m pytest tests/investigation/test_metrics_evidence.py -q` → FAIL (`ModuleNotFoundError: investigation`).

- [ ] **Step 3: Implement** the config block, `investigation/metrics.py` and `investigation/evidence.py` per the Interfaces block. `running`: `Loan Status` stripped upper == "RUN", or all True when absent.

- [ ] **Step 4: Run** the same command → PASS; then `python -m pytest -q` → all pass.

- [ ] **Step 5: Commit** `config.py investigation/__init__.py investigation/metrics.py investigation/evidence.py tests/investigation/` — "Investigation: 30+ metric definition and evidence store".

---

### Task 2: Dimensions

**Files:**
- Create: `investigation/dimensions.py`
- Test: `tests/investigation/test_dimensions.py`

**Interfaces:**
- Consumes: `ToolError` (Task 1), config `INVESTIGATION_MIN_SEGMENT`, `INVESTIGATION_MIN_SEGMENT_DRILL`.
- Produces:
  `NOT_RECORDED = "Not recorded"`;
  `@dataclass(frozen=True) class Dimension: id: str; label: str; columns: tuple[str, ...]; kind: str; scan: bool; top_n: int | None = None`
  where `kind` is one of `"category"`, `"top_n"`, `"quarter"`, `"fixed_bands"`, `"quantile_bands"`, `"vehicle_age"`, `"executive"`;
  `DIMENSIONS: dict[str, Dimension]`, in this order, with these ids, labels, columns and kinds:

  | id | label | columns | kind | scan |
  |---|---|---|---|---|
  | zone | Zone | Zone | category | yes |
  | region | Region | RegionName | category | yes |
  | branch | Branch | Unit | category | yes |
  | product | Product | SegmentName | category | yes |
  | channel | Channel | CHANNEL | category | yes |
  | scheme | Scheme | scheme | top_n (20) | yes |
  | make | Vehicle make | Make | top_n (25) | yes |
  | vintage | Agreement period | Ag_Date | quarter | yes |
  | tenure | Tenure (months) | Tenure | fixed_bands | yes |
  | loan_amount | Loan amount | Loan Amount | quantile_bands | yes |
  | vehicle_age | Vehicle age | Year Of Manufacture | vehicle_age | yes |
  | due_date | EMI date | Due Dt | category | yes |
  | fleet | Loans per customer | Customer Loan Count | fixed_bands | yes |
  | nach | NACH status | NACHStatus | category | yes |
  | fuel | Fuel | FUEL_TYPE | category | yes |
  | executive | Executive | MNT NAME, Unit | executive | no |
  | dealer | Source / dealer | SRC Name | top_n (30) | no |

  Band edges (label text exactly): tenure `≤12`, `13-24`, `25-36`, `37-48`, `49+`; fleet `1`, `2-4`, `5+`; vehicle age (years before the month's own year) `0-2`, `3-5`, `6-8`, `9+`; loan amount: quintile edges computed on the reference (previous) month, labelled `Q1 (lowest)` … `Q5 (highest)`; vintage: `YYYY-Qn`, except the six months up to and including `as_of` labelled `YYYY-MM`.
  `segment_labels(dim: Dimension, df: pd.DataFrame, as_of: str, ref: pd.DataFrame | None = None, top: list[str] | None = None) -> pd.Series` — one label per row, blanks → `NOT_RECORDED`, top_n values outside `top` → `"Other"`, executive → `"NAME (UNIT)"`;
  `top_values(dim: Dimension, ref: pd.DataFrame) -> list[str] | None` — the `top_n` most frequent values of the reference month (None for other kinds);
  `unavailable_reason(dim: Dimension, prev: pd.DataFrame, curr: pd.DataFrame) -> str | None` — `"no <column> column in <month word>"` when a column is missing from either month, `"only one value"` when the current month has fewer than 2 distinct non-blank values, else None;
  `get_dimension(dim_id: str) -> Dimension` raising `ToolError` for an unknown id.

- [ ] **Step 1: Write the failing tests** in `tests/investigation/test_dimensions.py`:
  - `test_registry_matches_the_spec`: ids in the table order; `scan` False only for `executive`, `dealer`.
  - `test_blank_values_become_not_recorded`: FUEL_TYPE `["DIESEL", None, ""]` → `["DIESEL", "Not recorded", "Not recorded"]`.
  - `test_tenure_and_fleet_bands`: Tenure `[12, 13, 48, 60]` → `["≤12", "13-24", "37-48", "49+"]`; Customer Loan Count `[1, 3, 7]` → `["1", "2-4", "5+"]`.
  - `test_loan_amount_bands_use_last_months_edges`: ref amounts 1..100 → a current value of 1,000 lands in `"Q5 (highest)"`, 1 in `"Q1 (lowest)"`.
  - `test_vintage_recent_months_then_quarters`: as_of `"2026-09"`, dates 2026-09-10, 2026-04-01, 2026-03-31, 2025-11-05 → `["2026-09", "2026-04", "2026-Q1", "2025-Q4"]`.
  - `test_top_n_groups_the_rest_as_other`: make with 30 distinct values, `top=top_values(...)` of size 25 → 5 labelled `"Other"`.
  - `test_executive_label_is_name_and_branch`: → `"RAJ (AKLUJ)"`.
  - `test_missing_or_flat_dimension_is_unavailable`: no FUEL_TYPE in prev → reason contains `"no FUEL_TYPE column"`; one-value CHANNEL → `"only one value"`.

- [ ] **Step 2: Run** `python -m pytest tests/investigation/test_dimensions.py -q` → FAIL.
- [ ] **Step 3: Implement** `investigation/dimensions.py` per Interfaces. Vehicle age uses `pd.Period(as_of, "M").year - YOM`; non-numeric YOM → `NOT_RECORDED`.
- [ ] **Step 4: Run** → PASS; full suite → pass.
- [ ] **Step 5: Commit** `investigation/dimensions.py tests/investigation/test_dimensions.py` — "Investigation: approved slicing dimensions with bands and blank handling".

---

### Task 3: PeriodPair and the loan join

**Files:**
- Create: `investigation/panel.py`
- Test: `tests/investigation/test_panel.py`

**Interfaces:**
- Consumes: `ToolError`, `get_dimension`, `segment_labels`, `top_values`, `NOT_RECORDED` (Tasks 1, 2); `utils.loan_key`, `utils.prev_held_now`.
- Produces:
  `@dataclass class PeriodPair: prev: pd.DataFrame; curr: pd.DataFrame; prev_month: str; curr_month: str` (months `"YYYY-MM"`) with
  `scoped(self, scope: dict[str, str] | None) -> "PeriodPair"` — keeps, in each month, the rows whose `segment_labels` for each `{dimension_id: label}` equal the label (prev month labels for the executive dimension come from `prev_held_now`; a scope matching nothing in BOTH months raises `ToolError("no loans match <label> in either month")`);
  `joined(self) -> pd.DataFrame` — cached; one row per Loan No across both months with columns `Loan No`, `in_prev`, `in_curr` (bool), `prev_bucket`, `curr_bucket` (`None` when absent that month), `prev_running`, `curr_running` (bool), plus every dimension column present, taken from the current month when the loan is in it, else from the previous month, and `left_book` (`in_prev and not in_curr`);
  `labels(self, dim_id: str, month: str) -> pd.Series` — segment labels for `"prev"` or `"curr"`, banded/top-n against the previous month as reference, and for executive in `"prev"`: this month's owner for loans still held, `"Left the running book"` for loans that left.
  `make_pair(curr: pd.DataFrame, prev: pd.DataFrame, curr_month: str, prev_month: str) -> PeriodPair` — raises `ToolError` when either frame is empty, lacks `Loan No` or `curr_bucket`, or `prev_month >= curr_month`; Loan No in both frames passed through `utils.loan_key` (on copies).

- [ ] **Step 1: Write the failing tests** in `tests/investigation/test_panel.py`:
  - `test_join_marks_matched_new_and_left_loans`: prev L1,L2; curr L2,L3 → L1 `left_book`, L2 in both, L3 `in_curr` only; buckets carried per month, `None` where absent.
  - `test_loan_numbers_typed_differently_still_match`: prev Loan No `12345.0`, curr `"12345"` → one joined row, in both months.
  - `test_months_must_be_in_order`: `make_pair(..., curr_month="2026-08", prev_month="2026-09")` and equal months raise `ToolError`.
  - `test_scope_filters_each_month_by_its_own_labels`: branch moved AKLUJ→PUNE between months; `scoped({"branch": "AKLUJ"})` keeps it in prev only.
  - `test_executive_last_month_is_the_loans_held_now`: loan moved RAJ→AMIT; `labels("executive", "prev")` gives `"AMIT (B1)"`; a loan that left gives `"Left the running book"`.
  - `test_scope_matching_nothing_is_refused`: `scoped({"branch": "NOWHERE"})` raises `ToolError`.

- [ ] **Step 2: Run** `python -m pytest tests/investigation/test_panel.py -q` → FAIL.
- [ ] **Step 3: Implement** `investigation/panel.py` per Interfaces; outer merge on `Loan No` with month suffixes, then coalesce dimension columns current-first.
- [ ] **Step 4: Run** → PASS; full suite → pass.
- [ ] **Step 5: Commit** `investigation/panel.py tests/investigation/test_panel.py` — "Investigation: two-month loan panel with outer join and scoping".

---

### Task 4: `metric_change`

**Files:**
- Create: `investigation/tools/__init__.py` (re-exports the five tools as they land), `investigation/tools/change.py`
- Test: `tests/investigation/test_metric_change.py`

**Interfaces:**
- Consumes: `PeriodPair`, `get_metric`, `EvidenceStore`, `Check`, `ToolError`, config `INVESTIGATION_RECONCILE_TOL_PP`.
- Produces: `metric_change(store: EvidenceStore, pair: PeriodPair, metric_id: str = "thirty_plus_rate", scope: dict | None = None) -> Evidence` with `tool == "metric_change"` and
  `summary = {"metric": id, "label": str, "version": int, "prev_month": str, "curr_month": str, "prev": Side, "curr": Side, "change_pp": float, "change_rel_pct": float | None, "all_running_change_pp": float, "numerator_effect_pp": float, "base_effect_pp": float}`
  where `Side = {"running": int, "eligible": int, "numerator": int, "rate": float, "rate_all_running": float}` (rates in %, unrounded);
  `numerator_effect_pp = 100 * (N1 - N0) / D1`, `base_effect_pp = 100 * N0 * (1/D1 - 1/D0)`;
  checks: `"effects_add_up"` (|numerator + base − change| ≤ tol), `"numerator_within_eligible"` (N ≤ D ≤ running both months).
  Raises `ToolError("no eligible loans in <Mon YYYY> for this scope")` when D0 or D1 is 0.

- [ ] **Step 1: Write the failing tests:**
  - `test_rate_change_and_its_two_parts`: prev 10 eligible / 2 at 30+ / 1 NA; curr 20 eligible / 5 at 30+ → prev rate 20.0, curr 25.0, `change_pp == 5.0`, `change_rel_pct == 25.0`, `numerator_effect_pp == 15.0`, `base_effect_pp == -10.0`, `rate_all_running` prev `200/11`, `checks_passed`.
  - `test_no_emi_loans_are_outside_the_base`: adding 100 NA loans to curr changes `rate_all_running` but not `rate`.
  - `test_scope_new_this_month_is_refused`: a branch only in curr, scoped → `ToolError` naming the previous month (`"Aug 2026"`).
  - `test_flat_previous_rate_has_no_relative_change`: prev numerator 0 → `change_rel_pct is None`.

- [ ] **Step 2: Run** `python -m pytest tests/investigation/test_metric_change.py -q` → FAIL.
- [ ] **Step 3: Implement** `metric_change` in `investigation/tools/change.py`; scope via `pair.scoped(scope)`.
- [ ] **Step 4: Run** → PASS; full suite → pass.
- [ ] **Step 5: Commit** `investigation/tools/ tests/investigation/test_metric_change.py` — "Investigation: metric change tool with numerator and base effects".

---

### Task 5: `flow_bridge`

**Files:**
- Modify: `investigation/tools/change.py`, `investigation/tools/__init__.py`
- Test: `tests/investigation/test_flow_bridge.py`

**Interfaces:**
- Consumes: Task 3 `joined()`, `labels()`; Task 1 metric sets.
- Produces: `flow_bridge(store: EvidenceStore, pair: PeriodPair, metric_id: str = "thirty_plus_rate", scope: dict | None = None, by: str | None = None) -> Evidence` with `tool == "flow_bridge"`;
  `summary["lines"]` — ordered dict of ints with exactly these keys: `opening`, `in_from_std`, `in_from_1_30`, `in_from_no_emi`, `new_to_book`, `cured_to_std`, `cured_to_1_30`, `became_no_emi`, `left_book`, `closing`;
  `summary["base_lines"]` — `opening`, `new_to_book`, `in_from_no_emi`, `became_no_emi`, `left_book`, `closing` for the eligible base;
  `summary["left_book_last_bucket"]` — `{bucket: count}` for 30+ loans that left (hint only);
  `tables["by"]` (only when `by` is given) — one row per segment, columns `segment` + the ten line keys, segment = current label for loans in the current month, previous label for loans that left (for executive, `"Left the running book"`);
  checks: `"bridge_closes"` (opening + ins − outs == closing), `"base_bridge_closes"`, `"each_segment_closes"` (when `by`).
  "In" lines count matched loans moving from that bucket into a 30+ bucket; "cured" lines count matched loans moving from 30+ to STD / 1-30 DPD; a loan not running in a month counts as absent that month.

- [ ] **Step 1: Write the failing tests:**
  - `test_bridge_lines_on_a_hand_built_book`: one loan for each path (STD→SMA-1, 1-30→NPA, NA→SMA-2, new SMA-1, SMA-1→STD, SMA-2→1-30, NPA→NA, NPA left, SMA-1→SMA-2 stays) → each line equals 1 except `opening == 5`, `closing == 5`, and the stay loan counted in neither in nor out; `left_book_last_bucket == {"NPA": 1}`; `checks_passed`.
  - `test_closing_equals_metric_change_numerator`: on a random 2,000-loan book (seeded `numpy.random.default_rng(7)`), `lines["closing"] == metric_change(...).summary["curr"]["numerator"]` and `lines["opening"] == ...["prev"]["numerator"]`; same for `base_lines` vs `eligible`.
  - `test_by_executive_uses_todays_owner`: a loan that moved RAJ→AMIT and rolled STD→SMA-1 counts in AMIT's row `in_from_std`; a 30+ loan last held by RAJ that left the book counts in the `"Left the running book"` row's `left_book`; every row closes.

- [ ] **Step 2: Run** `python -m pytest tests/investigation/test_flow_bridge.py -q` → FAIL.
- [ ] **Step 3: Implement** `flow_bridge` (vectorised boolean columns on `joined()`, one `groupby` for `by`).
- [ ] **Step 4: Run** → PASS; full suite → pass.
- [ ] **Step 5: Commit** `investigation/tools/ tests/investigation/test_flow_bridge.py` — "Investigation: 30+ flow bridge with cures, exits and new loans".

---

### Task 6: `mix_rate`

**Files:**
- Create: `investigation/tools/mix.py`; modify `investigation/tools/__init__.py`
- Test: `tests/investigation/test_mix_rate.py`

**Interfaces:**
- Consumes: Tasks 1 to 4 (`metric_change` for the total), `pair.labels`, config `INVESTIGATION_MIN_SEGMENT`, `INVESTIGATION_NOISE_SE`, `INVESTIGATION_RECONCILE_TOL_PP`, `INVESTIGATION_TOP_SEGMENTS`.
- Produces: `mix_rate(store: EvidenceStore, pair: PeriodPair, dimension: str, metric_id: str = "thirty_plus_rate", scope: dict | None = None, min_size: int | None = None) -> Evidence` with `tool == "mix_rate"`;
  `tables["segments"]` columns: `segment, n0, k0, n1, k1, s0, s1, r0, r1, rate_change_pp, mix_pp, rate_pp, new_pp, gone_pp, contribution_pp, noise_pp, outside_noise` (n = eligible, k = numerator, s = share 0..1, r = rate %, `None` where undefined), sorted by |contribution_pp| descending; segments with `max(n0, n1) < min_size` are summed into one row `"Smaller segments"` before any rate is computed;
  `summary = {"dimension": id, "label": str, "total_change_pp", "mix_pp", "rate_pp", "new_pp", "gone_pp", "top": [{"segment", "contribution_pp", "rate_change_pp", "mix_pp", "rate_pp", "outside_noise", "n1"}] (up to INVESTIGATION_TOP_SEGMENTS, outside-noise first)}`;
  checks: `"components_add_up"` (|mix + rate + new + gone − total| ≤ tol), `"shares_sum_to_one"` (each month, 1e-9), `"counts_match_totals"` (Σn, Σk equal `metric_change` eligible and numerator both months).
  Formulas (spec 7.4): both months `mix = (s1 − s0)·r0`, `rate = s1·(r1 − r0)`; current only `new = s1·r1`; previous only `gone = −s0·r0`; `noise_pp = NOISE_SE · 100 · sqrt(p0(1−p0)/n0 + p1(1−p1)/n1)` with p = r/100, only when both n > 0; `outside_noise = |rate_change_pp| > noise_pp`.
  Raises `ToolError` with `unavailable_reason` text when the dimension is unavailable.

- [ ] **Step 1: Write the failing tests:**
  - `test_pure_mix_case`: segment rates fixed (A 10%, B 50%), B's share 20%→40% → `rate_pp == 0`, `mix_pp == total_change_pp == 8.0`.
  - `test_pure_rate_case`: shares fixed 50/50, B's rate 50%→70% → `mix_pp == 0`, `rate_pp == total_change_pp == 10.0`.
  - `test_new_and_gone_segments_are_their_own_terms`: a segment only in curr and one only in prev → nonzero `new_pp` / `gone_pp`, their `r0` / `r1` are `None`, components add up.
  - `test_reconciles_on_random_books`: 20 seeded random books (`default_rng(seed)` for seed in range(20)), 3 to 12 segments, random new/gone segments → `checks_passed` every time.
  - `test_small_segments_are_grouped`: three segments of 30 loans with `min_size=100` → one `"Smaller segments"` row; totals unchanged.
  - `test_tiny_segment_swing_is_inside_the_noise_band`: 10-loan segment 10%→40% vs 5,000-loan segment 20%→21% with `min_size=1` → tiny `outside_noise is False`.
  - `test_unavailable_dimension_is_refused`: no FUEL_TYPE → `ToolError` containing `"no FUEL_TYPE column"`.

- [ ] **Step 2: Run** `python -m pytest tests/investigation/test_mix_rate.py -q` → FAIL.
- [ ] **Step 3: Implement** `mix_rate` (one `groupby` per month on `pair.labels(dimension, month)` over eligible rows; outer-align segments).
- [ ] **Step 4: Run** → PASS; full suite → pass.
- [ ] **Step 5: Commit** `investigation/tools/ tests/investigation/test_mix_rate.py` — "Investigation: mix versus rate decomposition that always reconciles".

---

### Task 7: `scan_dimensions`

**Files:**
- Modify: `investigation/tools/mix.py`, `investigation/tools/__init__.py`
- Test: `tests/investigation/test_scan_dimensions.py`

**Interfaces:**
- Consumes: `mix_rate`, `DIMENSIONS`, `unavailable_reason`, config `INVESTIGATION_NO_CHANGE_PP`.
- Produces: `scan_dimensions(store: EvidenceStore, pair: PeriodPair, metric_id: str = "thirty_plus_rate", scope: dict | None = None) -> Evidence` with `tool == "scan_dimensions"`; runs `mix_rate` (each its own Evidence in the store) for every `scan=True` dimension that is available;
  `summary = {"total_change_pp", "material": bool, "ranked": [{"dimension", "label", "evidence_id", "coverage_pct": float | None, "mix_pp", "rate_pp", "top": <mix_rate top>}], "unavailable": [{"dimension", "reason"}]}`;
  `coverage_pct = 100 · Σ(top outside-noise contribution_pp) / total_change_pp` (None when not material);
  `material = |total_change_pp| ≥ INVESTIGATION_NO_CHANGE_PP`;
  ranking: material → `coverage_pct` descending, ties by |top contribution|; not material → |largest outside-noise segment contribution| descending;
  checks: `"every_breakdown_reconciles"` (all child Evidence `checks_passed`).

- [ ] **Step 1: Write the failing tests:**
  - `test_planted_product_mix_ranks_product_first`: seeded 6,000-loan book where only the product mix shifts → `ranked[0]["dimension"] == "product"`, its top segment is the planted product, `coverage_pct` ≥ 60.
  - `test_unavailable_dimensions_are_listed_not_dropped`: book without FUEL_TYPE → `{"dimension": "fuel", "reason": ...}` in `unavailable`.
  - `test_flat_total_ranks_by_largest_segment_move`: two branches moving ±5 % points in equal halves → `material is False`, `coverage_pct is None` everywhere, `ranked[0]["dimension"] == "branch"`.
  - `test_each_breakdown_is_citable`: every `evidence_id` resolves in the store to a `mix_rate` result.

- [ ] **Step 2: Run** `python -m pytest tests/investigation/test_scan_dimensions.py -q` → FAIL.
- [ ] **Step 3: Implement** `scan_dimensions`. A shared book builder for Tasks 7 and 8 tests goes in `tests/investigation/books.py` (`make_book(n: int, seed: int, **plants) -> tuple[pd.DataFrame, pd.DataFrame]` returning prev, curr with every column the dimensions use).
- [ ] **Step 4: Run** → PASS; full suite → pass.
- [ ] **Step 5: Commit** `investigation/tools/ tests/investigation/` — "Investigation: scan every dimension and rank where the change sits".

---

### Task 8: `data_health`

**Files:**
- Create: `investigation/tools/health.py`; modify `investigation/tools/__init__.py`
- Test: `tests/investigation/test_data_health.py`

**Interfaces:**
- Consumes: `PeriodPair`, `get_metric`, config `INVESTIGATION_DPD_DISAGREE_WARN_PCT`, `INVESTIGATION_POP_CHANGE_HALT_PCT`, `INVESTIGATION_MISSING_UNIT_HALT_PCT`; `utils.count_agreed_after_month`; `df.attrs` keys `header_renames`, `missing_optional_cols`, `dropped_duplicate_loans` (set by `utils.load_and_validate`).
- Produces: `data_health(store: EvidenceStore, pair: PeriodPair, scope: dict | None = None) -> Evidence` with `tool == "data_health"`;
  `summary = {"status": "pass" | "warn" | "halt", "findings": [{"check": str, "status": "pass" | "info" | "warn" | "halt", "detail": str}]}`, status = worst finding (halt > warn > info/pass);
  findings, by `check` name: `months_in_order` (not consecutive → warn), `header_renames` (non-case renames either month → warn, naming them), `columns_one_month_only` (warn, naming them), `duplicates_dropped` (>0 → warn with counts), `units_new` (branches only in the current month → info), `units_missing` (branch only in the previous month: share of previous eligible > `MISSING_UNIT_HALT_PCT` → halt, else warn), `population_change` (eligible change beyond ±`POP_CHANGE_HALT_PCT` % → halt), `days_column_agreement` (share of eligible with non-blank `DelinquencyDays` where `days > 30` ≠ numerator, above `DPD_DISAGREE_WARN_PCT` → warn; column missing → info "not checked"), `executives_changed` (matched loans whose executive changed → info with count), `agreed_after_month` (>0 → warn with count per month);
  `checks = []` (findings, not invariants).

- [ ] **Step 1: Write the failing tests** (each builds a book with `tests/investigation/books.py` and plants one problem):
  - `test_clean_book_passes`: status `"pass"` or `"info"`-only → `summary["status"] == "pass"`.
  - `test_large_branch_missing_this_month_halts`: drop a branch holding 5% of last month's eligible from curr → status `"halt"`, finding `units_missing` names it.
  - `test_small_branch_missing_warns`: same with 0.5% → `"warn"`.
  - `test_population_jump_halts`: curr eligible +20% → `population_change` halt.
  - `test_days_disagreement_warns_above_threshold`: 3% of curr eligible with days ≤ 30 but SMA-1 → warn; 1% → no warn.
  - `test_header_renames_and_duplicates_are_reported`: attrs `header_renames={"CoLending_Loan s": "CoLending_Loans", "MNT Name": "MNT NAME"}`, `dropped_duplicate_loans=611` → warns mention `CoLending_Loan s` and `611`, not `MNT Name`.
  - `test_reassigned_loans_are_counted`: 7 loans change executive → `executives_changed` detail contains `"7"`.

- [ ] **Step 2: Run** `python -m pytest tests/investigation/test_data_health.py -q` → FAIL.
- [ ] **Step 3: Implement** `data_health`.
- [ ] **Step 4: Run** → PASS; full suite → pass.
- [ ] **Step 5: Commit** `investigation/tools/ tests/investigation/test_data_health.py` — "Investigation: data health checks that can stop an investigation".

---

### Task 9: Real-file acceptance (local, nothing committed but the script)

**Files:**
- Create: `investigation/real_check.py` (CLI: `python -m investigation.real_check <prev.xlsx> <curr.xlsx> <prev YYYY-MM> <curr YYYY-MM>`; loads both via `utils.load_and_validate.__wrapped__`, runs all five tools, prints counts, rates, bridge lines, the top 5 ranked dimensions with coverage, health findings, and every check result; prints no loan rows or customer fields)

**Interfaces:**
- Consumes: everything above.

- [ ] **Step 1: Run** `python -m investigation.real_check "C:/Users/91748/Downloads/Mumbai BU Aug closing RUN.xlsx" "C:/Users/91748/Downloads/Mumbai BU RUN Sept Closing.xlsx" 2026-08 2026-09`.
- [ ] **Step 2: Verify against spec section 3.1:** prev eligible 87,600 and numerator 24,151 (27.57%); curr 88,918 and 25,211 (28.35%); all-running 26.24% → 26.83%; bridge `closing == 25,211`, `opening == 24,151`; every Evidence `checks_passed`; health findings include `duplicates_dropped` (133 / 611), `units_new` (HINGL, WASHM), `executives_changed` (3,485), `header_renames` for Sept. Any mismatch is a bug to fix before Phase 1 is accepted.
- [ ] **Step 3: Run** `python -m pytest -q` and `python -m pyflakes investigation tests/investigation` → all pass, no new lint.
- [ ] **Step 4: Report** the printed summary to the user (counts and dimension names only) for review; record their sign-off.
- [ ] **Step 5: Commit** `investigation/real_check.py` — "Investigation: real file acceptance check for the phase one tools".
