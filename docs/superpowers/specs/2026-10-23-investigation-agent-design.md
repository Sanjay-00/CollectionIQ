# Deep Investigation: design

Status: draft for review, 2026-10-23. Nothing in this document is built yet,
except the header-drift fix in section 3.4, which ships in its own commit first.

## 1. Intent

A collections leader asks "why did the 30+ bucket worsen?". Today the
Investigator answers that with one generic roll-rate table (verified live on
2026-10-23: three phrasings all routed to `roll_rate_summary`, portfolio-wide,
with no 30+ figure, no bridge, no "where" and no "what kind").

Deep Investigation answers it the way a good analyst does, with one reusable
method instead of one workflow per question:

1. Rule out data problems first.
2. Measure the change exactly, and split it into "more loans went bad" vs
   "the base changed".
3. Find where it is concentrated, and whether that is a mix effect (more loans
   from riskier groups) or a rate effect (the same groups got worse).
4. Test the leading explanation against a rival before believing it.
5. Say what to do next, using lists the app already has.

Code computes every number. The language model chooses what to test next,
keeps a ledger of explanations, decides when the evidence is enough, and
writes the result. A verifier checks the write-up against the evidence before
anyone sees it.

## 2. Goals, non-goals, anti-goals

Goals
- Answer "why did metric M move between two months" for the 30+ rate, with
  every number traceable to a tested calculation.
- One method, many questions: adding a metric or a dimension is a definition,
  not a new workflow.
- Measurable quality: a scored evaluation set with planted causes, run before
  and after every prompt change.
- A deterministic baseline that is useful on its own, and that the agent must
  beat to ship.

Non-goals (v1)
- Metrics other than the 30+ rate (metric switching is v2 priority 1).
- SOH-weighted rates (v1 is account-based; SOH is shown alongside, not decomposed).
- SQL execution, MCP, vector stores, multi-agent setups.
- Collection agency or campaign analysis (no such data; the user confirmed it isn't needed).
- Explaining why a loan left the running book (the files cannot say).

Anti-goals (would be failure even if it "works")
- A narrative number that no tool produced, or that a failed check produced.
- A causal claim ("caused by", "due to") from descriptive data.
- A confident story when nothing material changed, or when the data is broken.
- A second definition of 30+ that disagrees with the Dashboard or Migration tab.
- Customer names or mobile numbers sent to the model.

## 3. Verified facts (real Mumbai BU files, Aug and Sept 2026 closing)

All counts are after the app's own upload pipeline (`utils.load_and_validate`).

### 3.1 Population
| | Aug | Sept |
|---|---|---|
| Running loans (all `Loan Status` = RUN) | 92,042 | 93,976 |
| No EMI due yet (bucket `NA`, blank Arrears / EMI) | 4,442 | 5,058 |
| Eligible (EMI due: STD, 1-30 DPD, SMA-1, SMA-2, NPA) | 87,600 | 88,918 |
| 30+ (SMA-1, SMA-2, NPA) | 24,151 | 25,211 |
| **30+ rate, eligible base (primary)** | **27.57%** | **28.35%** (+0.78 % points) |
| 30+ rate, all running loans (companion) | 26.24% | 26.83% (+0.59 % points) |
| Duplicate loan rows dropped at upload | 133 | 611 |

### 3.2 Movement between the two files
- In both months: 88,852 loans. Only in Aug (left the running book): 3,190,
  of which 2,416 were STD, 340 NPA, 196 1-30 DPD, 136 SMA-1, 94 SMA-2, 8 NA.
  Only in Sept: 5,124 (3,077 agreed in Sept; 3,205 of them have no EMI yet).
- Six Sept loans have an agreement date after Sept (day/month swaps in the
  source; the app already flags these).
- Matched loans whose region changed: 1; branch: 2; executive: 3,485.
- Branches new in Sept: HINGL, WASHM. None disappeared.
- The files hold running loans only, so a loan that left cannot be classed
  as closed, written off or repossessed.

### 3.3 Bucket rule and the days column
- The app has one bucket rule, `utils.bucket_from_arrears_emi`, on Arrears / EMI
  (EMIs overdue): blank → NA; ≤ 0 → STD; (0, 1) → 1-30 DPD; [1, 2) → SMA-1;
  [2, 3) → SMA-2; ≥ 3 → NPA. So **30+ means one or more full EMIs overdue**.
- The file also has `DelinquencyDays`. On Sept eligible loans the two rules
  disagree on 30+ for 985 loans (1.11%): 940 are SMA-1 by EMIs but 1-30 by
  days; 45 are 1-30 DPD by EMIs but over 30 days.
- `NPA Status` ("Loan DPD" / "Cust DPD", 13k rows) is not a sticky-NPA flag
  and is not used.

### 3.4 Header drift (fixed separately, before this work)
Sept's headers were `DelinquencyDay s` and `CoLending_Loan s`. The app did
not find either column, so on Sept: "Co-lending Loans at Risk" showed 0 (real
500), the co-lending priority category was missing (real 500), and
`DelinquencyDays` was unavailable. Book figures (accounts, Collection%, NPA%,
Delinquency%) were unaffected. The fix reads headers ignoring internal spaces,
records every rename in `df.attrs["header_renames"]`, and lists non-case
renames in the data checks panel. Aug outputs are unchanged.

### 3.5 Dimensions available (Sept eligible loans)
Scannable across the whole book: Zone (3), Region (16), Branch (146),
SegmentName (14), CHANNEL (6), scheme (42), Make (174, top 25 by size),
vintage by agreement quarter (26), Tenure (62, banded), Loan Amount (bands),
vehicle age from Year Of Manufacture (23, banded), Due Dt (4 EMI dates),
Customer Loan Count (fleet size), NACHStatus (2), FUEL_TYPE (7, 20% blank).
Drill-only (too fine for a book-wide scan): executive (794 values, 30 with
200+ loans), SRC Name / dealer (1,695). No usable spread: SaleType, BU,
CUSTOMER_STATUS.

### 3.6 Code facts
- Model: `config.GEMINI_MODEL = "gemini-2.5-flash-lite"` for every AI call.
- LangGraph 1.2.1 is used by AI Query only (`graph.py`), with a side channel
  that keeps DataFrames out of graph state, and LangSmith tracing. The
  Investigator is plain Python.
- The month join (`app._cached_filter`) is a left join: this month's loans get
  last month's bucket as `prev_bucket`. Loans that left are not in it.
- `analysis/roll_flow.py` (Migration tab) counts roll steps per this month's
  region / branch / executive.
- Tests: 1,341 collected (all unit or integration; no agent evaluations exist).

## 4. Decisions

Made by the user (2026-10-23):
1. 30+ uses the app's EMI bucket rule. The primary rate excludes `NA` loans
   from numerator and denominator; the all-running-loans rate is reported
   alongside to show the effect of new disbursals.
2. `DelinquencyDays` is a data-quality cross-check only, with a configurable
   disagreement threshold.
3. Reuse `unit_metrics`, `roll_flow` and the Root Cause engines; no competing
   definitions.
4. Every loan is attributed to this month's executive, who handles it from now
   and is accountable for it (user, 2026-10-23, revising an earlier "last
   month's owner" rule). Roll-ins, cures, scans and action lists all use it,
   matching the Migration tab and AI Query's loan movement questions. Data
   health still reports how many loans changed executive.
5. Loans absent next month are "left running book", with the last bucket as a
   hint only.
6. `scan_dimensions` tests every eligible dimension; minimum segment size and a
   noise band keep small volatile groups from ranking as drivers.
7. Hypothesis ledger with statuses open / supported / refuted / untestable: no data.
   A rival explanation must be tested before anything is presented as supported.
8. Build the deterministic baseline and the evaluation harness before the agent.
9. Deterministic verifier first, including a causal-wording check; no AI judge
   in v1; manual spot checks.
10. Coverage is reported within one breakdown only; never added across dimensions,
    never presented as proof.
11. Plain Python tools and state; LangGraph for orchestration. Pandas now, SQL
    later behind the same tool interface. Gemini first, provider replaceable.
12. Next steps point to the existing Action Lists, not to free text.

## 5. The metric

```
id: thirty_plus_rate        version: 1          grain: loan (Loan No)
eligible:   Loan Status == RUN and curr_bucket in {STD, 1-30 DPD, SMA-1, SMA-2, NPA}
numerator:  eligible and curr_bucket in {SMA-1, SMA-2, NPA}   (1+ EMIs overdue)
rate:       100 * numerator / eligible     (ratio of summed counts, never an average of %)
companion:  thirty_plus_rate_all_running = 100 * numerator / running loans
cross-check: DelinquencyDays > 30 vs numerator; warn above 2% of eligible (config)
on screen:  "30+ (SMA-1 and worse: one or more EMIs overdue)"
```

Changes are reported both in % points and as a relative change. Definitions
live in a small Python registry (`investigation/metrics.py`): each metric is a
dataclass naming tested functions, never a formula string to evaluate.

## 6. Architecture

```
investigation/                 new package; investigator/ (quick Q&A) is untouched
  metrics.py      metric registry (section 5)
  dimensions.py   approved dimensions: column, banding rule, min size, scan or drill-only
  panel.py        PeriodPair: both months' frames + one full outer join on Loan No
  evidence.py     EvidenceStore: result_id -> tool output + checks passed + LLM summary
  tools.py        the analytic tools (section 7), pure functions, pandas
  baseline.py     fixed-script investigation (section 8)
  agent.py        LangGraph loop (section 9)
  prompts.py      versioned prompt text and JSON schemas
  verifier.py     section 10
  report.py       report model -> Streamlit rendering and Excel download
  synthetic.py    two-month book generator with planted causes (section 12)
  trace.py        run traces (section 13)
tests/investigation/          unit + integration tests
tests/investigation/evals/    LLM evaluation cases (marker: llm_eval, not run by default)
```

UI: a "Deep investigation" mode in the Investigator tab, offered for "why did
X move" questions, run on the current sidebar scope. The quick Q&A stays as it is.

## 7. Tool contracts

Every tool takes a `PeriodPair` and a `scope` (a dict of dimension → value,
already validated against the data), returns a typed result with a `result_id`,
and runs its own invariant checks. A result that fails its checks is stored
with `checks_passed = False` and can never back a claim. Tools raise a
`ToolError` with a plain message for can't-answer cases; they never return an
empty table that reads as zero.

### 7.1 `data_health(pair, scope)`
Checks, each pass / warn / halt:
- both months loaded, same metric definition version
- header renames, columns missing in one month only (from `header_renames`,
  `missing_optional_cols`)
- duplicate loan rows dropped per month
- units present in one month only (a branch missing in the current month with
  > 1% of last month's eligible loans halts; a branch only in the current month
  whose loans were agreed by last month's close is "missing from last month's
  file" and halts on the same 1% rule, else a genuinely new branch is
  information. Real Sept 2026: HINGL, WASHM, 1.8%)
- labels changed for the same loans (added in Phase 1): a dimension whose label
  differs between the files for > 5% of the loans in both months warns and is
  left out of the scan (real Sept 2026: CHANNEL 31% held a file name, FUEL_TYPE
  10%, Zone 9.8%)
- eligible population change beyond ±15% halts (config)
- `DelinquencyDays` disagreement above threshold warns
- executive reassignments (count) and agreement dates after the month (count) warn
- snapshot sanity: the two months are consecutive closings (from the reporting
  months the user picked)
Halting stops the investigation and reports the data problem as the answer.

### 7.2 `metric_change(metric_id, pair, scope)`
Numerator, denominator and rate for both months; change in % points and
relative %; the companion all-running rate. Split of the change:
`numerator effect = (N1 - N0) / D1`, `base effect = N0 * (1/D1 - 1/D0)`;
the two add to the change exactly (checked).

### 7.3 `flow_bridge(metric_id, pair, scope, owner="previous")`
On the full outer join of both months:
```
last month's 30+ loans
+ rolled in from STD                (matched, STD -> 30+)
+ rolled in from 1-30 DPD           (matched, 1-30 -> 30+)
+ entered from no-EMI-yet           (matched, NA -> 30+)
+ new to the running book already 30+
- cured to STD                      (matched, 30+ -> STD)
- cured to 1-30 DPD                 (matched, 30+ -> 1-30)
- became no-EMI (NA)                (matched, 30+ -> NA)
- left the running book             (last bucket shown as a hint)
= this month's 30+ loans            (checked exactly)
```
The same bridge for the eligible base. Optional `by` dimension, with each
loan under this month's labels (decision 4). Loans that left the running book
have no current owner and are counted under last month's labels, marked as
such. Reuses the bucket rule and
`VALID_BUCKETS`; adds the outer join that `roll_flow` lacks.

### 7.4 `mix_rate(metric_id, dimension, pair, scope)`
Segments are the dimension's values in each month (banded as in
`dimensions.py`; segments under the minimum size are grouped as
"Smaller segments", so the split stays exhaustive and totals unchanged).
For segments present in both months:
```
mix_i  = (s1_i - s0_i) * r0_i
rate_i = s1_i * (r1_i - r0_i)
```
Segments only in the current month add `s1_i * r1_i` to a "new segments"
term; segments only in the previous month add `-s0_i * r0_i` to "gone
segments". `mix + rate + new + gone` must equal the total change to within
0.01 % points (checked). An undefined rate (zero eligible) is never treated as zero.
Each segment also gets two noise bands, both paired because most loans are
the same loans seen twice (Phase 1 review: independent-sample bands read a
planted 2.5-point rise in 4,000 loans as noise). Rate:
`2 * sqrt((b + c)/(n0*n1) + u0*p0(1-p0)/n0^2 + u1*p1(1-p1)/n1^2)`, where b, c
are loans in the segment both months that left or entered the numerator and
u0, u1 the loans in it one month only. Share:
`2 * sqrt(s0(1-s0)*U0/N0^2 + s1(1-s1)*U1/N1^2)`, U the loans not in the same
segment both months. A segment is notable when either move is beyond its
band: a pure mix move has no rate change at all. "Other" and "Smaller
segments" are never cited.

Scoping and `by` give each loan one label for both months (this month's, or
last month's for a loan that left), so a loan that moved branch is never read
as leaving one and joining another; a dimension with a missing column or
labels changed beyond 5% is refused in plain words.

### 7.5 `scan_dimensions(metric_id, pair, scope)`
Runs `mix_rate` on every scannable dimension. Per dimension it reports the
mix / rate split, the top two notable segments' contribution, coverage (their
contribution as a share of the total change, within this breakdown only) and
concentration, `100 * sum max(0, |contribution_i - s1_i * total| - band_i) / (2 * |total|)`
with `band_i = s1_i * rate band + share band * r_i / 100`: how far the change
sits from an even spread by segment size, beyond noise (without the band, 146
noisy branches outranked a planted cause in review). Ranked by
concentration (Phase 1 testing showed coverage can't rank: a dimension with a
few huge segments "covers" any change). When the total isn't material, ranked
by the largest notable segment move. Deterministic; it is what stops the
model guessing where to look.

### 7.6 `controlled_decompose(metric_id, dimension, control, pair, scope)`
`mix_rate(dimension)` inside each value of `control` (for example vintage
within product). Reports in how many control groups, covering what share of
loans, the dimension's rate effect keeps its sign and stays outside the noise
band. This is the rival-explanation test.

### 7.7 `drill(metric_id, dimension, segment, pair, scope)`
`scan_dimensions` with the scope narrowed to one segment, including drill-only
dimensions (executive, dealer).

### 7.8 `sample_loans(evidence_id, n <= 50)`
The loans behind one evidence item (for example "rolled in from 1-30, branch
SHLPR"), largest SOH first, for the screen and the download. The model only
ever sees the count and the evidence id, never the rows.

## 8. Deterministic baseline (built and scored before the agent)

```
data_health -> metric_change -> flow_bridge -> scan_dimensions
-> drill(top dimension, top segment)
-> controlled_decompose(top dimension, second-ranked dimension)
-> report from templates (no model call)
```
It is a complete, shippable answer on its own. The agent ships only if it
beats this on the evaluation set (section 12.4).

## 9. The agent

### 9.1 State (plain Python dataclasses; LangGraph carries ids only)
```
question, metric_id, scope, pair_ref
evidence: dict[result_id, EvidenceSummary]   (compact: numbers, labels, checks_passed)
ledger: list[Hypothesis{id, statement, kind (mix|rate|flow|base|data),
                        status (open|supported|refuted|untestable), evidence_ids,
                        rival_tested_by: result_id | None}]
calls_used, llm_calls_used, started_at
status: running | halted_data | no_material_change | concluded | budget_exhausted | failed
```
DataFrames live in the `EvidenceStore` side channel, the same pattern
`graph.py` already uses, so traces stay small.

### 9.2 Graph
```
resolve      (model: question -> metric, scope; validated against registries)
health       (tool)            --halt--> report_data_problem
quantify     (metric_change + flow_bridge)
             --no material change--> report_no_change
scan         (scan_dimensions)
plan         (model: update ledger, pick ONE next action from the allowed set)
act          (tool) -> plan          [loop]
gate         (code: may we conclude?) --no--> plan
write        (model: structured report JSON citing evidence ids)
verify       (code)  --fail once--> write (with the failure list)
             --fail twice--> baseline report + "the AI write-up failed checks"
render
```
Allowed actions in `plan`: `mix_rate`, `drill`, `controlled_decompose`,
`flow_bridge(by=...)`, `sample_loans`, `mark_untestable`, `conclude`.
Arguments are a JSON object checked against each tool's schema; an invalid
dimension or segment is rejected in code and counted as a call.

### 9.3 Rules the code enforces (not the prompt)
- `health` always runs first; `quantify` and `scan` always run.
- No material change: the total change is inside the overall noise band, or
  under 0.20 % points (config). The answer is "no material change", plus the
  bridge.
- `conclude` is accepted only when the leading hypothesis has a rival test
  (`controlled_decompose`, or a competing dimension's `mix_rate`) and, within
  its own breakdown, the cited segments cover at least 60% of the change
  (config), or the budget is spent (then the status says so).
- Budget: 10 analytical calls after the scan, 15 model calls, 2 retries per
  model call, 120 seconds (all config). Measured in the evals, then tuned.

### 9.4 Model use
- A provider interface, `generate_json(prompt, schema) -> dict`, with the Gemini
  implementation reusing the existing client setup. A different provider is a
  second implementation.
- `INVESTIGATION_MODEL` in config, default `GEMINI_MODEL`. The evals compare
  flash-lite with a stronger Gemini model before the default is settled.
- The prompt holds the method (section 1), the tool catalogue with schemas, the
  ledger rules, two worked examples from synthetic data, and the output
  schema. Prompts are versioned in `prompts.py`; each trace records the version.
- The model sees evidence summaries only: rounded numbers, segment labels
  (branch, region, product, executive names), evidence ids. Never customer
  names, mobile numbers or loan rows.

### 9.5 Addendum (2026-10-24): AI Query as a tool, investigation order, budget guards

Verified in code (not assumed):
- `compiler.core.compile_logical(ir, columns, as_of) -> (plan, errors)` and
  `agents.plan_executor.execute_plan(df, plan, df_prev) -> (DataFrame, error)`
  are pure pandas: the compiler imports only `registry`, `agents.plan_executor`
  and `utils`; no model call anywhere on that path.
- The model calls in AI Query are the planner (`logical_planner_node` ->
  `plan_logical`, also re-called for the one compile repair), the clarify route,
  and the insight writer (`analyze_node`). The view route (`view_node`) runs
  registered analysis functions but takes its view name from the planner.
- `execute_node` reshapes results for the screen (merged "MNT NAME (Unit)",
  default column trim, `compute_result_kpis`); the raw `execute_plan` output is
  a DataFrame whose columns are the IR's aliases: stable enough once wrapped.
- AI Query has no 30+ metric, and its `count_ratio` metrics count every row in
  the denominator, including loans with no EMI due yet. A "30+ %" written in IR
  today would NOT equal this design's rate.

Decisions:
1. `run_query(ir1)` is an exploratory tool: the agent writes IR-1 JSON; the
   tool runs `compile_logical` + `execute_plan` on the like-for-like pair (this
   month's frame with last month's bucket carried by the same loan join), never
   the planner, the clarify step or the insight writer. Output: result_id,
   columns, row count, the first 10 rows with customer fields removed, column
   totals. A compile error comes back as a structured error; the agent gets one
   repair, counted against the budget.
2. Typed tools own every 30+ figure. A `run_query` result can give context
   (counts, lists, cross-tabs, columns outside the approved dimensions) but can
   never back a driver claim; the verifier rejects a driver citing it.
3. One 30+ definition: add `thirty_plus_pct` to `registry/ontology.py`
   (numerator Arrears / EMI >= 1, denominator the eligible buckets) with a
   consistency test against `investigation/metrics.py`, the same pattern as
   `tests/test_metric_consistency.py`. Only then may `run_query` compute 30+ rates.
4. Order is fixed, the model reasons inside it: prepare (health, like for like)
   -> metric_change with materiality -> flow_bridge (what kind of problem: more
   roll-ins, fewer cures, base) -> scan_dimensions -> model loop -> gate ->
   write -> verify. The loop drills whichever dimension the scan ranks highest,
   not geography first. A rise spread evenly over every segment is reported as
   a portfolio-wide finding, not dismissed. Links between findings are tested
   with controlled_decompose, never by adding overlapping breakdowns.
5. Budget guards in code, checked before every model turn: tool calls (10 after
   the scan), model calls (15), tokens (60,000), wall time (120 s), drill depth
   (3), one repair per compile error; at 80% of any cap the run is forced to
   write up what it has and list open hypotheses; every tool call is hashed
   (tool + normalised arguments + data version) and a repeat returns the cached
   result but still spends a step; two steps in a row with no new evidence
   (no ledger status change, no new segment beyond its share) stop the loop.
   Agent mode never asks the user: an ambiguous term becomes untestable or a
   logged default.
6. Not adopted: LangGraph checkpoints (a run is under two minutes in one
   session; nothing to resume), sharing Streamlit caches with the agent
   (different lifetimes; the agent keeps its own per-run cache keyed by data
   version), and "database query cost" limits (no database: pandas in memory).
7. Evals gain: a loop that would not end must end; a repeated call must hit the
   cache; a no-evidence run must stop on the progress rule; a `run_query`
   compile error's repair is counted; every score records tool calls, model
   calls, tokens and seconds.

Phasing: none of this changes Phase 3 (baseline: controlled_decompose, drill,
sample_loans, fixed script, scored). `thirty_plus_pct` and its consistency test
land in Phase 3 too (cheap, and the baseline can then cross-check). `run_query`,
the guards and the new evals land in Phase 4 with the agent.

## 10. Verifier (code)

1. Every number in the write-up matches a value in an evidence item it cites,
   within display rounding, and that item passed its own checks.
2. Direction words (rose, fell, up, down, worsened, improved, increase,
   decrease) agree with the sign of the cited value.
3. Months, metric name and version agree with the run.
4. Causal wording ("caused by", "due to", "driven by", "because of", "led to",
   "resulted in") is rejected: v1 has no evidence type that supports a causal
   conclusion, so the report uses "accounts for", "concentrated in",
   "associated with".
5. Every data warning, untestable hypothesis and refuted rival appears in the report.
6. No customer identifiers in the text.

## 11. The answer on screen

1. Headline (built in code): the change, the base effect, where it is concentrated.
2. What changed: the rate both ways, then the bridge table.
3. Where: the top breakdown with mix / rate split and the cited segments.
4. Checks: the rival test(s) and their result.
5. Not tested: hypotheses marked untestable, with the missing data.
6. Data notes: warnings from `data_health`.
7. What to do next: links to existing Action Lists (for example "New
   defaulters", "1-30 → SMA-1+") filtered to the cited segment, plus the sample
   loans table and an Excel download of every evidence table.
8. How I got here: the steps taken, from the trace.

Wording follows the app's rules: "delinquent", "MoM", "% points", no dashes on
screen, no verdict words such as "worse" or "better" for a unit's figures.

## 12. Evaluation

### 12.1 Synthetic data
`investigation/synthetic.py` builds a two-month book (about 20,000 loans) with
the real file's column names and realistic bucket shares, then plants one
scenario per case. Generated in tests, never from customer data.

### 12.2 Cases (expected result written before any run)
| # | Planted | Expected |
|---|---|---|
| 1 | One vintage's roll into 30+ doubles | rate effect, that vintage |
| 2 | A risky product's share triples | mix effect, that product |
| 3 | A large branch missing in the current month | halt, data problem |
| 4 | Headers drift (spaced names) | read correctly, rename noted |
| 5 | Duplicate loan rows in one month | warn, counts correct |
| 6 | No material change | "no material change", no story |
| 7 | The planted cause is in a column removed from the file | untestable reported, no guess |
| 8 | Roll-ins up while cures fall | flow mechanism, both parts |
| 9 | Offsetting segment changes under a flat total | no material total; offsets shown |
| 10 | A vintage effect that is really a product effect | product, after the rival test |
| 11 | Two simultaneous causes | both found |
| 12 | A tiny segment with a huge % swing | ignored (noise band) |
| 13 | A cause two levels deep (region → branch → executive) | the executive group found |
| 14 | Base-only change: disbursals stop, NA loans fall | base effect named |
| 15 | Many loans reassigned between executives | reassignment warned; flows by last month's owner |
| 16 | Days column disagrees with buckets above threshold | warning shown |

Phase 2 changes to the table (2026-10-24, review): case 3 now expects the
missing branch left out like for like and the cause found; case 12 is "a small
segment's real but minor swing beside the main cause"; cases 18 (eligible
loans jump by a quarter) and 19 (six of ten branches missing) expect a halt;
case 17 is the real Sept 2026 fault (a branch missing from last month's file).
The cases live in `investigation/eval_cases.py`, the plants in
`investigation/synthetic.py`.

Rules the cases rely on (written down after review):
- A driver is a segment beyond its fair share of the change and noise
  (`beyond_share`), pushing the same way as a material total, and accounting
  for at least `INVESTIGATION_DRIVER_MIN_SHARE_PCT` (10%) of it. It may be
  named at any depth (region, then branch, then executive).
- When no testable slice concentrates a material change, the answer names no
  driver; "concluded" with the flow mechanism and "inconclusive" are both
  right (one month's bridge can't tell fewer cures from more roll-ins).
- A case with nothing to explain fails any named driver.
- A cited result must be the breakdown by that dimension that shows the segment,
  and must have passed its own checks.

### 12.3 Scoring (automated)
Cause found (top dimension and segment, or the right halt / no-change
status); numbers exact; unsupported claims (verifier failures that reached the
final text: must be 0); required checks done; tool calls; model calls; tokens;
seconds.

### 12.4 Protocol
- Unit tests (tool maths: reconciliation on random books, edge cases) and
  integration tests (baseline on every case) run with the normal suite.
- LLM evals: each case 5 runs; cases 4, 9, 12, 15, 16 held out from prompt
  tuning; pass rate per case reported separately from the unit test count.
- Ship gate: the agent matches the baseline on every case it passes, beats it
  on 10, 11 and 13, has zero unsupported claims, and has median cost under the
  budget. Otherwise the baseline ships and the gap is documented.
- Real-data golden case (local only, never committed): Aug → Sept. The baseline
  output is reviewed by the user once; that reviewed answer becomes the
  expected result.

## 13. Traces and cost

One JSON trace per run in a git-ignored `runs/` folder: question, metric,
scope, prompt version, model, each tool call (arguments, result id, checks,
milliseconds), each model call (tokens, milliseconds, retries), termination
status. No loan rows or customer fields. LangSmith tracing is reused when it
is configured.

## 14. Failure handling

| Failure | Behaviour |
|---|---|
| Only one month loaded | Explain that two closings are needed; nothing runs |
| `data_health` halt | The data problem is the answer, with what to fix |
| Tool error (bad dimension, empty scope) | Counted as a call; the model sees the message; three in a row ends the run |
| Model returns invalid JSON | Retry twice, then fall back to the baseline report |
| Budget spent | Report what is supported so far; status "budget reached" |
| Verifier fails twice | Baseline report, marked "the AI write-up failed checks" |
| No material change | Say so with the bridge; no driver story |

## 15. Phases and acceptance

| Phase | Builds | Accepted when |
|---|---|---|
| 1 | metrics, dimensions, panel, evidence store; tools 7.1 to 7.5 | Aug → Sept: rate and bridge reconcile exactly; 30+ counts match the Dashboard's buckets; mix/rate reconciles on every dimension; unit tests pass |
| 2 | synthetic generator, cases 1 to 16, scoring harness | each case's planted truth is recoverable by direct calculation |
| 3 | tools 7.6 to 7.8, baseline, report rendering | baseline scored on all cases; the user reviews the real Aug → Sept answer |
| 4 | agent, verifier, traces | scored against the baseline; ship gate decided |
| 5 | Investigator tab mode, Excel download | renders on the real files, no format problems, existing tabs unchanged |
| 6 (optional) | SQL engine behind the same tools | every tool's SQL result equals the pandas result on the eval data |

## 16. v2 backlog
1. Metric switching inside a run (30+ rate → cure rate → executive productivity).
2. SOH-weighted decomposition.
3. Other metrics (NPA%, Collection%) as registry entries.
4. An AI judge for faithfulness, if spot checks show a need.
5. SQL / warehouse path.

## 17. Open questions for the user
1. Resolved: this month's executive everywhere (decision 4).
2. Resolved (user, 2026-10-23): every executive "last month" figure in the
   app is the loans they hold now (`utils.prev_held_now`), done before this
   design is built. Region and branch figures keep last month's file. On the
   real files this changed the last-month NPA for 328 of 747 executives.

## 18. Calls made by the author (please check)
- New `investigation/` package beside `investigator/`, so the quick Q&A stays untouched.
- Deep Investigation is a mode inside the Investigator tab, not a tenth tab.
- Causal verbs are rejected outright in v1 (section 10.4).
- Thresholds: minimum segment 200 eligible loans for book-wide scans (50 inside a
  drill), noise band of 2 standard errors, no-material-change under 0.20 % points,
  conclude at 60% coverage within a breakdown, days disagreement warning at 2%,
  population change halt at ±15%. All in config and tuned with the evals.
- Vintage by agreement quarter, the latest two quarters by month.
- Loan Amount banded on last month's quintile edges so both months share bands.
- Runs on the current sidebar scope.
