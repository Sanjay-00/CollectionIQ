# Portfolio Review Agent: design

Status: draft, 2026-10-24. Extends `2026-10-23-investigation-agent-design.md`
(Deep Investigation): that design answers "why did metric M move"; this one
answers "how are my regions performing, and where should I focus first", then
hands the chosen focus to Deep Investigation. Phase 1 (section 6) is being
built; every later phase waits for review.

## 1. Intent

A leader asks "how are my regions performing". A good analyst:

1. Builds one scorecard: business, collection, risk, change, for every region.
2. Compares each region like for like, so a region with a harder book is not
   ranked as a worse performer.
3. Names the few regions that genuinely stand out, in separate lists (cash
   this month, book risk, getting worse), with rupees at stake and noise checked.
4. Explains the trade-off between them and recommends a first move, or says
   plainly that two priorities are different kinds of problem.
5. Investigates the chosen one when the leader agrees, and proposes actions
   from the lists the app already has.

Code computes every number and every candidate. The model recommends,
explains and chooses the next investigation step among options code allows.

## 2. Review of the two external critiques

Both critiques were checked against the code and data (2026-10-24), not taken
on trust. Accepted, changed or rejected:

| # | Point | Verdict | Why |
|---|---|---|---|
| G1 | Benchmark = rest of portfolio, region excluded | Accept | Including the region biases its own benchmark toward it, worst for big regions |
| G1 | Collection gap = rate gap x demand | **Change** | Verified: `Net Collection Demand Inst+Exp+BC` is this month's EMI demand only (matches Month Due Inst+Exp+BC on 99.9% of loans), while the collection side includes arrears recovered. A region with more overdue loans collects arrears on top of EMI, so raw Collection% depends on book mix. Fixed by standardising (C1 below) |
| G1 | Excess risk = (NPA% gap) x SOH | **Reject as written** | Mixes units: an account-based rate times rupees is not a rupee figure of anything. Replaced by two consistent figures: excess accounts (account rate x accounts) and excess SOH (SOH-weighted rate x SOH) |
| G1 | No single weighted score; separate lists | Accept | Weights would be invented. Leaders still get a first move from C7 (overlap) and the model's explained recommendation |
| G2 | Evidence-backed candidate schema | Accept | Section 5.4 |
| G3 | Approval semantics, idempotent side effects, decision log | Accept | Section 7.3 |
| G4 | Compact state, tool-call log, findings as hypotheses | Accept | Section 7.4 |
| G5 | Acceptance criteria before code | Accept | Section 8 |
| C1 | Like-for-like benchmark (standardise within strata) | **Accept, with a correction** | The critique stratifies collection by bucket but doesn't say which. The closing bucket is partly an outcome of this month's collection (a loan that didn't pay moves to a worse bucket), so stratifying by it is circular. Strata are the **opening** bucket: last month's closing bucket (`prev_bucket`), with loans new this month as their own stratum. For 30+ (a stock), strata are agreement vintage |
| C2 | Multiple testing; amount-weighted noise | Accept, method changed | Benjamini-Hochberg within each list plus rupee and point floors. Noise for amount-weighted rates uses the delta-method variance of a ratio estimator, not a bootstrap: deterministic (a seeded bootstrap still changes when the data changes slightly), and exact enough at these sizes |
| C3 | Start saving history now | Accept, one correction | "Can't be backfilled" is true only if old monthly files are gone. If past LCC files exist, a one-off backfill builds a year of history in minutes. **Open question 1** |
| C4 | Close the loop on past recommendations | Accept for Phase 3 | Needs persistence (section 7.5) |
| C5 | Brief on upload, no LLM | Accept for Phase 3 | Phase 1's code is exactly that brief; the UI slot is the Investigator opener ("What moved most") |
| C6 | Recoverability from roll_flow cure rates | **Reject for now** | roll_flow has one transition (two files), not history. One month's cure rate per bucket is too noisy to weight decisions. Revisit once history holds 6+ months |
| C7 | Flag regions on 2+ lists | Accept | Deterministic, no weights |
| C8 | `interrupt()` unnecessary: a pause is the end of a run segment | **Accept** | Every checkpoint sits at a segment boundary. Section 7.1 |
| C9 | SQLite won't survive Streamlit Community Cloud | Accept, and it shapes the design | Verified: the app is deployed on Streamlit Cloud (project_guide Phase 8) and its stated design is "no database, data stays local". Section 7.5 |
| C10 | Ask about targets | Accept | **Open question 2**. Benchmark is rest-of-portfolio until targets exist |

## 3. Constraints (what the data and platform allow)

1. **Two months only.** Every comparison is this month vs last. No seasonality,
   no "normal for the month", no multi-month cure rates until history exists.
2. **Account vs amount bases differ.** Dashboard NPA% is NPA loans / all
   accounts in the file (`unit_metrics`). The investigation's 30+ rate uses
   running loans with an EMI due. Candidates use the investigation's eligible
   base; the scorecard shows the Dashboard's own figures so it reconciles.
   Both are labelled; they are never mixed in one formula.
3. **Collection% is amount-weighted and can exceed 100%** for a loan or a
   stratum (arrears recovered on top of the month's EMI). Fine inside a stratum.
4. **Opening bucket needs last month's file.** The review needs both months
   (as Deep Investigation does); loans new this month are their own stratum.
8. **The Dashboard's Collection% counts cash from loans with no EMI due this
   month** (e.g. matured loans clearing arrears) against zero demand. Found in
   Phase 1: on the dummy file a region reads 139.99% on the Dashboard and
   89.77% on loans that had demand; 36% of its collection had no demand
   behind it. The shortfall list uses loans with demand only, reports the rest
   separately, and its evidence reconciles both to the scorecard. Whether the
   Dashboard should keep this definition is **open question 6**.
5. **No persistent disk on Streamlit Cloud**, and no database by design.
6. **Gemini 2.5 Flash-Lite** is the model; it is weakest at multi-step planning.
7. **No causal claims** from descriptive data (inherited anti-goal).

## 4. What is not possible, and is flagged instead of assumed

- **Recoverable cash.** The collection gap is a benchmark-relative estimate,
  not money that can be collected. Stated on screen.
- **Avoidable losses.** Excess exposure is benchmark-relative SOH, not loss.
- **Normal seasonality** (monsoon, festival, quarter end): unknowable with two
  months. The brief says so when asked.
- **Region targets**: not in the data. Not used until supplied.
- **Business (new advances) as a focus candidate**: new advances per region
  are shown in the scorecard, but "too little new business" is a growth
  question with no noise model yet (counts of new loans by month). Phase 1
  shows it; a business list is deferred (open question 3).

## 5. Phase 1: deterministic scorecard and prioritisation

Package: `investigation/review/` (`scorecard.py`, `prioritise.py`). Pure
pandas, no model call. Grain: region (branch and zone use the same code).

### 5.1 Scorecard
One row per region from the existing engines, never re-derived:
- `utils.unit_metrics` (this month and last): Accounts, Collection%, Demand,
  Collected, Delinquency%, NPA%, Strike%, SOH (Cr), and MoM changes in % points.
- `investigation.metrics` 30+ rate on the eligible base, both months.
- `analysis.new_business.compute_new_advances_by_dimension`: new accounts and
  funded amount this month, with MoM.
- Share of book (SOH). Rows sorted by name; ranking lives in the lists.
Reconciliation: every Dashboard column equals `compute_region_scorecard`'s.

### 5.2 Notation
Region g, rest of portfolio r = every loan not in g. For a loan i: demand d_i
(`Net Collection Demand Inst+Exp+BC`), collected c_i (`Month Collection
(Excluding Reserve Collection)`), SOH h_i, 30+ indicator y_i (investigation
metric), stratum s(i). Sums over a set are written D, C, H, Y, N (count).

### 5.3 The three lists

**L1. Collection shortfall (this month, rupees).** Strata: opening bucket
s(i) = last month's closing bucket for loans on last month's book; "New this
month" otherwise.
```
rest rate per stratum      R_rs = C_rs / D_rs
expected collection        E_g  = sum_s D_gs * R_rs
estimated shortfall (Rs)   G_g  = E_g - C_g                 (> 0: collected less than a like-for-like book)
standardised gap (pts)     100 * G_g / D_g
raw gap (Rs)               D_g * (C_r / D_r) - C_g          (shown, for transparency)
book-mix part (Rs)         raw gap - G_g                     ("this much of the apparent gap is book mix")
```
A stratum where the rest of the portfolio has no demand uses the rest's overall
rate for that stratum, and the candidate carries a caveat. Without last
month's file every loan is one stratum (raw = standardised), with a caveat.

Noise (delta method, ratio estimator; loans independent):
```
Var(G_g) = sum_{i in g} (c_i - R_{r,s(i)} d_i)^2  +  sum_s D_gs^2 * Var(R_rs)
Var(R_rs) = sum_{i in r,s} (c_i - R_rs d_i)^2 / D_rs^2
band = INVESTIGATION_NOISE_SE * sqrt(Var(G_g))
```

**L2. Excess 30+ exposure (book risk).** Eligible loans only. Strata: agreement
vintage (the investigation's `vintage` dimension: quarter, recent months by
month), so seasoning is controlled.
```
expected 30+ loans          X_g = sum_s N_gs * (Y_rs / N_rs)
excess 30+ loans            A_g = Y_g - X_g
excess 30+ SOH (Rs)         S_g = H30_g - sum_s H_gs * (H30_rs / H_rs)    (SOH-weighted rate x SOH: one unit)
```
Ranked by S_g; A_g shown alongside. Noise (binomial on the account version):
```
Var(A_g) = sum_s [ N_gs p_gs (1 - p_gs) + N_gs^2 q_rs (1 - q_rs) / N_rs ]
p = region rate in stratum, q = rest rate in stratum
```
The same machinery with numerator "NPA" (eligible base) gives an NPA view on
request; the default list is 30+ because it is the earlier, more actionable
signal and matches Deep Investigation.

**L3. Getting worse (momentum).** Per region, `metric_change` on the 30+ rate
(the existing tool, paired noise band, materiality floor). A region qualifies
when its own change is material and worse than the rest of the portfolio's
change by more than the larger of the two bands. Reported in % points and
accounts. Shares Deep Investigation's definitions exactly.

### 5.4 A candidate
```
id, list (collection_shortfall | excess_30plus | worsening), region,
metric (definition label + eligible base), period (curr, prev),
value, benchmark, gap_pts, impact {amount, unit, formula}, raw_impact, mix_part,
numerator, denominator, eligible accounts,
band, z, q_value (BH), material (bool), reasons_not_material,
evidence_id, data_version, caveats[]
```
`data_version` is a hash of both months' key columns. Every candidate is an
Evidence record with its strata table and self-checks:
- strata sums equal the region's totals;
- raw = standardised + book mix (to the rupee);
- the region's figures equal the scorecard's.

### 5.5 Qualifying, ranking, overlap
A candidate is listed when all hold (all thresholds in config):
- eligible accounts >= `REVIEW_MIN_ACCOUNTS` (200); smaller regions are shown
  in the scorecard and listed as "too small to rank";
- beyond its noise band, and Benjamini-Hochberg q <= `REVIEW_FDR` (0.10)
  across **every test of the review together** (16 regions x 3 lists is 48
  tests; 95% bands alone would flag about 2 by chance). Built per list first,
  the pure-noise eval showed about a quarter of reviews listing something;
  one family is what a leader reads, so one correction;
- at most `REVIEW_MAX_UNMATCHED_PCT` (20%) of its base in strata no other
  region has. Beyond that there is no like-for-like benchmark (the seasoning
  eval: a region holding the only 2019-2022 agreements was compared with a
  younger book and wrongly listed), so it is shown with that reason;
- impact >= `REVIEW_MIN_RS` (Rs 10 lakh, lists L1 and L2) and gap >=
  `REVIEW_MIN_PTS` (0.5 pts, all lists).
Each list ranks by impact, ties by region name, so the output is identical on
every run. A region on two or more lists is flagged `on_lists = [...]`.
When nothing qualifies anywhere, the brief says no region stands out beyond
noise, which is an answer, not a failure.

## 6. Phase 1 scope (being built)

In: 5.1-5.5, region grain, tests, evaluation cases. Out: model calls, UI,
persistence, investigation loop. Phase 1 evals (synthetic books, no customer data):
1. Bucket-mix trap: a region whose raw Collection% is lowest only because its
   book has more overdue loans; standardised it is average. Must not be listed.
2. Real shortfall: a region collecting 10% less in every stratum. Listed first.
3. Seasoning trap: a region with older vintages (higher 30+ naturally). Not on L2.
4. Small region with an extreme rate: shown, never ranked.
5. Pure noise book (every region same odds): no candidate on any list.
6. Two-list region: listed on L1 and L3, flagged.
7. Stability: the same input twice gives identical candidates; reordering rows
   gives identical candidates.

## 7. Later phases (design only; each needs review)

### 7.1 Run segments, not mid-node pauses
A run is a bounded segment that ends in a state: `awaiting_choice`,
`awaiting_action_approval`, `done`, `halted`. A human decision starts the next
segment with the saved state plus the decision. No `interrupt()`: no node
re-execution, no index-matched resumes, works the same in tests (call the
segment function with a scripted decision) and in Streamlit (a button calls it).
LangGraph stays for the investigation loop inside a segment, as already planned.

Segments: `review` (Phase 1 code + model recommendation) -> choice ->
`investigate` (Deep Investigation, scoped to the chosen candidate) ->
approval -> `act` (action list, email draft) -> done.

### 7.2 What the model does
- `recommend`: reads candidates (summaries only), the decision log, business
  notes; outputs `{recommended_id | null, tradeoff_text, next_step}`. It may
  decline to pick when two candidates are different kinds of problem. It
  cannot add a candidate or a number; the verifier rejects any number not in
  the candidates and any causal wording.
- `redirect`: turns free text ("ignore NPA, legal backlog") into a scope or
  list change, validated against the registry, or a clarification.

### 7.3 Approval and side effects
- Choice: approve recommended / pick another / redirect / stop.
- Action approval is separate from investigation approval and is asked
  immediately before any external effect (email, share).
- Each side effect has an idempotency key `hash(run_id, action, target,
  data_version)`; a repeated approval or retry returns the first result.
- Decision log entry: run_id, segment, decision, candidate_id, approver,
  timestamp, evidence_ids, data_version.
- The 20% "close call" rule is `REVIEW_CLOSE_CALL_PCT`, config.

### 7.4 State and memory
- Run state: ids, candidates (summaries), ledger, decision log, budget, status.
  No DataFrames, no customer fields.
- Tool-call log: tool, args, data_version, result hash, seconds. Tools are
  pure, so evidence is rebuilt by replaying the log on the same data; a hash
  mismatch means the data changed and the run says so.
- Long-term findings: statement, region/scope, period, source run, confidence,
  `revalidate_by`. Read back only as hypotheses; the next run tests them.
- Monthly snapshot: per unit and stratum, counts, sums, rates, metric
  definition version, eligible-base rule, schema version, data_version.

### 7.5 Where history and memory live (open question 4)
Options, in recommended order:
1. **A memory pack file** (one small `.xlsx` or `.json`: snapshots, decision
   log, findings, notes; aggregates only). Downloaded after each review,
   uploaded with next month's files. Works on Streamlit Cloud, keeps "no
   database, data stays with you", and the user owns it.
2. A local folder when the app runs on the user's machine (same format).
3. An external database: durable and automatic, but company aggregates leave
   the machine, and the "no database" position ends. Only by deliberate decision.

### 7.6 Phase order (revised)
1. Deterministic scorecard and lists (this phase).
2. Snapshot saving in the memory pack, plus backfill from archive files if they exist.
3. Brief on upload (code only) and outcome tracking of past recommendations.
4. Review and investigate segments with the model, verifier, decision log.
5. Long-term findings and business notes.
6. Model comparison for `recommend` and `plan` only.

## 8. Acceptance criteria

Phase 1:
- Scorecard Dashboard columns equal `compute_region_scorecard` to 2 decimals on
  the dummy files and every synthetic book.
- Every candidate passes its self-checks (5.4); a failed check removes it.
- Evals 1-7 in section 6 pass. Evals 1 and 3 must fail on the raw (unstandardised) ranking, which proves
  the standardising is what passes them.
- No customer field in any candidate or summary.

Phase 2-3: a snapshot written then read gives identical lists; a pack from a
different BU or out-of-order month is refused; outcome tracking says
"improved / not improved / within noise" against the band.

Phase 4: scripted decision sequences (approve, pick other, redirect, stop,
repeated approval) give the expected states; a repeated action approval sends
nothing new; refreshing the browser mid-run resumes from the last segment;
recommendation numbers 100% traceable; zero causal phrases; the same input
gives the same recommended_id in 9 of 10 runs (the model part).

Phase 5: a finding older than its revalidate date is shown as a hypothesis and
retested; findings from one BU never appear in another BU's run.

## 9. Open questions for the user

1. Do archive monthly LCC files exist (how many months)? If yes, history can be
   backfilled now.
2. Are there region or branch targets (Collection%, NPA%, 30+)? Leaders manage
   to targets; they would replace the rest-of-portfolio benchmark.
3. Should "new business fell" be its own list? It needs a noise rule for counts.
4. Where should history and memory live (7.5)? Recommended: memory pack file.
5. Confirm the floors: Rs 10 lakh, 0.5 pts, 200 eligible accounts, FDR 0.10,
   20% unmatched.
6. Should the Dashboard's Collection% keep counting cash from loans with no
   EMI due this month (constraint 8)? It inflates some regions far above 100%.

## 10. Phase 1 status (2026-10-24)

Built: `investigation/review/` (`scorecard.py`, `prioritise.py`,
`synthetic.py`), config `REVIEW_*`, `tests/investigation/test_review.py`
(18 tests: the three formulas checked by hand, evals 1-7, reconciliation,
evidence checks, no customer fields). On the dummy files: scorecard
reconciles with Portfolio Intelligence on all 7 Dashboard columns, every
evidence check passes, no region listed (random data: correct), 2.1 s.
Not built: any UI, model call or persistence.

## 11. Build status, 2026-10-25 (order agreed after the 90k measurement)

1. Speed (done): at 90k loans the review went from 169 s to 2.8 s and a
   region deep-dive from about 100 s to 5 s, results unchanged (a fingerprint
   of every candidate and evidence summary). Narrowed pairs share row-wise
   labels, metric masks and both-months positions with the pair they came
   from (`PeriodPair.narrowed`), guarded by an equality test.
2. Background runner (done): `investigation/review/` `runner.py` (one worker
   thread, jobs keyed per browser session + data + filter, last 3 kept),
   `deepdive.py` (shortfall and excess split by stratum, adding up exactly,
   and by branch; worsening runs the Deep Investigation baseline on the shared
   pair, and says how loans moved when no segment concentrates it),
   `brief.py` (fixed templates, no causal wording, one button per region
   looked into). UI: `ui/review_panel.py`, started at the end of each app
   run, shown in the Investigator tab (live steps, then the brief); "Look
   deeper" opens the matching existing Investigator step, no AI call.
   90k loans: whole background run 5.8 s. Sample data: 3.7 s.
3. Analyst brief (done): `investigation/review/analyst.py`. One Gemini call
   over the brief's own sentences and options (no loan rows), JSON out. The
   verifier refuses: missing keys, an option that isn't offered, any number
   not written in the facts, causal wording, share claims the facts don't
   state, a recommendation that doesn't name its region. One retry with the
   failures; a second failure keeps the template brief and says so. "Sits
   primarily in X" is rewritten to "has its largest parts in X" (Flash-Lite
   kept it through a retry; true in one live run, a fifth in another).
   Live eval (`pytest -m llm_eval`, opt-in): 20 of 20 answers accepted on the
   first try, about 2.1 s each; the recommended region was the same 10 of 10
   on both cases. Open: on the sample data the lens for NASHIK (on all three
   lists) split 7 collection shortfall, 3 getting worse.
4. Decision log and next steps (done): `investigation/review/decisions.py`
   (append-only, JSON-lines file at REVIEW_DECISION_LOG, a repeated choice is
   one entry, thread-safe, download and restore per set of files, an uploaded
   log is validated field by field) and `next_steps.py` (priority loans in the
   branch with the largest gap, where the existing "Draft email" builds a
   download that is never sent; priority loans across the region; mark as
   handled, which reopens the review; back). Each entry: the choice, the AI's
   recommendation and whether it was followed, evidence ids, data version,
   period, approver (the signed-in email when the app has sign-in). The
   "Draft email" click is logged as the action's approval. Driven end to end
   through the Investigator tab: 4 decisions and 1 action recorded, a repeated
   click recorded once.
5. Outcome tracking (done): `investigation/review/outcomes.py`. A book key
   (business units plus regions; a sidebar region filter is another book)
   carries across months, unlike the data version. A review choice stores
   the candidate's gap and its error in points (`Candidate.baseline`), since
   last month's own gap needed the month before. Next month: levels
   (shortfall, excess 30+) compare the gap now with then against a 2-error
   band of the combined error; "getting worse" asks whether the region now
   moves better than the rest beyond its band. Statuses: improved, worse, no
   clear change, not comparable (too small or gone), plus "marked handled".
   The page says these compare before and after and don't show a cause.
   The error is calibrated (actual spread over 30 seeds within 10% of the
   reported error), so about 1 unchanged region in 20 reads improved or
   worse by chance. The download now carries the whole book across months.
   Driven end to end across two months with a restart and a restored log.
6. Narration and layout (2026-10-26, from screenshots): the Investigator's
   narration called a table "generally improving" where Collection% fell in
   26 of 50 branches. It now gets the changes counted over every row and
   drops any sentence whose change words those counts don't support
   (`investigator/llm.py::direction_facts`, `_contradicts`); live, 5 of 5
   narrations then described the directions correctly. The tab is now a
   briefing and workspace: the monthly briefing on top (open on arrival,
   folded while a region is worked on), one card per region that stands out
   (`investigation/review/focus.py`: AI pick first, status Open / Looked into
   / Handled from the decision log), and each card opens its own
   conversation with lenses for its lists; questions there are routed with
   the region and the lens ("About the NASHIK region (focus: collection
   shortfall): ..."). Multi-part questions (2026-10-26): in a workspace the
   router may return up to 3 parts (`investigator/llm.py::_parts`, a
   hand-off to AI Query or an unknown step is never a part); each runs
   through validate_and_correct and one deterministic step, shown as "Part 1
   of 2: By segment" under the one question; a part that can't run says so
   and the others still run. Outside a workspace, routing is unchanged.
   Live: "is it a segment or an executive?" gave the two parts 3 of 3 times;
   "by branch and by fuel type" two parts; "which branches are weakest?"
   stayed one answer.
7. Last month in the analyst summary (2026-10-26): `analyst_facts` carries
   last month's outcomes (region, list, status, the outcome sentence, whose
   figures may then be quoted), the prompt (version 3) asks how they moved,
   and the verifier refuses a "last month" sentence that gives a region a
   status its outcome doesn't have, and any "<region> is on N lists" that
   doesn't match the brief (live: "NORTH ... is on two lists" for a region on
   one; a count in words the number check can't see). The background run
   reads the outcomes from the log; a log restored later rewrites the
   analyst view once (one call per review and set of outcomes). Live: 10 of
   10 accepted on the first try, all reporting WEST's improvement; the
   restore-later path rewrote the view with it.
8. Not built: multi-month history for "normal for this month" still needs
   the history file (section 7.5).
