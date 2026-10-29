# CollectionIQ demo runbook

A 12 minute stakeholder demo on the built-in sample, with checks before it and
a fallback for everything that can go wrong on the day.

## The day before

1. `python demo_check.py`: every step must say PASS and the last line
   READY FOR THE DEMO. It clicks through the real app on the sample (every tab,
   the briefing, all three report formats) and asks every example question and
   suggestion of the AI for real.
2. Rehearse the talk track below once, start to finish, out loud, with a timer.
3. Decide which data you show (see "The sample's numbers" below).

## One hour before

1. Laptop on power, notifications off, other apps closed, browser zoom 100%.
2. Check the network the room will use. Keep a phone hotspot ready.
3. `python demo_check.py` again (two minutes). Anything other than READY:
   use the AI-off plan below.
4. Start the app: `streamlit run app.py`. Open it in the browser.
5. **Run the whole talk track once in this browser and do not restart the app
   afterwards.** AI Query keeps each answered question's plan, so in the demo
   the same example buttons answer at once, with no AI call at all.
6. Click Generate on a Leadership PDF once, so the download is quick later.
7. Refresh the page (keeps the app running, clears the screen) and leave it on
   the landing page.

## Talk track (about 12 minutes)

| Time | Show | Say |
|---|---|---|
| 0:00 | Landing page | Today the monthly collection review means hours in Excel across regional files. This reads the two monthly files and does that first pass. |
| 0:30 | Click **Fill Sample Data** | Sample data: 8,000 loans, 10 regions, August against July. With real files you upload them here. |
| 1:00 | **Dashboard** | Collection 93.7%, up from 91.9% in July. Every figure here is calculated directly from the file, no AI involved. |
| 2:30 | **Investigator**: the briefing | Without being asked anything, it compared both months and picked 4 regions that stand out, each against a fair comparison: loans in the same buckets or of the same age elsewhere. |
| 3:30 | The **NASHIK** card | NASHIK collected ₹33.96L less than a like-for-like book: 71.6% against 95.8%. The shortfall sits mostly in MALEGAON and YEOLA. Its 30+ rate rose 9.4 points while the rest of the portfolio moved 0.4. |
| 5:00 | **Open** NASHIK, ask "is it a segment or an executive?" | It answers by running the analysis on the data, and shows the steps it took under the answer. |
| 6:30 | Under the answer: **Analyse further: which loans are driving ...** (the metric named varies with the AI's answer; click whichever appears) | Each answer offers the next drill: from the region to its executives, to the exact loans behind the number. |
| 7:00 | Region header: **Priority loans in MALEGAON (largest branch gap)** | And straight to the action list for the worst branch: which loans to chase first. |
| 7:45 | **AI Query**, example **Branches with the biggest NPA drop vs last month** | Plain English in, an exact table out. The AI only reads the question; the numbers come from the file and are checked. |
| 9:00 | **Report**: Leadership, PDF, **Generate**, download | The monthly pack for leadership, in one click. Also by zone, region or branch, as PDF, HTML or Excel. |
| 10:00 | Close | Data stays in our hands: customer names and phone numbers are never sent to the AI, and the AI can be switched off entirely. Proposed next step: a one-month pilot with 2 or 3 managers on real files. |

## Questions you will likely get

- **Is our data safe?** Customer names and mobile numbers never go to the AI.
  Region, branch and loan figures do go to Google Gemini when a question is
  asked. One setting (`COLLECTIONIQ_AI=off` in `.env`) stops every AI call and
  every trace; the dashboard, reports and briefing still work. Approval for
  real data is being requested before the pilot.
- **Can the AI make up numbers?** It doesn't calculate anything. Code computes
  every figure from the file; the AI picks which analysis to run and words the
  result, and any number in its wording that isn't in the data is removed.
- **How do we know the figures are right?** The dashboard formulas are the same
  ones every tab uses, and the pilot starts by checking them against the MIS
  we already produce.
- **"Accounts that need immediate action" says 3,709?** That's every account
  that matches one of 7 priority rules. It ranks them and shows the top 30 to
  work first; the count is the pool, not the to-do list.
- **What does it cost?** Gemini Flash-Lite, the lowest-cost tier: about
  11,000 tokens per question, most of them served from Google's prompt cache. Repeated questions reuse their plan for free.

## The sample's numbers

The sample is synthetic, calibrated against our real portfolio in aggregate
(README, "Try It in 30 Seconds"), and each region carries a story, so have
these ready if someone asks about a number:

- **NPA% around 37% of accounts:** the calibrated level. If it doesn't look
  like our book to this audience, say so up front or demo on real files (only
  with data approval).
- **SATARA's Collection% of 128.6%:** an improving region with catch-up
  collections; overdue arrears recovered on top of the month's demand.
- **SOLAPUR:** chronic non-payers, the highest NPA. **NASHIK:** worsening month
  on month, one weak branch (MALEGAON). **PUNE:** healthiest, with a star and a
  weak executive in the same branch.

## If something goes wrong

| What happens | Do this |
|---|---|
| An AI answer is slow (over 10s) or fails | Say "let me show the part that needs no AI" and go to the briefing cards and Analyse further buttons: they're computed in code. |
| Network or Gemini down | Stop the app (Ctrl+C), set `COLLECTIONIQ_AI=off` in `.env`, run `streamlit run app.py`, Fill Sample Data. Dashboard, briefing, drills and reports all work; say AI is switched off for this session. |
| A tab shows an error | Click **Clear cache & reload** in the sidebar. The error is also in the terminal for afterwards. |
| The app won't start | Have the Leadership PDF and a 2-minute screen recording of the talk track, made during the rehearsal, ready to show. |
