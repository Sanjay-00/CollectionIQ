"""
The Portfolio Review in the Investigator tab, laid out as a briefing and a
workspace (2026-10-26, replacing one long panel folded into an expander):

  render_briefing      the monthly briefing: live steps while the review
                       runs, then the headline, last month's focus, the
                       analyst view, and the scorecard / data notes /
                       decisions as tabs
  render_focus_strip   one card per region that stands out (focus.py), plus
                       "All regions"; the Investigator opens a card as that
                       region's own conversation (its workspace)
  choose               logs a choice from the review and sets the context its
                       next steps and actions are logged against

The review starts by itself at the end of each app run (app.py), so it is
usually ready by the time the Investigator tab is opened. Jobs live in one
process-wide runner, keyed by this browser session plus the data on screen,
so one session's brief can never be shown in another. Every figure is
computed in code; the analyst view is one AI call per review, checked against
those figures, and left out when it fails the checks or there is no API key.
"""
from __future__ import annotations

import hashlib
import logging
import uuid

import pandas as pd
import streamlit as st

from investigation.review.analyst import outcomes_signature, write_analysis
from config import REVIEW_DECISION_LOG
from investigation.review.decisions import DecisionLog
from investigation.review.focus import HANDLED, LOOKED_INTO, Focus, focuses
from investigation.review.next_steps import largest_branch
from investigation.review.outcomes import IMPROVED, NOT_COMPARABLE, OUTCOME_NOTE, WORSE, track_outcomes
from investigation.review.runner import Job, ReviewRunner
from investigation.tools.change import month_word
from ui.components import _esc, _safe_df

logger = logging.getLogger(__name__)

_SESSION, _SCORECARD = "_review_session", "_review_scorecard"
REVIEW_CONTEXT = "_review_ctx"   # the review choice a workspace came from: its next steps and actions log against it
REVIEW_OPEN = "_review_open"     # "Back to the review" / "Mark as handled": unfold the briefing
ALL_REGIONS = "__all_regions__"  # the focus strip's general (whole portfolio) conversation


@st.cache_resource
def _runner() -> ReviewRunner:
    # The analyst summary is one Gemini call per review, skipped without a key,
    # written with how last month's chosen regions moved when the decision log
    # already holds last month (resolved here, in the page's thread; the
    # worker only reads it). A log restored later is handled by
    # _refresh_analysis.
    log = decision_log()
    return ReviewRunner(analyst=lambda b: write_analysis(b, outcomes=track_outcomes(b, log.entries_for_book(b.book))))


def _refresh_analysis(job: Job) -> None:
    """A decision log restored after the background run can change last
    month's outcomes: rewrite the analyst view once with them (one AI call,
    tried once per review and set of outcomes)."""
    b = job.brief
    a = b.analysis
    if a is None or a.status not in ("ok", "fallback") or not b.findings:
        return
    try:
        outcomes = track_outcomes(b, decision_log().entries_for_book(b.book))
    except Exception:
        logger.exception("Couldn't track last month's outcomes")
        return
    seen = outcomes_signature(outcomes)
    tried = st.session_state.setdefault("_analyst_refreshed", set())
    if seen == a.outcomes_seen or (job.key, seen) in tried:
        return
    tried.add((job.key, seen))
    with st.spinner("Updating the analyst view with last month's outcomes..."):
        fresh = write_analysis(b, outcomes=outcomes)
    if fresh.status != "skipped":
        b.analysis = fresh


def _key(data_version: int, filter_key: str, curr_month: str, prev_month: str) -> str:
    session = st.session_state.setdefault(_SESSION, uuid.uuid4().hex)
    return f"{session}|{data_version}|{filter_key}|{curr_month}|{prev_month}"


def start_review(df_curr: pd.DataFrame, df_prev: pd.DataFrame, curr_month: str, prev_month: str | None,
                 data_version: int, filter_key: str) -> Job | None:
    """This session's review of the data on screen, started if it isn't yet.
    None when there is no last month to compare with. Never raises: a review
    problem must not break the page it runs behind."""
    if not prev_month or df_prev is None or df_prev.empty or df_curr is None or df_curr.empty:
        return None
    try:
        key, runner = _key(data_version, filter_key, curr_month, prev_month), _runner()
        job = runner.get(key)
        if job is not None and job.status != "failed":
            return job
        # Copied for the worker thread only when a run starts: the page may
        # change its own frames while the review reads these.
        return runner.submit(key, df_curr.copy(), df_prev.copy(), curr_month, prev_month)
    except Exception:
        logger.exception("Couldn't start the portfolio review")
        return None


@st.cache_resource
def decision_log() -> DecisionLog:
    """One log per server process, written to REVIEW_DECISION_LOG. Each
    session only ever sees and downloads the entries for its own files."""
    return DecisionLog(REVIEW_DECISION_LOG)


def _approver() -> str:
    """Who chose: the signed-in user's email when the app has sign-in."""
    try:
        if st.user.is_logged_in:
            return str(st.user.email)[:120]
    except Exception:
        pass
    return "this browser session"


def review_context(job: Job) -> dict:
    """What every decision from this review records about it."""
    b, a = job.brief, job.brief.analysis
    return {"data_version": b.data_version, "run": hashlib.sha256(job.key.encode()).hexdigest()[:12],
            "period": dict(b.period), "book": b.book,
            "ai_recommended": a.recommended_option if a is not None and a.status == "ok" else None}


def log_decision(ctx: dict, segment: str, option_id: str, **fields) -> None:
    """One entry in the durable decision log; a repeat of the same choice is
    the same entry. Never breaks the page."""
    try:
        rec = ctx.get("ai_recommended")
        decision_log().append(
            data_version=ctx["data_version"], run=ctx["run"], period=ctx.get("period", {}), book=ctx.get("book", ""),
            segment=segment,
            option_id=option_id, ai_recommended=rec,
            followed_ai=(option_id == rec) if (segment == "review" and rec) else None,
            approver=_approver(), **fields)
    except Exception:
        logger.exception("Couldn't record the decision")


def choose(job: Job, candidate_id: str) -> dict:
    """A region and list chosen from the review (opening its card, or
    switching its lens): logged with its evidence and its figures (for next
    month's outcome), and set as the context its next steps log against."""
    b, ctx = job.brief, review_context(job)
    name, region = candidate_id.split(":", 1)
    finding = next((f for f in b.findings if f.candidate_id == candidate_id), None)
    branch = largest_branch(b, candidate_id)
    cand = next((c for c in b.candidates.get(name, []) if c.region == region), None)
    log_decision(ctx, "review", f"investigate:{candidate_id}", region=region, list_name=name, branch=branch or "",
                 evidence_ids=finding.evidence_ids if finding else [],
                 baseline=cand.baseline() if cand is not None else {})
    st.session_state[REVIEW_CONTEXT] = {**ctx, "candidate_id": candidate_id, "region": region, "list": name,
                                        "branch": branch}
    return st.session_state[REVIEW_CONTEXT]


def review_focuses(job: Job | None) -> list[Focus]:
    """The focus cards for a finished review, with where the leader is on each."""
    if job is None or job.status != "done" or job.brief is None or job.brief.status != "ok":
        return []
    return focuses(job.brief, decision_log().for_data(job.brief.data_version))


_CSS = """<style>
.rv { background:#0d1117; border:1px solid #21262d; border-radius:12px; padding:14px 18px; margin:4px 0 6px 0; }
.rv-tag { color:#FFC000; font-size:10.5px; font-weight:800; letter-spacing:1px; text-transform:uppercase; }
.rv-head { color:#ffffff; font-size:15.5px; font-weight:700; margin:4px 0 8px 0; }
.rv-item { color:#c9d1d9; font-size:13.5px; line-height:1.6; margin:0 0 8px 0; padding-left:12px;
           border-left:3px solid #FFC000; }
.rv-item b { color:#ffffff; }
.rv-note { color:#8b949e; font-size:12px; margin-top:6px; }
.rv-analyst { background:#161b22; border:1px solid #2d333b; border-radius:10px; padding:12px 14px; margin:2px 0 12px 0; }
.rv-sum { color:#e6edf3; font-size:14px; line-height:1.6; margin:4px 0 8px 0; }
.rv-rec { color:#e6edf3; font-size:13.5px; line-height:1.6; margin:0 0 6px 0; }
.rv-rec span { color:#FFC000; font-weight:700; margin-right:4px; }
.rv-sub { color:#c9d1d9; font-size:13px; line-height:1.6; }
.rv-sub b { color:#8b949e; font-weight:600; margin-right:4px; }
.rv-step { color:#c9d1d9; font-size:13px; line-height:1.7; }
.rv-step span { color:#3fb950; margin-right:6px; }
.fc-name { font-size:15px; font-weight:800; color:#111827; }
.fc-pick { background:#FFC000; color:#111; font-size:10.5px; font-weight:800; border-radius:99px; padding:1px 8px; margin-left:6px; }
.fc-status { font-size:11px; font-weight:700; border-radius:99px; padding:1px 8px; margin-left:6px; }
.fc-open { background:#f3f4f6; color:#4b5563; }
.fc-looked { background:#dbeafe; color:#1e40af; }
.fc-handled { background:#dcfce7; color:#166534; }
.fc-chip { font-size:12.5px; color:#374151; line-height:1.55; }
.fc-chip b { color:#111827; font-weight:700; }
</style>"""


@st.fragment(run_every="1s")
def _live(key: str) -> None:
    """The steps as they finish; hands back to a full rerun once done."""
    job = _runner().get(key)
    if job is None or job.status != "running":
        st.rerun(scope="app")
        return
    steps = "".join(f'<div class="rv-step"><span>&#10003;</span>{_esc(t)}</div>' for t, _ in job.steps)
    st.markdown(f'{_CSS}<div class="rv"><div class="rv-tag">Monthly briefing</div>'
                f'<div class="rv-head">Reviewing every region in the background</div>{steps}'
                f'<div class="rv-note">{job.seconds:.0f} s so far. Every figure is computed in code; the summary '
                f'at the end is one AI call, checked against them. You can keep working.</div></div>',
                unsafe_allow_html=True)


def _briefing_html(job: Job) -> str:
    b = job.brief
    overlaps = "".join(f'<div class="rv-item">{_esc(o)}</div>' for o in b.overlaps)
    a = b.analysis
    analyst, source = "", "Code only, no AI."
    if a is not None and a.status == "ok":
        labels = {o["id"]: o["label"].replace("Look deeper at ", "") for o in b.options}
        rec = (f'<div class="rv-rec"><span>Look first:</span> {_esc(labels.get(a.recommended_option, ""))}. '
               f'{_esc(a.why)}</div>' if a.recommended_option else
               f'<div class="rv-rec"><span>Your call:</span> {_esc(a.why)}</div>')
        more = "".join(f'<div class="rv-sub"><b>{k}</b> {_esc(v)}</div>'
                       for k, v in (("Instead:", a.tradeoff), ("Next month:", a.watch)) if v)
        analyst = (f'<div class="rv-analyst"><div class="rv-tag">Analyst view</div>'
                   f'<div class="rv-sum">{_esc(a.summary)}</div>{rec}{more}'
                   f'<div class="rv-note">Written by AI from the figures in the region cards; every number was '
                   f'checked against them.</div></div>')
        source = "The figures are computed in code."
    elif a is not None and a.status == "fallback":
        source = "The AI summary didn't pass its checks, so only the computed figures are shown."
    pick = ("Each region below has its own workspace: open one to see where its gap sits and what to do next."
            if b.findings else "")
    return (f'{_CSS}<div class="rv"><div class="rv-head">{_esc(b.headline)}</div>{_outcomes_html(b)}{analyst}'
            f'{overlaps}<div class="rv-note">{_esc(pick)} Worked out in {job.seconds:.1f} s from your two files. '
            f'{source} Every figure compares like for like with the rest of the portfolio.</div></div>')


def render_briefing(job: Job | None, prev_month: str | None, expanded: bool) -> None:
    """The monthly briefing, or where the review is. expanded: open when no
    region is being worked on, folded while one is (it stays one click away)."""
    if job is None:
        if not prev_month:
            st.caption("Upload last month's file too, and a monthly briefing of every region appears here.")
        return
    if job.status == "running":
        _live(job.key)
        return
    if job.status == "failed":
        st.warning(f"The monthly briefing couldn't finish ({job.error}).")
        if st.button("Try the briefing again", key="rv_retry"):
            st.rerun()                                # start_review retries a failed job
        return
    _refresh_analysis(job)
    b = job.brief
    when = f"{month_word(b.period['curr'])} vs {month_word(b.period['prev'])}" if b.period else ""
    suffix = hashlib.md5(job.key.encode()).hexdigest()[:8]    # widget keys unique per review, stable across reruns
    with st.expander(f"Monthly briefing, {when}", expanded=expanded, icon=":material/summarize:"):
        st.markdown(_briefing_html(job), unsafe_allow_html=True)
        tab_card, tab_notes, tab_log = st.tabs(["Full scorecard", "Data notes and how to read this",
                                                f"Your decisions ({len(decision_log().for_data(b.data_version))})"])
        with tab_card:
            if b.scorecard is not None and not b.scorecard.empty:
                st.dataframe(_safe_df(b.scorecard.copy()), hide_index=True, width="stretch")
        with tab_notes:
            for line in [*b.data_notes, *b.caveats] or ["No data notes for these files."]:
                st.markdown(f"- {_esc(line)}", unsafe_allow_html=True)
        with tab_log:
            _decisions(b, suffix)


_STATUS_CLASS = {LOOKED_INTO: "fc-looked", HANDLED: "fc-handled"}


def render_focus_strip(cards: list[Focus], active: str | None, suffix: str) -> str | None:
    """One card per region that stands out, then "All regions". Returns the
    region clicked (ALL_REGIONS for the general conversation), else None."""
    clicked = None
    with st.container(horizontal=True, gap="small", key="focus_strip"):
        for card in cards:
            with st.container(border=True, key=f"focus_card_{_key_part(card.region)}", width=250):
                pick = '<span class="fc-pick">AI pick</span>' if card.ai_pick else ""
                status = (f'<span class="fc-status {_STATUS_CLASS.get(card.status, "fc-open")}">'
                          f'{_esc(card.status)}</span>')
                chips = "".join(f'<div class="fc-chip"><b>{_esc(label.capitalize())}</b> {_esc(amount)}</div>'
                                for label, amount in card.chips)
                st.markdown(f'{_CSS}<div class="fc-name">{_esc(card.region)}{pick}{status}</div>{chips}',
                            unsafe_allow_html=True)
                is_active = card.region == active
                if st.button("Working on it" if is_active else "Open", key=f"focus_open_{_key_part(card.region)}_{suffix}",
                             type="primary" if (card.ai_pick and not is_active) else "secondary",
                             disabled=is_active, width="stretch"):
                    clicked = card.region
        with st.container(border=True, key="focus_card_all", width=200):
            st.markdown(f'{_CSS}<div class="fc-name">All regions</div>'
                        '<div class="fc-chip">Ask about anything in the portfolio.</div>', unsafe_allow_html=True)
            if st.button("Working on it" if active == ALL_REGIONS else "Open", key=f"focus_open_all_{suffix}",
                         disabled=active == ALL_REGIONS, width="stretch"):
                clicked = ALL_REGIONS
    return clicked


def _key_part(text: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in str(text))[:40]


_OUTCOME_COLOUR = {IMPROVED: "#3fb950", WORSE: "#f85149", NOT_COMPARABLE: "#8b949e"}


def _outcomes_html(b) -> str:
    """Last month's chosen regions, this month (outcomes.py). Empty when there
    are none: a first review, or the log from last month isn't here yet."""
    try:
        outcomes = track_outcomes(b, decision_log().entries_for_book(b.book))
    except Exception:
        logger.exception("Couldn't track last month's outcomes")
        return ""
    if not outcomes:
        return ""
    rows = "".join(
        f'<div class="rv-sub"><b style="color:{_OUTCOME_COLOUR.get(o.status, "#d29922")}">'
        f'{_esc(o.status.capitalize())}.</b> {_esc(o.text)}</div>' for o in outcomes)
    return (f'<div class="rv-analyst"><div class="rv-tag">Last month&#39;s focus</div>{rows}'
            f'<div class="rv-note">{_esc(OUTCOME_NOTE)}</div></div>')


_STEP_NAMES = {"review": "Chose from the review", "next_step": "Next step", "action": "Approved an action",
               "close": "Marked as handled"}


def _decisions(b, suffix: str) -> None:
    """This review's decisions, and the log to keep or restore (Streamlit
    Cloud forgets files on a restart; the downloaded log doesn't)."""
    log = decision_log()
    mine = log.for_data(b.data_version)
    if mine:
        st.dataframe(_safe_df(pd.DataFrame([{
            "When": d.at, "Step": _STEP_NAMES.get(d.segment, d.segment), "Choice": d.option_id,
            "AI recommended": d.ai_recommended or "", "Followed the AI": "" if d.followed_ai is None
            else ("yes" if d.followed_ai else "no"), "By": d.approver} for d in mine])),
            hide_index=True, width="stretch")
        st.download_button("Download the decision log", data=log.export_jsonl(book=b.book or None,
                                                                           data_version=None if b.book else b.data_version),
                           mime="application/x-ndjson",
                           file_name=f"decision_log_{b.data_version}.jsonl", key=f"rv_log_dl_{suffix}")
    else:
        st.caption("Choices you make from the region cards are recorded here, with the AI's recommendation "
                   "next to them.")
    upload = st.file_uploader("Restore a downloaded decision log", type=["jsonl"], key=f"rv_log_up_{suffix}")
    if upload is not None:
        added, skipped = log.import_jsonl(upload.getvalue())
        st.caption(f"Restored {added} decision(s); {skipped} line(s) already here or not readable.")
