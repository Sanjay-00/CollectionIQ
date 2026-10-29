"""Pre-flight check before a demo: drives the real app (app.py, under
Streamlit's test runner) through exactly what the demo shows, and prints PASS
or FAIL for each step. Run it the morning of the demo, and again after any
change:

    python demo_check.py            # the whole demo, AI included
    python demo_check.py --no-ai    # only what works with AI switched off

On the built-in sample (Fill Sample Data), so no file is needed. With AI on,
every example question and suggestion is asked for real (about 20 Gemini
calls); each answer must come back without an error, with rows, in time.
Exit code 0 only when every step passed.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import traceback

from dotenv import load_dotenv

SLOW_SECONDS = 12          # an answer slower than this would feel broken on stage
REVIEW_WAIT_SECONDS = 90   # the background review on the sample takes a few seconds


class Check:
    def __init__(self):
        self.rows: list[tuple[str, bool, float, str]] = []

    def step(self, name: str, fn) -> object:
        start = time.perf_counter()
        try:
            detail = fn() or ""
            ok = True
        except AssertionError as e:
            ok, detail = False, str(e) or "check failed"
        except Exception as e:
            ok, detail = False, f"{type(e).__name__}: {e}"
            traceback.print_exc()
        took = time.perf_counter() - start
        self.rows.append((name, ok, took, str(detail)))
        print(f"  {'PASS' if ok else 'FAIL'}  {name:58} {took:5.1f}s  {str(detail)[:90]}", flush=True)
        return ok

    def report(self) -> int:
        failed = [r for r in self.rows if not r[1]]
        print(f"\n{len(self.rows) - len(failed)} of {len(self.rows)} steps passed.")
        for name, _, _, detail in failed:
            print(f"  FAILED: {name}: {detail}")
        print("READY FOR THE DEMO" if not failed else "NOT READY: fix the failures above first")
        return 0 if not failed else 1


def _clean(at, where: str) -> None:
    """No crash and no red error box on the page."""
    assert not at.exception, f"{where}: {at.exception[0].message[:200]}"
    errors = [e.value for e in at.error]
    assert not errors, f"{where}: error on the page: {errors[0][:200]}"


def _app():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file("app.py", default_timeout=300).run()
    _clean(at, "landing page")
    next(b for b in at.button if b.label == "Fill Sample Data").click().run()
    _clean(at, "after loading the sample")
    return at


def _open(at, tab: str):
    at.button_group(key="_active_section").set_value(tab).run()
    _clean(at, tab)
    return at


def _wait_for_briefing(at) -> str:
    deadline = time.time() + REVIEW_WAIT_SECONDS
    while time.time() < deadline:
        cards = [b for b in at.button if (b.key or "").startswith("focus_open_") and "_all_" not in b.key]
        if cards:
            return f"{len(cards)} region cards"
        time.sleep(2)
        at.run()
        _clean(at, "waiting for the briefing")
    raise AssertionError(f"the review briefing wasn't ready after {REVIEW_WAIT_SECONDS}s")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--no-ai", action="store_true", help="check only what works with AI switched off")
    args = parser.parse_args()
    # Tab names carry emoji: a Windows pipe or log file (cp1252) can't print them.
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8", errors="replace")

    load_dotenv(".env")
    if args.no_ai:
        os.environ["COLLECTIONIQ_AI"] = "off"
    os.environ["LANGSMITH_TRACING"] = "false"          # a rehearsal is not usage: keep it out of LangSmith
    from gemini_client import ai_unavailable_reason
    from ui.components import TAB_LABELS
    ai = not ai_unavailable_reason()
    print(f"Demo check: AI {'ON' if ai else 'OFF'} ({ai_unavailable_reason() or 'Gemini will be called for real'})\n")
    c = Check()
    state: dict = {}

    print("The app, on the built-in sample")
    if not c.step("Load the app and Fill Sample Data", lambda: state.update(at=_app())):
        return c.report()
    for tab in TAB_LABELS:
        c.step(f"Open {tab}", lambda tab=tab: _open(state["at"], tab) and "")
    c.step("Investigator: the review briefing is ready", lambda: _wait_for_briefing(_open(state["at"], TAB_LABELS[7])))

    print("\nReports (Leadership, each format)")
    for fmt in ("PDF", "HTML", "Excel"):
        def report(fmt=fmt):
            at = _open(state["at"], TAB_LABELS[8])
            at.button_group(key="rpt2_preset").set_value("Leadership").run()
            at.button_group(key="rpt2_fmt").set_value(fmt).run()
            at.button(key="rpt2_generate").click().run()
            _clean(at, f"{fmt} report")
            out = at.session_state["rpt2"]
            assert out["data"] and len(out["data"]) > 2_000, f"the {fmt} file is empty"
            return f"{out['name']}, {len(out['data']) // 1024} KB"
        c.step(f"Generate the {fmt} report", report)

    if ai:
        print("\nAI Query: every example button")
        from ui.tabs.ai_query import _EXAMPLES
        for i, (short, _) in enumerate(_EXAMPLES):
            def example(i=i):
                at = _open(state["at"], TAB_LABELS[6])
                at.button(key=f"aiq_example_{i}").click().run()
                t = time.perf_counter()
                at.button(key="aiq_run").click().run()
                took = time.perf_counter() - t
                _clean(at, "AI Query")
                r = at.session_state["ai_result"]
                assert not r.get("error"), r["error"]
                if r.get("needs_clarification"):
                    return f"asks which reading: {r.get('clarification_options')} ({took:.1f}s)"
                rows = len(r["result_df"])
                assert rows > 0, "answered with no rows"
                assert took < SLOW_SECONDS, f"took {took:.1f}s"
                return f"{rows} rows in {took:.1f}s"
            c.step(f"AI Query: {short}", example)

        def summary():
            at = _open(state["at"], TAB_LABELS[6])
            at.checkbox(key="ai_gen_summary").check().run()
            at.button(key="aiq_example_1").click().run()
            at.button(key="aiq_run").click().run()
            _clean(at, "AI Query with summary")
            insights = at.session_state["ai_result"].get("insights") or ""
            assert insights.strip() and "unavailable" not in insights, f"no AI summary: {insights[:100]}"
            at.checkbox(key="ai_gen_summary").uncheck().run()
            return f"{len([x for x in insights.splitlines() if x.strip()])} checked observations"
        c.step("AI Query: with the AI summary ticked", summary)

        print("\nInvestigator: every suggestion (fresh session each)")
        from ui.tabs.investigator import _SUGGESTIONS
        for label in _SUGGESTIONS:
            def suggestion(label=label):
                at = _open(_app(), TAB_LABELS[7])
                t = time.perf_counter()
                at.button_group(key="investigator_suggestions").set_value(label).run()
                took = time.perf_counter() - t
                _clean(at, "Investigator")
                turns = [x for th in at.session_state["investigator_threads"].values() for x in th["turns"]]
                answer = turns[-1]
                assert answer.get("role") == "assistant", "no answer"
                assert answer.get("kind") != "handoff" or answer.get("question"), "empty hand-off"
                text = answer.get("text") or ""
                assert not text.startswith("I couldn't"), text
                assert took < SLOW_SECONDS + 8, f"took {took:.1f}s"
                what = answer.get("entity_type") or answer.get("kind") or "reply"
                return f"{what} in {took:.1f}s"
            c.step(f"Investigator: {label}", suggestion)

        def workspace():
            at = _open(_app(), TAB_LABELS[7])
            _wait_for_briefing(at)
            card = next(b for b in at.button if (b.key or "").startswith("focus_open_") and "_all_" not in b.key)
            card.click().run()
            _clean(at, "opening a region")
            at.text_area(key="ask_box").set_value("is it a segment or an executive?")
            at.button(key="ask_submit").click().run()
            _clean(at, "asking in the region")
            turns = [x for th in at.session_state["investigator_threads"].values() for x in th["turns"]]
            assert turns and turns[-1].get("role") == "assistant", "no answer"
            assert not (turns[-1].get("text") or "").startswith("I couldn't"), turns[-1]["text"]
            state["workspace"] = at
            return card.key.replace("focus_open_", "").rsplit("_", 1)[0]
        if c.step("Investigator: open a region, ask a follow-up", workspace):
            def drill():
                # Last click in this session: a drill reruns the page itself,
                # after which the test runner's copy of the page is stale.
                at = state["workspace"]
                btn = next((b for b in at.button if (b.label or "").startswith("Analyse further: which loans")), None)
                assert btn is not None, "no 'which loans are driving' button under the answer"
                btn.click().run()
                _clean(at, "the drill")
                turns = [x for th in at.session_state["investigator_threads"].values() for x in th["turns"]]
                assert turns[-2].get("text", "").startswith("which loans are driving"), "the drill added no turn"
                return btn.label
            c.step("Investigator: Analyse further from that answer", drill)

        def branch_priority():
            at = _open(_app(), TAB_LABELS[7])
            _wait_for_briefing(at)
            next(b for b in at.button if (b.key or "").startswith("focus_open_") and "_all_" not in b.key).click().run()
            btn = next((b for b in at.button if (b.label or "").startswith("Priority loans in ")), None)
            assert btn is not None, "no 'Priority loans in <branch>' button on the region"
            btn.click().run()
            _clean(at, "priority loans")
            turns = [x for th in at.session_state["investigator_threads"].values() for x in th["turns"]]
            assert turns and turns[-1].get("role") == "assistant", "no answer"
            return btn.label
        c.step("Investigator: the region's 'Priority loans in <branch>' button", branch_priority)
    else:
        print("\nAI steps skipped (AI off): the demo's AI part needs AI on.")

    return c.report()


if __name__ == "__main__":
    sys.exit(main())
