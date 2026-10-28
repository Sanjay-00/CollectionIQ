"""The whole Investigator tab, end to end (2026-10-27). Nothing in the suite
rendered it before: refactors were checked with a scratch harness that ended
with the session. The demo portfolio, a real review brief and a scripted
router (tests/tab_harness.py) drive every way a question can go."""
import pytest
from streamlit.testing.v1 import AppTest

import tab_harness as h

pytestmark = pytest.mark.filterwarnings("ignore")


@pytest.fixture(scope="module")
def data():
    return h.build()


@pytest.fixture
def page(data, monkeypatch, tmp_path):
    """A fresh browser session on the tab, with an AI key set (no call is made)."""
    monkeypatch.setenv("GOOGLE_API_KEY", "test")
    h.install(monkeypatch, str(tmp_path / "decisions.jsonl"))
    at = AppTest.from_function(h.app, default_timeout=180).run()
    assert not at.exception, at.exception
    return at


def ask(at, question):
    at.text_area(key="ask_box").set_value(question)
    at.button(key="ask_submit").click().run()
    assert not at.exception, at.exception
    return at


def turns(at):
    return [t for th in at.session_state["investigator_threads"].values() for t in th["turns"]]


def remembered(at):
    return [r for th in at.session_state["investigator_threads"].values() for r in th["memory"]._entities.values()]


def texts(at):
    return " ".join(str(getattr(e, "value", "")) + str(getattr(e, "body", "")) for e in
                    [*at.markdown, *at.warning, *at.error, *at.info, *at.caption])


def test_first_page_has_the_briefing_cards_and_question_box(page):
    assert any((b.key or "").startswith("focus_open_") for b in page.button)
    assert page.text_area(key="ask_box") is not None


def test_a_conversation_through_every_kind_of_answer(page, data):
    """One session, as a user would: each question adds to the same conversation."""
    at = ask(page, "q_branch")
    answer = turns(at)[-1]
    assert answer["entity_type"] == "branch" and answer["entity_value"] == data["branch"]
    steps = " ".join(answer["steps"])
    assert "Understood" in steps and "Ran:" in steps
    assert any(r.step_type == "entity_summary" for r in remembered(at))

    # An executive typed in lower case is stored under the file's own spelling
    ask(at, "q_exec")
    assert tuple(turns(at)[-1]["entity_value"]) == data["executive"]

    ask(at, "q_clarify")
    assert turns(at)[-1]["text"] == "Which branch do you mean?"

    ask(at, "q_handoff")
    assert turns(at)[-1]["kind"] == "handoff" and turns(at)[-1]["question"] == "list the NPA loans"

    ask(at, "q_bad")
    assert turns(at)[-1]["text"].startswith("I couldn't run that")

    before = len(remembered(at))
    ask(at, "q_parts")
    assert len(remembered(at)) == before + 2

    ask(at, "what can you help me with")
    assert turns(at)[-1].get("kind") == "info"


def test_analyse_further_drills_into_the_branch_executives(page):
    """Last in its session: a drill ends in st.rerun(), after which AppTest's
    copy of the page keeps the first pass's widgets (a browser clears them),
    so a further click would send a value for a picker that no longer exists."""
    ask(page, "q_branch")
    drill = next(b for b in page.button if (b.key or "").startswith("drill_") and "executives driving" in b.label)
    drill.click().run()
    assert not page.exception, page.exception
    assert turns(page)[-2]["text"].startswith("executives driving")
    assert any(r.entity_type == "branch_executives" and r.step_type == "underperformer_quartile"
               for r in remembered(page))


def test_without_an_ai_key_the_question_is_refused_plainly(page, monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY")
    ask(page, "q_branch")
    assert "isn't set up here" in turns(page)[-1]["text"] and not remembered(page)


def test_a_region_workspace_tells_the_router_which_region(page):
    card = next(b for b in page.button if (b.key or "").startswith("focus_open_") and "_all_" not in b.key)
    region = card.key[len("focus_open_"):].rsplit("_", 1)[0]
    card.click().run()
    assert not page.exception, page.exception
    ask(page, "q_parts")
    question, allow_multi = page.session_state["ROUTED"][-1]
    assert question.startswith("About the ") and region.replace("_", " ") in question.upper()
    assert allow_multi is True
