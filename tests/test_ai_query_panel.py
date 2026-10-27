"""AI Query's example questions (code review, 2026-10-27): they used
st.components.v1.html, which Streamlit deprecated ("will be removed after
2026-06-01"), with JavaScript reaching into the parent page to fill the
question box. Now native buttons fill it through session state."""
import pathlib

from streamlit.testing.v1 import AppTest

_SCRIPT = """
import pandas as pd
from ui.tabs.ai_query import render_ai_query_tab
render_ai_query_tab(pd.DataFrame(), {})
"""


def test_an_example_fills_the_question_box():
    from ui.tabs.ai_query import _EXAMPLES, INPUT_KEY
    at = AppTest.from_string(_SCRIPT).run()
    short, full = _EXAMPLES[1]
    next(b for b in at.button if b.label == short).click().run()
    assert not at.exception
    assert at.text_area(key=INPUT_KEY).value == full


def test_every_example_has_a_button():
    from ui.tabs.ai_query import _EXAMPLES
    at = AppTest.from_string(_SCRIPT).run()
    labels = {b.label for b in at.button}
    assert {short for short, _ in _EXAMPLES} <= labels


def test_the_deprecated_component_is_gone():
    source = pathlib.Path("ui/tabs/ai_query.py").read_text(encoding="utf-8")
    assert "components.v1" not in source and "window.parent" not in source
