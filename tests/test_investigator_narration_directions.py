"""The Investigator's narration must not state a direction its table
contradicts (live, 2026-10-26: "Collection percentages are generally
improving ... significant drops in NPA and Delinquency for many" over a table
where Collection% fell in 26 of 50 branches and NPA% rose in 28). The number
check can't see this: the sentence has no numbers."""
import pandas as pd
import pytest

from investigator.llm import _build_preview, _checked, direction_facts


@pytest.fixture(scope="module")
def table():
    """The screenshot's table: sample data, Loan Status RUN, branches."""
    from investigator.steps import dimension_breakdown
    from utils import apply_filters, load_and_validate
    c, _ = load_and_validate(open("sample_data/Current_Month_Demo.xlsx", "rb"))
    p, _ = load_and_validate(open("sample_data/Previous_Month_Demo.xlsx", "rb"))
    return dimension_breakdown(apply_filters(c, "All", "All", "RUN", (), None), "branch",
                               df_prev=apply_filters(p, "All", "All", "RUN", (), None))


def _small():
    return pd.DataFrame({"Branch": ["A", "B", "C", "D", "E"],
                         "Collection%": [70.0, 72.0, 80.0, 85.0, 90.0],
                         "Δ Collection%": [-5.0, -3.0, -1.0, -2.0, 4.0],      # 4 of 5 fell
                         "NPA%": [20.0, 18.0, 15.0, 12.0, 10.0],
                         "Δ NPA%": [2.0, 1.0, 0.5, -1.0, 0.3]})               # 4 of 5 rose


class TestDirectionFacts:
    def test_counts_every_row_not_just_the_preview(self, table):
        facts = direction_facts(table)
        assert facts["Collection%"] == {"up": 24, "down": 26, "rows": 50}
        assert facts["NPA%"]["up"] == 28 and facts["NPA%"]["down"] == 20

    def test_the_preview_states_them_for_the_model(self, table):
        preview = _build_preview("dimension_breakdown", table)
        assert "Collection%: fell in 26 of 50 rows, rose in 24" in preview

    def test_a_table_without_changes_has_no_facts(self):
        assert direction_facts(pd.DataFrame({"Branch": ["A"], "NPA%": [3.0]})) == {}


class TestCheck:
    def test_the_live_sentence_is_dropped(self, table):
        live = ("Collection percentages are generally improving across units, with significant drops in NPA "
                "and Delinquency for many. However, Hard Bucket percentages remain elevated in several "
                "locations like Akkolkot and Mohol.")
        out = _checked(live, _build_preview("dimension_breakdown", table), direction_facts(table))
        assert "improving" not in out and "drops in NPA" not in out
        assert "Hard Bucket percentages remain elevated" in out          # a level statement, not a change: kept

    def test_a_true_direction_is_kept(self):
        df = _small()
        out = _checked("Collection% fell in most branches, while NPA% rose for most.",
                       _build_preview("dimension_breakdown", df), direction_facts(df))
        assert "fell in most branches" in out

    @pytest.mark.parametrize("sentence", [
        "Collection is improving across branches.",                       # improving Collection% = up: 1 of 5
        "NPA has declined in most branches.",                             # NPA down: 1 of 5
        "NPA is getting better broadly.",                                 # better NPA = down
        "Delinquency rose sharply.",                                      # no Delinquency change column: unknown
    ])
    def test_wrong_or_unsupported_directions_are_dropped(self, sentence):
        df = _small()
        out = _checked(sentence, _build_preview("dimension_breakdown", df), direction_facts(df))
        if "Delinquency" in sentence:
            assert out == sentence                                        # no column to check: left alone
        else:
            assert out == "" or "removed" in out

    def test_some_needs_only_a_few(self):
        df = _small()
        out = _checked("Collection improved in some branches.", _build_preview("dimension_breakdown", df),
                       direction_facts(df))
        assert "improved in some branches" in out                         # 1 of 5 = a fifth

    def test_comparisons_between_units_are_not_changes(self):
        df = _small()
        out = _checked("A has the lowest Collection% and the highest NPA%.",
                       _build_preview("dimension_breakdown", df), direction_facts(df))
        assert out.startswith("A has the lowest Collection%")

    def test_no_facts_means_no_direction_check(self):
        assert _checked("Collection is improving.", "Matching rows: 3", None) == "Collection is improving."
