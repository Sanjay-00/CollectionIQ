"""Guards a prompt-accuracy regression: the system prompt used to tell the LLM
"the system returns ALL columns by default" for loan_table queries when
display_columns is left empty -- false. It actually falls back to the curated
~40-column QUERY_DISPLAY_COLS (agents/data_executor.py), which excludes 44 real
source columns including business-relevant evidence fields (CoLending_Loans,
LGL_FLAG/LGL_DESCRIPTION, SegmentName, CUSTOMER_STATUS). Combined with "never
set display_columns just because a column is relevant", the LLM had no reason
to ever surface those columns even when a query's own filter concept depended
on one as its evidence (e.g. a co-lending query not showing CoLending_Loans).
"""
from agents.logical_planner import _build_full_system_prompt
from agents.data_executor import QUERY_DISPLAY_COLS


class TestDisplayColumnsPromptAccuracy:
    def test_prompt_does_not_claim_all_columns_returned_by_default(self):
        prompt = _build_full_system_prompt()
        assert "returns ALL columns by default" not in prompt

    def test_prompt_names_at_least_one_column_outside_the_default_view(self):
        # Spot-check evidence columns a legal/co-lending/segment query would need
        # but that QUERY_DISPLAY_COLS doesn't include -- the prompt must call
        # these out explicitly so the LLM knows to add them via display_columns.
        prompt = _build_full_system_prompt()
        for col in ("CoLending_Loans", "LGL_FLAG", "SegmentName"):
            assert col not in QUERY_DISPLAY_COLS, f"{col} is now in the default view -- update this test"
            assert col in prompt, f"prompt doesn't mention '{col}' as outside the default view"
