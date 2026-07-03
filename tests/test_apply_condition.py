"""Regression: _apply_condition must fail loudly (raise) on an op that doesn't
apply to a column's dtype, never silently return the full unfiltered DataFrame.

Bug: "contains" (or any op absent from the date/numeric ops dict) against a
date or numeric column used to fall through `if mask is not None else df` and
return df UNCHANGED - indistinguishable from "everything matched", the
"confidently wrong answer" class of bug this codebase otherwise guards against
(e.g. the malformed-count rejection logic in the old aggregation validator).
"""
import pandas as pd
import pytest

from agents.data_executor import _apply_condition


def _numeric_df():
    return pd.DataFrame({"SOH": [100.0, 200.0, 300.0]})


def _date_df():
    return pd.DataFrame({"Ag_Date": pd.to_datetime(["2024-01-01", "2024-06-01", "2025-01-01"])})


class TestNumericColumnInvalidOp:
    def test_contains_raises_instead_of_returning_unfiltered(self):
        with pytest.raises(ValueError, match="not valid for numeric column"):
            _apply_condition(_numeric_df(), {"column": "SOH", "op": "contains", "value": "100"})

    def test_valid_numeric_ops_still_work(self):
        out = _apply_condition(_numeric_df(), {"column": "SOH", "op": ">", "value": 150})
        assert len(out) == 2


class TestDateColumnInvalidOp:
    def test_contains_raises_instead_of_returning_unfiltered(self):
        with pytest.raises(ValueError, match="not valid for date column"):
            _apply_condition(_date_df(), {"column": "Ag_Date", "op": "contains", "value": "2024"})

    def test_in_raises_instead_of_returning_unfiltered(self):
        # "in" is also absent from the date ops dict - same failure class.
        with pytest.raises(ValueError, match="not valid for date column"):
            _apply_condition(_date_df(), {"column": "Ag_Date", "op": "in", "value": ["2024-01-01"]})

    def test_valid_date_ops_still_work(self):
        out = _apply_condition(_date_df(), {"column": "Ag_Date", "op": ">=", "value": "2024-06-01"})
        assert len(out) == 2
