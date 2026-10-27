"""The approved slicing dimensions (spec 3.5 and plan Task 2)."""
import pandas as pd
import pytest

from investigation.dimensions import (
    DIMENSIONS, NOT_RECORDED, get_dimension, segment_labels, top_values, unavailable_reason,
)
from investigation.evidence import ToolError

ORDER = ["zone", "region", "branch", "product", "channel", "scheme", "make", "vintage", "tenure",
         "loan_amount", "vehicle_age", "due_date", "fleet", "nach", "fuel", "executive", "dealer"]


def test_registry_matches_the_spec():
    assert list(DIMENSIONS) == ORDER
    assert [d for d, dim in DIMENSIONS.items() if not dim.scan] == ["executive", "dealer"]
    with pytest.raises(ToolError):
        get_dimension("agency")


def test_blank_values_become_not_recorded():
    df = pd.DataFrame({"FUEL_TYPE": ["DIESEL", None, ""]})
    assert list(segment_labels(get_dimension("fuel"), df, "2026-09")) == ["DIESEL", NOT_RECORDED, NOT_RECORDED]


def test_tenure_and_fleet_bands():
    t = pd.DataFrame({"Tenure": [12, 13, 48, 60]})
    assert list(segment_labels(get_dimension("tenure"), t, "2026-09")) == ["≤12", "13-24", "37-48", "49+"]
    f = pd.DataFrame({"Customer Loan Count": [1, 3, 7]})
    assert list(segment_labels(get_dimension("fleet"), f, "2026-09")) == ["1", "2-4", "5+"]


def test_loan_amount_bands_use_last_months_edges():
    ref = pd.DataFrame({"Loan Amount": range(1, 101)})
    cur = pd.DataFrame({"Loan Amount": [1000, 1]})
    assert list(segment_labels(get_dimension("loan_amount"), cur, "2026-09", ref=ref)) == ["Q5 (highest)", "Q1 (lowest)"]


def test_vintage_recent_months_then_quarters():
    df = pd.DataFrame({"Ag_Date": pd.to_datetime(["2026-09-10", "2026-04-01", "2026-03-31", "2025-11-05"])})
    assert list(segment_labels(get_dimension("vintage"), df, "2026-09")) == ["2026-09", "2026-04", "2026-Q1", "2025-Q4"]


def test_vehicle_age_from_year_of_manufacture():
    df = pd.DataFrame({"Year Of Manufacture": [2026, 2022, 2018, 2010, "x"]})
    assert list(segment_labels(get_dimension("vehicle_age"), df, "2026-09")) == ["0-2", "3-5", "6-8", "9+", NOT_RECORDED]


def test_top_n_groups_the_rest_as_other():
    ref = pd.DataFrame({"Make": [f"M{i}" for i in range(30) for _ in range(30 - i)]})   # M0 most frequent
    top = top_values(get_dimension("make"), ref)
    assert len(top) == 25 and top[0] == "M0"
    labels = segment_labels(get_dimension("make"), pd.DataFrame({"Make": [f"M{i}" for i in range(30)]}), "2026-09", top=top)
    assert (labels == "Other").sum() == 5


def test_executive_label_is_name_and_branch():
    df = pd.DataFrame({"MNT NAME": ["RAJ"], "Unit": ["AKLUJ"]})
    assert list(segment_labels(get_dimension("executive"), df, "2026-09")) == ["RAJ (AKLUJ)"]


def test_missing_or_flat_dimension_is_unavailable():
    prev = pd.DataFrame({"CHANNEL": ["A", "B"]})
    curr = pd.DataFrame({"CHANNEL": ["A", "A"], "FUEL_TYPE": ["D", "P"]})
    assert "no FUEL_TYPE column" in unavailable_reason(get_dimension("fuel"), prev, curr)
    assert unavailable_reason(get_dimension("channel"), prev, curr) == "only one value"
    assert unavailable_reason(get_dimension("channel"), curr, prev) is None
