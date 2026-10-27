"""
The dimensions Deep Investigation may slice a metric by (spec 3.5): which
file column, how it is banded, and whether it is scanned across the whole
book or only used to drill into one segment (executive and dealer have too
many values for a book-wide scan). Blank values are their own segment,
"Not recorded", so every split stays exhaustive and adds up.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from investigation.evidence import ToolError

NOT_RECORDED = "Not recorded"
OTHER = "Other"

_INF = math.inf
_TENURE = ((12, "≤12"), (24, "13-24"), (36, "25-36"), (48, "37-48"), (_INF, "49+"))
_FLEET = ((1, "1"), (4, "2-4"), (_INF, "5+"))
_VEHICLE_AGE = ((2, "0-2"), (5, "3-5"), (8, "6-8"), (_INF, "9+"))
_QUINTILES = ("Q1 (lowest)", "Q2", "Q3", "Q4", "Q5 (highest)")
_RECENT_MONTHS = 6          # the agreement months up to the reporting month shown one by one


@dataclass(frozen=True)
class Dimension:
    id: str
    label: str
    columns: tuple[str, ...]
    kind: str               # category | top_n | quarter | fixed_bands | quantile_bands | vehicle_age | executive
    scan: bool
    top_n: int | None = None
    bands: tuple[tuple[float, str], ...] | None = None


def _d(id_, label, columns, kind, scan=True, top_n=None, bands=None) -> Dimension:
    return Dimension(id_, label, tuple(columns) if isinstance(columns, (list, tuple)) else (columns,), kind, scan,
                     top_n, bands)


DIMENSIONS: dict[str, Dimension] = {d.id: d for d in (
    _d("zone", "Zone", "Zone", "category"),
    _d("region", "Region", "RegionName", "category"),
    _d("branch", "Branch", "Unit", "category"),
    _d("product", "Product", "SegmentName", "category"),
    _d("channel", "Channel", "CHANNEL", "category"),
    _d("scheme", "Scheme", "scheme", "top_n", top_n=20),
    _d("make", "Vehicle make", "Make", "top_n", top_n=25),
    _d("vintage", "Agreement period", "Ag_Date", "quarter"),
    _d("tenure", "Tenure (months)", "Tenure", "fixed_bands", bands=_TENURE),
    _d("loan_amount", "Loan amount", "Loan Amount", "quantile_bands"),
    _d("vehicle_age", "Vehicle age", "Year Of Manufacture", "vehicle_age", bands=_VEHICLE_AGE),
    _d("due_date", "EMI date", "Due Dt", "category"),
    _d("fleet", "Loans per customer", "Customer Loan Count", "fixed_bands", bands=_FLEET),
    _d("nach", "NACH status", "NACHStatus", "category"),
    _d("fuel", "Fuel", "FUEL_TYPE", "category"),
    _d("executive", "Executive", ["MNT NAME", "Unit"], "executive", scan=False),
    _d("dealer", "Source / dealer", "SRC Name", "top_n", scan=False, top_n=30),
)}


def get_dimension(dim_id: str) -> Dimension:
    if dim_id not in DIMENSIONS:
        raise ToolError(f"unknown dimension '{dim_id}'. Known: {', '.join(DIMENSIONS)}")
    return DIMENSIONS[dim_id]


def _text(s: pd.Series) -> pd.Series:
    """Values as clean text (5.0 -> "5" in a number column), blanks as
    NOT_RECORDED. Text columns stay text: converting them was most of an
    investigation's time and merged codes like "007" and "7"."""
    if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
        num = pd.to_numeric(s, errors="coerce").astype(float)
        whole = num.notna() & np.isfinite(num) & (num == num.round())
        out = s.astype(object).where(~whole, num.where(whole).astype("Int64").astype(str))
    else:
        out = s.astype(object)
    out = out.astype(str).str.strip()
    blank = s.isna() | out.isin(["", "nan", "None", "NaT", "<NA>"])
    return out.mask(blank, NOT_RECORDED)


def _band(values: pd.Series, bands) -> pd.Series:
    v = pd.to_numeric(values, errors="coerce")
    labels = pd.Series(NOT_RECORDED, index=values.index, dtype=object)
    lower = -_INF
    for upper, label in bands:
        labels[(v > lower) & (v <= upper)] = label
        lower = upper
    return labels


def top_values(dim: Dimension, ref: pd.DataFrame) -> list[str] | None:
    """The reference month's most frequent values, for a top_n dimension."""
    if dim.kind != "top_n" or ref is None or dim.columns[0] not in ref.columns:
        return None
    counts = _text(ref[dim.columns[0]]).value_counts()
    counts = counts[counts.index != NOT_RECORDED]
    return list(counts.index[: dim.top_n])


def segment_labels(dim: Dimension, df: pd.DataFrame, as_of: str, ref: pd.DataFrame | None = None,
                   top: list[str] | None = None) -> pd.Series:
    """One segment label per row of df. ref: the reference (previous) month,
    for quintile edges and the top_n list when `top` isn't given."""
    col = dim.columns[0]
    if dim.kind == "executive":
        name, unit = _text(df[dim.columns[0]]), _text(df[dim.columns[1]])
        return (name + " (" + unit + ")").mask(name == NOT_RECORDED, NOT_RECORDED)
    if dim.kind == "category":
        return _text(df[col])
    if dim.kind == "top_n":
        keep = top if top is not None else top_values(dim, ref if ref is not None else df)
        labels = _text(df[col])
        return labels.where(labels.isin(keep) | (labels == NOT_RECORDED), OTHER)
    if dim.kind == "fixed_bands":
        return _band(df[col], dim.bands)
    if dim.kind == "vehicle_age":
        age = (pd.Period(as_of, "M").year - pd.to_numeric(df[col], errors="coerce")).clip(lower=0)
        return _band(age, dim.bands)
    if dim.kind == "quantile_bands":
        base = pd.to_numeric((ref if ref is not None else df)[col], errors="coerce").dropna()
        v = pd.to_numeric(df[col], errors="coerce")
        if base.empty:
            return pd.Series(NOT_RECORDED, index=df.index, dtype=object)
        edges = np.quantile(base, [0.2, 0.4, 0.6, 0.8])
        idx = np.searchsorted(edges, v.fillna(0).to_numpy(), side="right")
        return pd.Series([_QUINTILES[i] for i in idx], index=df.index, dtype=object).mask(v.isna(), NOT_RECORDED)
    if dim.kind == "quarter":
        when = pd.to_datetime(df[col], errors="coerce")
        month = when.dt.to_period("M")
        recent = month >= pd.Period(as_of, "M") - (_RECENT_MONTHS - 1)
        q = when.dt.to_period("Q")
        labels = pd.Series(
            np.where(recent, month.astype(str), q.dt.year.astype("Int64").astype(str) + "-Q" + q.dt.quarter.astype("Int64").astype(str)),
            index=df.index, dtype=object)
        return labels.mask(when.isna(), NOT_RECORDED)
    raise ToolError(f"dimension kind '{dim.kind}' is not supported")


def unavailable_reason(dim: Dimension, prev: pd.DataFrame, curr: pd.DataFrame,
                       curr_text: pd.Series | None = None) -> str | None:
    """Why this dimension can't split these two months, or None when it can.
    curr_text: this month's first column already as _text (a pair caches it)."""
    for frame, which in ((prev, "last month's file"), (curr, "this month's file")):
        for col in dim.columns:
            if col not in frame.columns:
                return f"no {col} column in {which}"
    values = set((_text(curr[dim.columns[0]]) if curr_text is None else curr_text).unique()) - {NOT_RECORDED}
    return "only one value" if len(values) < 2 else None
