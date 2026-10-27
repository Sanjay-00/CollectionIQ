"""
Metrics Deep Investigation can explain. Each is a definition naming tested
functions, never a formula string to evaluate. v1 has one: the 30+ rate on
the app's own EMI bucket rule (utils.bucket_from_arrears_emi, carried as
curr_bucket), with loans that have no EMI due yet outside the base
(user decision, 2026-10-23).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd

from investigation.evidence import ToolError

ELIGIBLE_BUCKETS = ("STD", "1-30 DPD", "SMA-1", "SMA-2", "NPA")
THIRTY_PLUS_BUCKETS = ("SMA-1", "SMA-2", "NPA")


@dataclass(frozen=True)
class MetricDef:
    id: str
    label: str
    version: int
    running: Callable[[pd.DataFrame], pd.Series]
    eligible: Callable[[pd.DataFrame], pd.Series]
    numerator: Callable[[pd.DataFrame], pd.Series]


def running_mask(df: pd.DataFrame) -> pd.Series:
    """Loans in the running book (Loan Status RUN; every row when the column is absent)."""
    if "Loan Status" not in df.columns:
        return pd.Series(True, index=df.index)
    return df["Loan Status"].astype(str).str.strip().str.upper() == "RUN"


def _eligible(df: pd.DataFrame) -> pd.Series:
    return running_mask(df) & df["curr_bucket"].isin(ELIGIBLE_BUCKETS)


def _thirty_plus(df: pd.DataFrame) -> pd.Series:
    return _eligible(df) & df["curr_bucket"].isin(THIRTY_PLUS_BUCKETS)


METRICS: dict[str, MetricDef] = {
    "thirty_plus_rate": MetricDef(
        id="thirty_plus_rate", label="30+ (SMA-1 and worse: one or more EMIs overdue)", version=1,
        running=running_mask, eligible=_eligible, numerator=_thirty_plus,
    ),
}


def get_metric(metric_id: str) -> MetricDef:
    if metric_id not in METRICS:
        raise ToolError(f"unknown metric '{metric_id}'. Known: {', '.join(METRICS)}")
    return METRICS[metric_id]
