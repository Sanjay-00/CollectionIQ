"""
PeriodPair: the two monthly closings an investigation compares, and one
join of them on Loan No (the app's month join keeps this month's loans only,
so it can't see loans that left the running book).

Segment labels are worked out against this month (the reporting month), for
BOTH months, so a loan keeps one label: a March agreement is "2026-Q1" in
August and September alike. Executive labels follow the loan to the
executive holding it now (user decision 2026-10-23, utils.prev_held_now);
every other dimension uses each month's own value.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from config import INVESTIGATION_LABEL_DRIFT_WARN_PCT, INVESTIGATION_MIN_SEGMENT
from investigation.dimensions import (
    DIMENSIONS, _text, get_dimension, segment_labels, top_values, unavailable_reason,
)
from investigation.evidence import ToolError
from investigation.metrics import running_mask
from utils import loan_key

LEFT_BOOK = "Left the running book"

# Label kinds that depend on the row alone (not on last month's book or on
# who holds the loan now): a narrowed pair takes them from the pair it was
# narrowed from instead of working them out again. Rebuilt per scope, labels
# and their text cleaning were most of a 90k-loan deep-dive.
_ROW_WISE_KINDS = {"category", "fixed_bands", "vehicle_age", "quarter"}


@dataclass
class PeriodPair:
    prev: pd.DataFrame
    curr: pd.DataFrame
    prev_month: str
    curr_month: str
    _joined: pd.DataFrame | None = field(default=None, init=False, repr=False)
    _labels: dict = field(default_factory=dict, init=False, repr=False)
    _cache: dict = field(default_factory=dict, init=False, repr=False)
    # (pair narrowed from, its prev rows kept, its curr rows kept), as boolean arrays
    _origin: tuple | None = field(default=None, init=False, repr=False)

    def narrowed(self, keep_prev, keep_curr) -> "PeriodPair":
        """These months restricted to some rows, sharing this pair's row-wise work."""
        keep_prev, keep_curr = np.asarray(keep_prev, dtype=bool), np.asarray(keep_curr, dtype=bool)
        out = PeriodPair(self.prev[keep_prev], self.curr[keep_curr], self.prev_month, self.curr_month)
        out._origin = (self, keep_prev, keep_curr)
        return out

    def _from_origin(self, month: str, compute):
        """compute(origin) on the pair this one was narrowed from, cut to our rows."""
        origin, keep_prev, keep_curr = self._origin
        out = compute(origin).iloc[np.flatnonzero(keep_prev if month == "prev" else keep_curr)]   # keeps the dtype
        out.index = (self.prev if month == "prev" else self.curr).index
        return out

    def _curr_text(self, col: str) -> pd.Series:
        """This month's column as clean text (dimensions._text), worked out once."""
        key = ("curr_text", col)
        if key not in self._cache:
            self._cache[key] = (self._from_origin("curr", lambda o: o._curr_text(col)) if self._origin
                                else _text(self.curr[col]))
        return self._cache[key]

    def unavailable(self, dim_id: str) -> str | None:
        """unavailable_reason for these two months, worked out once per pair."""
        key = ("unavailable", dim_id)
        if key not in self._cache:
            dim = get_dimension(dim_id)
            text = self._curr_text(dim.columns[0]) if dim.columns[0] in self.curr.columns else None
            self._cache[key] = unavailable_reason(dim, self.prev, self.curr, curr_text=text)
        return self._cache[key]

    def labels(self, dim_id: str, month: str) -> pd.Series:
        """Segment labels for "prev" or "curr", aligned to that month's rows."""
        key = (dim_id, month)
        if key not in self._labels and self._origin and get_dimension(dim_id).kind in _ROW_WISE_KINDS:
            self._labels[key] = self._from_origin(month, lambda o: o.labels(dim_id, month))
        if key not in self._labels:
            dim = get_dimension(dim_id)
            top = top_values(dim, self.prev)
            if month == "curr":
                out = segment_labels(dim, self.curr, self.curr_month, ref=self.prev, top=top)
            elif dim.kind == "executive":
                owner = segment_labels(dim, self.curr, self.curr_month)
                held = dict(zip(self.curr["Loan No"], owner))
                out = self.prev["Loan No"].map(held).fillna(LEFT_BOOK).astype(object)
            else:
                out = segment_labels(dim, self.prev, self.curr_month, ref=self.prev, top=top)
            self._labels[key] = out
        return self._labels[key]

    def label_drift_pct(self, dim_id: str) -> float | None:
        """Of the loans in both months, the % whose label for this dimension
        differs between the two files. A stable attribute (zone, product,
        channel) should barely move; a high figure means the column changed
        meaning or filling between extracts. None with fewer than
        INVESTIGATION_MIN_SEGMENT loans in both months: a % of a handful of
        loans says nothing about the column."""
        key = ("drift", dim_id)
        if key not in self._cache:
            self._cache[key] = self._drift(dim_id)
        return self._cache[key]

    def masks(self, metric) -> dict:
        """{"prev" | "curr": (running, eligible, numerator)} as boolean arrays
        aligned to that month's rows, worked out once per pair (row-wise, so a
        narrowed pair cuts its origin's)."""
        key = ("masks", metric.id)
        if key not in self._cache:
            if self._origin:
                origin, keep_prev, keep_curr = self._origin
                full = origin.masks(metric)
                self._cache[key] = {"prev": tuple(m[keep_prev] for m in full["prev"]),
                                    "curr": tuple(m[keep_curr] for m in full["curr"])}
            else:
                self._cache[key] = {
                    month: tuple(np.asarray(f(df), dtype=bool) for f in (metric.running, metric.eligible, metric.numerator))
                    for month, df in (("prev", self.prev), ("curr", self.curr))}
        return self._cache[key]

    def _both_months(self) -> tuple[np.ndarray, np.ndarray]:
        """Row positions (prev, curr) of the loans in both months, matched
        pairwise. A narrowed pair keeps its origin's pairs whose two rows it
        both kept, renumbered to its own rows (no second match on Loan No)."""
        key = ("both_months",)
        if key not in self._cache:
            if self._origin:
                origin, keep_prev, keep_curr = self._origin
                p, c = origin._both_months()
                kept = keep_prev[p] & keep_curr[c]
                self._cache[key] = ((np.cumsum(keep_prev) - 1)[p[kept]], (np.cumsum(keep_curr) - 1)[c[kept]])
            else:
                at = pd.Index(self.curr["Loan No"]).get_indexer(self.prev["Loan No"])
                self._cache[key] = (np.flatnonzero(at >= 0), at[at >= 0])
        return self._cache[key]

    def _drift(self, dim_id: str) -> float | None:
        p, c = self._both_months()
        if len(p) < INVESTIGATION_MIN_SEGMENT:
            return None
        prev = self.labels(dim_id, "prev").to_numpy()[p].astype(str)
        curr = self.labels(dim_id, "curr").to_numpy()[c].astype(str)
        return 100 * float((prev != curr).mean())

    def require(self, dim_id: str) -> None:
        """Refuse, in plain words, a dimension these two files can't be cut
        by: a column missing from either month, or labels that changed for
        more than INVESTIGATION_LABEL_DRIFT_WARN_PCT of the same loans."""
        dim = get_dimension(dim_id)
        reason = self.unavailable(dim_id)
        if reason and reason != "only one value":
            raise ToolError(f"can't use {dim.label}: {reason}")
        drift = self.label_drift_pct(dim_id)
        if drift is not None and drift > INVESTIGATION_LABEL_DRIFT_WARN_PCT:
            raise ToolError(f"can't use {dim.label}: labels changed for {drift:.1f}% of the same loans between months")

    def loan_label(self, dim_id: str) -> pd.Series:
        """One label per loan for both months: this month's, or last month's
        for a loan that left the book. A loan that moved branch belongs to
        its branch now in both months, so a scope never reads a move as the
        loan leaving one branch and joining another. Worked out once per pair,
        like labels(): rebuilt on every scope, it was 70% of a 90k-loan review."""
        key = ("loan_label", dim_id)
        if key not in self._cache:
            curr = pd.Series(self.labels(dim_id, "curr").to_numpy(), index=self.curr["Loan No"].to_numpy())
            prev = pd.Series(self.labels(dim_id, "prev").to_numpy(), index=self.prev["Loan No"].to_numpy())
            prev_only = np.ones(len(self.prev), dtype=bool)
            prev_only[self._both_months()[0]] = False      # positions, not a second match on Loan No
            self._cache[key] = pd.concat([curr, prev[prev_only]])
        return self._cache[key].copy()

    def without_units(self, units: list[str]) -> "PeriodPair":
        """Both months without these branches, each month by its own file's
        Unit (trimmed, upper case): the same rule data_health uses to decide
        a branch is in one file only, so what is reported is what is removed."""
        if not units:
            return self
        drop = {str(u).strip().upper() for u in units}
        keep_p = ~self.prev["Unit"].astype(str).str.strip().str.upper().isin(drop)
        keep_c = ~self.curr["Unit"].astype(str).str.strip().str.upper().isin(drop)
        out = self.narrowed(keep_p.to_numpy(), keep_c.to_numpy())
        out.prev.attrs, out.curr.attrs = dict(self.prev.attrs), dict(self.curr.attrs)
        return out

    def scoped(self, scope: dict[str, str] | None) -> "PeriodPair":
        """Both months narrowed to the loans in each {dimension: label}, each
        loan by its loan_label. Refuses a dimension require() refuses, and a
        scope that matches nothing at all."""
        if not scope:
            return self
        keep_p = pd.Series(True, index=self.prev.index)
        keep_c = pd.Series(True, index=self.curr.index)
        for dim_id, label in scope.items():
            self.require(dim_id)
            want = str(label).strip().upper()
            per_loan = self.loan_label(dim_id).astype(str).str.upper() == want
            keep_p &= self.prev["Loan No"].map(per_loan).fillna(False).astype(bool).to_numpy()
            keep_c &= self.curr["Loan No"].map(per_loan).fillna(False).astype(bool).to_numpy()
        if not keep_p.any() and not keep_c.any():
            shown = ", ".join(f"{get_dimension(d).label} {v}" for d, v in scope.items())
            raise ToolError(f"no loans match {shown} in either month")
        return self.narrowed(keep_p.to_numpy(), keep_c.to_numpy())

    def joined(self) -> pd.DataFrame:
        """One row per loan across both months: in_prev / in_curr / left_book,
        each month's bucket (None when the loan isn't in that month) and
        running flag, and every dimension column (this month's value when the
        loan is in this month, else last month's)."""
        if self._joined is None:
            dim_cols = sorted({c for d in DIMENSIONS.values() for c in d.columns})

            def side(df: pd.DataFrame, tag: str) -> pd.DataFrame:
                cols = ["curr_bucket", *[c for c in dim_cols if c in df.columns]]
                out = df[["Loan No", *cols]].rename(columns={c: f"{tag}:{c}" for c in cols})
                out[f"{tag}:running"] = running_mask(df).to_numpy()
                return out

            j = side(self.prev, "p").merge(side(self.curr, "c"), on="Loan No", how="outer", indicator=True)
            in_prev = j["_merge"].isin(["both", "left_only"])
            in_curr = j["_merge"].isin(["both", "right_only"])
            out = pd.DataFrame({"Loan No": j["Loan No"], "in_prev": in_prev, "in_curr": in_curr,
                                "left_book": in_prev & ~in_curr})
            out["prev_bucket"] = j["p:curr_bucket"].astype(object).where(in_prev, None)
            out["curr_bucket"] = j["c:curr_bucket"].astype(object).where(in_curr, None)
            out["prev_running"] = j["p:running"].astype("boolean").fillna(False).astype(bool) & in_prev
            out["curr_running"] = j["c:running"].astype("boolean").fillna(False).astype(bool) & in_curr
            for c in dim_cols:
                cur = j[f"c:{c}"] if f"c:{c}" in j.columns else None
                pre = j[f"p:{c}"] if f"p:{c}" in j.columns else None
                if cur is not None and pre is not None:
                    out[c] = cur.astype(object).where(in_curr, pre.astype(object))
                elif cur is not None or pre is not None:
                    out[c] = (cur if cur is not None else pre).astype(object)
            self._joined = out.reset_index(drop=True)
        return self._joined


def make_pair(curr: pd.DataFrame, prev: pd.DataFrame, curr_month: str, prev_month: str) -> PeriodPair:
    """Check both months and normalise loan numbers (12345.0, " 12345" and
    "12345" are one loan). One row per loan per month."""
    for df, which in ((prev, "last month's"), (curr, "this month's")):
        if df is None or df.empty:
            raise ToolError(f"{which} file has no loans; an investigation needs two monthly closings")
        missing = [c for c in ("Loan No", "curr_bucket") if c not in df.columns]
        if missing:
            raise ToolError(f"{which} file has no {', '.join(missing)} column")
    if not prev_month or not curr_month or str(prev_month) >= str(curr_month):
        raise ToolError(f"last month ({prev_month}) must come before this month ({curr_month})")

    def clean(df: pd.DataFrame) -> pd.DataFrame:
        out = df.assign(**{"Loan No": loan_key(df["Loan No"])}).drop_duplicates("Loan No").reset_index(drop=True)
        out.attrs = dict(df.attrs)          # the upload's notes (renamed headers, duplicates) for data_health
        return out

    return PeriodPair(clean(prev), clean(curr), str(prev_month), str(curr_month))
