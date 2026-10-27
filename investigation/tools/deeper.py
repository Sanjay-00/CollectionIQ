"""
Going deeper (spec 7.6 to 7.8): the rival-explanation test, a drill inside
one segment, and the loans behind a finding.
"""
from __future__ import annotations

import pandas as pd

from config import INVESTIGATION_MIN_SEGMENT_DRILL
from investigation.dimensions import get_dimension
from investigation.evidence import Evidence, EvidenceStore, ToolError
from investigation.metrics import THIRTY_PLUS_BUCKETS, get_metric
from investigation.panel import PeriodPair
from investigation.tools.change import material_change
from investigation.tools.mix import mix_rate, scan_summary

SAMPLE_KINDS = ("thirty_plus_now", "rolled_in", "cured")
SAMPLE_MAX = 50
DOMINANT_SHARE = 0.8          # a segment this much of a level can't be told apart from the level


def controlled_decompose(store: EvidenceStore, pair: PeriodPair, dimension: str, segment: str, control: str,
                         metric_id: str = "thirty_plus_rate", scope: dict | None = None) -> Evidence:
    """Does `segment` of `dimension` keep the change's direction INSIDE each
    level of `control` (spec 7.6: its rate moves the way the whole change did,
    beyond its noise band)? A cause persists inside its rival's levels; a
    pattern that only borrowed it (vintage standing in for product)
    disappears. persists: in levels holding at least half of the tested
    loans. Levels where the segment is over DOMINANT_SHARE of the level can't
    separate it from the level and are left out; fewer than two tested
    levels is no test (ToolError). Descriptive, not proof of cause."""
    metric = get_metric(metric_id)
    p = pair.scoped(scope)
    p.require(control)
    direction = material_change(p, metric)[0]
    want = str(segment).strip().upper()
    # Labels from the full pair, the same ones mix_rate's re-scope uses below
    # (banded / top-n dimensions must name the same loans in both places).
    in_scope = pair.curr["Loan No"].isin(set(p.curr["Loan No"])).to_numpy()
    mine = in_scope & metric.eligible(pair.curr).to_numpy() & \
        (pair.labels(dimension, "curr").astype(str).str.upper() == want).to_numpy()
    sizes = pair.labels(control, "curr")[mine].astype(str).value_counts()
    levels = [lv for lv, n in sizes.items() if n >= INVESTIGATION_MIN_SEGMENT_DRILL]
    rows = []
    for level in levels:
        try:
            ev = mix_rate(store, pair, dimension, metric_id, {**(scope or {}), control: level},
                          min_size=INVESTIGATION_MIN_SEGMENT_DRILL)
        except ToolError:
            continue                              # e.g. the dimension has one value inside this level
        seg = ev.tables["segments"]
        hit = seg[seg["segment"].astype(str).str.upper() == want]
        if hit.empty or hit.iloc[0]["s1"] > DOMINANT_SHARE:
            continue
        r = hit.iloc[0]
        change, noise = r["rate_change_pp"], r["noise_pp"]
        keeps = change is not None and noise is not None and change * direction > 0 and abs(change) > noise
        rows.append({"level": level, "loans": int(r["n1"]), "keeps_direction": bool(keeps),
                     "rate_change_pp": change, "evidence_id": ev.result_id})
    if len(rows) < 2:
        raise ToolError(f"can't test {segment} inside {get_dimension(control).label}: it needs at least two "
                        f"{get_dimension(control).label} groups holding enough of its loans")
    tested = sum(r["loans"] for r in rows)
    persisting = sum(r["loans"] for r in rows if r["keeps_direction"])
    summary = {"dimension": dimension, "segment": segment, "control": control, "levels": rows,
               "tested_loans": tested, "persisting_loans": persisting, "persists": persisting >= tested / 2}
    return store.add("controlled_decompose", {"dimension": dimension, "segment": segment, "control": control,
                                              "scope": dict(scope or {})}, summary)


def drill(store: EvidenceStore, pair: PeriodPair, dimension: str, segment: str,
          metric_id: str = "thirty_plus_rate", scope: dict | None = None) -> Evidence:
    """Scan every dimension (drill-only ones too: executive, dealer) inside
    one segment, with the drill's smaller minimum segment size."""
    inner = {**(scope or {}), dimension: segment}
    summary, checks = scan_summary(store, pair, metric_id, inner, include_drill_only=True,
                                   min_size=INVESTIGATION_MIN_SEGMENT_DRILL, skip=(dimension,))
    summary["parent"] = [dimension, segment]
    return store.add("drill", {"metric_id": metric_id, "dimension": dimension, "segment": segment,
                               "scope": dict(scope or {})}, summary, checks=checks)


def sample_loans(store: EvidenceStore, pair: PeriodPair, scope: dict, kind: str, n: int = SAMPLE_MAX) -> Evidence:
    """The loans behind a finding, largest SOH first, for the screen and the
    download. The summary holds counts only; customer fields stay in the table."""
    if kind not in SAMPLE_KINDS:
        raise ToolError(f"unknown kind of loans '{kind}'. Known: {', '.join(SAMPLE_KINDS)}")
    if int(n) < 1:
        raise ToolError("ask for at least 1 loan")
    p = pair.scoped(scope)
    j = p.joined().set_index("Loan No")
    curr = p.curr[get_metric("thirty_plus_rate").running(p.curr).to_numpy()]
    prev_b = curr["Loan No"].map(j["prev_bucket"].where(j["prev_running"]))
    now_30 = curr["curr_bucket"].isin(THIRTY_PLUS_BUCKETS)
    was_30 = prev_b.isin(THIRTY_PLUS_BUCKETS)
    pick = {"thirty_plus_now": now_30, "rolled_in": prev_b.notna() & ~was_30 & now_30,
            "cured": was_30 & curr["curr_bucket"].isin(["STD", "1-30 DPD"])}[kind]
    rows = curr[pick.to_numpy()].assign(**{"Last month": prev_b[pick].to_numpy()})
    if "SOH" in rows.columns:
        rows = rows.assign(_soh=pd.to_numeric(rows["SOH"], errors="coerce")).sort_values("_soh", ascending=False)
    cols = [c for c in ("Loan No", "Unit", "MNT NAME", "Last month", "curr_bucket", "SOH", "Cust Name", "Cust Mob No")
            if c in rows.columns]
    shown = rows[cols].head(min(int(n), SAMPLE_MAX)).rename(columns={"curr_bucket": "This month"})
    summary = {"kind": kind, "scope": dict(scope or {}), "count_all": int(len(rows)), "shown": int(len(shown))}
    return store.add("sample_loans", {"kind": kind, "scope": dict(scope or {}), "n": n}, summary,
                     {"loans": shown.reset_index(drop=True)})
