"""
Where a change sits: one dimension's mix versus rate split (spec 7.4), and a
scan of every dimension (spec 7.5). Attribution is descriptive, never causal,
and splits by different dimensions overlap: they are never added together.
"""
from __future__ import annotations

import math

import pandas as pd

from config import (
    INVESTIGATION_LABEL_DRIFT_WARN_PCT, INVESTIGATION_MIN_SEGMENT, INVESTIGATION_NO_CHANGE_PP, INVESTIGATION_NOISE_SE,
    INVESTIGATION_RECONCILE_TOL_PP, INVESTIGATION_TOP_SEGMENTS,
)
from investigation.dimensions import DIMENSIONS, OTHER, get_dimension
from investigation.evidence import Check, Evidence, EvidenceStore, ToolError
from investigation.metrics import get_metric
from investigation.panel import LEFT_BOOK, PeriodPair
from investigation.tools.change import material_change, pair_counts
from investigation.tools.noise import rate_noise_pp

SMALLER = "Smaller segments"


def _month_rows(loans: pd.Series, labels: pd.Series, eligible: pd.Series, numerator: pd.Series) -> pd.DataFrame:
    return pd.DataFrame({"loan": loans[eligible].to_numpy(), "segment": labels[eligible].astype(str).to_numpy(),
                         "k": numerator[eligible].astype(int).to_numpy()})


def _segments(prev: pd.DataFrame, curr: pd.DataFrame, min_size: int) -> pd.DataFrame:
    """Both months' counts per segment (prev / curr: one eligible loan per
    row: loan, segment, k = in the numerator), smaller segments summed into
    one row, then shares, rates and the four terms of the change (in % points).

    Noise bands are paired: most loans are the same loans seen twice, so a
    loan in the same segment both months adds noise only if it crossed into
    or out of the numerator (b + c below); loans in one month only are
    independent draws. Treating the months as independent samples overstated
    noise so much that a planted 2.5-point rise in 4,000 loans read as noise."""
    both = prev.merge(curr, on=["loan", "segment"], suffixes=("0", "1"))
    t = pd.concat([
        prev.groupby("segment").agg(n0=("k", "size"), k0=("k", "sum")),
        curr.groupby("segment").agg(n1=("k", "size"), k1=("k", "sum")),
        both.assign(b=both["k0"] > both["k1"], c=both["k1"] > both["k0"]).groupby("segment")
            .agg(m=("b", "size"), b=("b", "sum"), c=("c", "sum")),
    ], axis=1).fillna(0).astype(int)
    small = t[["n0", "n1"]].max(axis=1) < min_size
    if small.any():
        t = pd.concat([t[~small], t[small].sum().to_frame(SMALLER).T])
    N0, N1 = t["n0"].sum(), t["n1"].sum()
    U0, U1 = N0 - len(both), N1 - len(both)          # loans not in the same segment both months
    # Mix, new and gone terms are measured against last month's book rate R0:
    # a segment whose share grows but whose rate sits below the book pulls
    # the rate DOWN (review C2: an undemeaned (s1 - s0) * r0 called it a
    # driver of a rise). They still add up exactly, since shares net to zero.
    R0 = 100 * t["k0"].sum() / N0
    rows = []
    for seg, r in t.iterrows():
        s0, s1 = r["n0"] / N0, r["n1"] / N1
        r0 = 100 * r["k0"] / r["n0"] if r["n0"] else None
        r1 = 100 * r["k1"] / r["n1"] if r["n1"] else None
        mix = rate = new = gone = 0.0
        noise = None
        if r0 is not None and r1 is not None:
            mix, rate = (s1 - s0) * (r0 - R0), s1 * (r1 - r0)
            noise = rate_noise_pp(r["n0"], r["k0"], r["n1"], r["k1"], r["m"], r["b"], r["c"])
        elif r1 is not None:
            new = s1 * (r1 - R0)
        elif r0 is not None:
            gone = -s0 * (r0 - R0)
        change = r1 - r0 if r0 is not None and r1 is not None else None
        # A pure mix move has no rate change at all, so a segment is notable
        # when its rate OR its share of loans moved beyond noise. Shares only
        # move through loans that aren't in the same segment both months.
        share_noise = INVESTIGATION_NOISE_SE * 100 * math.sqrt(s0 * (1 - s0) * U0 / N0 ** 2 + s1 * (1 - s1) * U1 / N1 ** 2)
        rate_out = bool(change is not None and noise is not None and abs(change) > noise)
        share_out = abs(s1 - s0) * 100 > share_noise
        rows.append({
            "segment": seg, "n0": int(r["n0"]), "k0": int(r["k0"]), "n1": int(r["n1"]), "k1": int(r["k1"]),
            "s0": s0, "s1": s1, "r0": r0, "r1": r1, "rate_change_pp": change,
            "mix_pp": mix, "rate_pp": rate, "new_pp": new, "gone_pp": gone,
            "contribution_pp": mix + rate + new + gone, "noise_pp": noise,
            "share_change_pp": (s1 - s0) * 100, "share_noise_pp": share_noise,
            "rate_outside_noise": rate_out, "share_outside_noise": share_out, "outside_noise": rate_out or share_out,
        })
    rows.sort(key=lambda r: -abs(r["contribution_pp"]))
    out = pd.DataFrame(rows)
    for col in ("r0", "r1", "rate_change_pp", "noise_pp"):     # undefined stays None, never NaN or 0
        out[col] = pd.Series([r[col] for r in rows], dtype=object)
    return out


def _top(seg: pd.DataFrame) -> list[dict]:
    """The segments to cite: real segments only (not the leftover "Smaller
    segments", "Other" or "Left the running book" groups); those beyond their
    fair share first, then those outside noise, then by size of contribution."""
    real = seg[~seg["segment"].isin([SMALLER, OTHER, LEFT_BOOK])]
    order = real.assign(_b=~real["beyond_share"], _o=~real["outside_noise"], _e=-real["excess_pp"],
                        _a=-real["contribution_pp"].abs()).sort_values(["_b", "_o", "_e", "_a"])
    return [{"segment": r["segment"], "contribution_pp": r["contribution_pp"], "rate_change_pp": r["rate_change_pp"],
             "mix_pp": r["mix_pp"], "rate_pp": r["rate_pp"], "new_pp": r["new_pp"], "gone_pp": r["gone_pp"],
             "outside_noise": bool(r["outside_noise"]),
             "beyond_share": bool(r["beyond_share"]), "excess_pp": float(r["excess_pp"]),
             "n1": int(r["n1"])} for _, r in order.head(INVESTIGATION_TOP_SEGMENTS).iterrows()]


def mix_rate(store: EvidenceStore, pair: PeriodPair, dimension: str, metric_id: str = "thirty_plus_rate",
             scope: dict | None = None, min_size: int | None = None) -> Evidence:
    """The change in the metric split by one dimension's segments into mix,
    rate, new-segment and gone-segment terms that add up to the change."""
    metric, dim = get_metric(metric_id), get_dimension(dimension)
    p = pair.scoped(scope)
    reason = p.unavailable(dim.id)
    if reason:
        raise ToolError(f"can't split by {dim.label}: {reason}")
    min_size = INVESTIGATION_MIN_SEGMENT if min_size is None else min_size
    (_, e0, y0), (_, e1, y1) = p.masks(metric)["prev"], p.masks(metric)["curr"]
    prev_c = _month_rows(p.prev["Loan No"], p.labels(dimension, "prev"), pd.Series(e0, index=p.prev.index),
                         pd.Series(y0, index=p.prev.index))
    curr_c = _month_rows(p.curr["Loan No"], p.labels(dimension, "curr"), pd.Series(e1, index=p.curr.index),
                         pd.Series(y1, index=p.curr.index))
    tot_p, tot_c = pair_counts(p, metric, "prev"), pair_counts(p, metric, "curr")
    if not tot_p["eligible"] or not tot_c["eligible"]:
        raise ToolError("no eligible loans in one of the months for this scope")
    total = tot_c["rate"] - tot_p["rate"]
    seg = _beyond_share(_segments(prev_c, curr_c, min_size), total)
    parts = {k: float(seg[k].sum()) for k in ("mix_pp", "rate_pp", "new_pp", "gone_pp")}
    summary = {"dimension": dimension, "label": dim.label, "scope": dict(scope or {}), "total_change_pp": total,
               **parts, "top": _top(seg)}
    checks = [
        Check("components_add_up", abs(sum(parts.values()) - total) <= INVESTIGATION_RECONCILE_TOL_PP,
              f"{sum(parts.values()):+.4f} vs {total:+.4f}"),
        Check("shares_sum_to_one", abs(seg["s0"].sum() - 1) < 1e-9 and abs(seg["s1"].sum() - 1) < 1e-9),
        Check("counts_match_totals", int(seg["n0"].sum()) == tot_p["eligible"] and int(seg["k0"].sum()) == tot_p["numerator"]
              and int(seg["n1"].sum()) == tot_c["eligible"] and int(seg["k1"].sum()) == tot_c["numerator"]),
    ]
    return store.add("mix_rate", {"metric_id": metric_id, "dimension": dimension, "scope": dict(scope or {}),
                                  "min_size": min_size}, summary, {"segments": seg}, checks)


def _beyond_share(seg: pd.DataFrame, total: float) -> pd.DataFrame:
    """Each segment's move beyond its fair share of the change (s1 * total),
    less what noise alone would move it (s1 * rate band + share band at the
    segment's rate): excess_pp. beyond_share: that excess is positive and, when
    the total moved, in the total's direction. A big segment that only rose
    with everything else isn't beyond its share (review: "fleet 1" passed as
    a driver), and a dimension with many wobbling segments scores nothing."""
    excess, beyond = [], []
    for r in seg.itertuples():
        rate_ref = r.r0 if r.r0 is not None else r.r1
        band = (r.s1 * r.noise_pp if r.noise_pp is not None else 0.0) + r.share_noise_pp / 100 * (rate_ref or 0.0)
        gap = r.contribution_pp - r.s1 * total
        e = max(0.0, abs(gap) - band)
        excess.append(e)
        beyond.append(bool(e > 0 and (abs(total) < INVESTIGATION_NO_CHANGE_PP or gap * total > 0)))
    return seg.assign(excess_pp=excess, beyond_share=beyond)


def _concentration(seg: pd.DataFrame, total: float) -> float:
    """100 * sum of each segment's move beyond its fair share and noise
    (excess_pp) / (2 * |total|). Without the noise allowance a dimension with
    many segments wins on wobble alone (review: 146 noisy branches outranked a
    planted product cause)."""
    return float(100 * seg["excess_pp"].sum() / (2 * abs(total)))


def scan_dimensions(store: EvidenceStore, pair: PeriodPair, metric_id: str = "thirty_plus_rate",
                    scope: dict | None = None, include_drill_only: bool = False, min_size: int | None = None,
                    skip: tuple[str, ...] = ()) -> Evidence:
    summary, checks = scan_summary(store, pair, metric_id, scope, include_drill_only, min_size, skip)
    return store.add("scan_dimensions", {"metric_id": metric_id, "scope": dict(scope or {})}, summary, checks=checks)


def scan_summary(store: EvidenceStore, pair: PeriodPair, metric_id: str, scope: dict | None,
                 include_drill_only: bool = False, min_size: int | None = None,
                 skip: tuple[str, ...] = ()) -> tuple[dict, list[Check]]:
    """mix_rate on every scannable dimension (each its own citable result),
    ranked by concentration (_concentration): how far the change sits from an
    even spread by segment size, beyond noise. 0 means every segment moved in
    proportion to its size (the dimension says nothing); the planted cause
    scores far higher. Coverage (the share of
    the change in the cited segments) is reported but can't rank: a dimension
    with a few huge segments "covers" any change. When the total barely moved,
    both mean nothing, so the ranking is by the largest notable segment move:
    offsetting moves under a flat total still surface."""
    metric = get_metric(metric_id)
    p = pair.scoped(scope)
    tot_p, tot_c = pair_counts(p, metric, "prev"), pair_counts(p, metric, "curr")
    if not tot_p["eligible"] or not tot_c["eligible"]:
        raise ToolError("no eligible loans in one of the months for this scope")
    total, _, material = material_change(p, metric)
    ranked, unavailable, children = [], [], []
    for dim in DIMENSIONS.values():
        if (not dim.scan and not include_drill_only) or dim.id in skip:
            continue
        reason = p.unavailable(dim.id)
        drift = None if reason else p.label_drift_pct(dim.id)
        if drift is not None and drift > INVESTIGATION_LABEL_DRIFT_WARN_PCT:
            reason = f"labels changed for {drift:.1f}% of the same loans between months"
        if reason:
            unavailable.append({"dimension": dim.id, "reason": reason})
            continue
        ev = mix_rate(store, pair, dim.id, metric_id, scope, min_size=min_size)
        children.append(ev)
        s = ev.summary
        cited = [t for t in s["top"] if t["beyond_share"]]
        moves = [t["excess_pp"] for t in cited]
        seg = ev.tables["segments"]
        ranked.append({
            "dimension": dim.id, "label": dim.label, "evidence_id": ev.result_id,
            "concentration_pct": _concentration(seg, total) if material else None,
            "coverage_pct": 100 * sum(t["contribution_pp"] for t in cited) / total if material else None,
            "mix_pp": s["mix_pp"], "rate_pp": s["rate_pp"], "top": s["top"],
            "_key": max(moves, default=0.0),
        })
    if material:
        ranked.sort(key=lambda r: (-r["concentration_pct"], -r["_key"]))
    else:
        ranked.sort(key=lambda r: -r["_key"])
    for r in ranked:
        del r["_key"]
    summary = {"metric": metric_id, "scope": dict(scope or {}), "total_change_pp": total, "material": material,
               "ranked": ranked, "unavailable": unavailable}
    return summary, [Check("every_breakdown_reconciles", all(c.checks_passed for c in children))]
