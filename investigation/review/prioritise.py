"""
Which regions stand out, in three separate lists (spec 5.3-5.5), each
compared like for like with the rest of the portfolio (the region itself
left out of its own benchmark):

  collection_shortfall  this month's collection against what the rest of the
                        book collects from loans in the same opening bucket
  excess_30plus         30+ loans and SOH against the rest of the book's 30+
                        rate for loans of the same agreement vintage
  worsening             the 30+ rate's change, against the rest of the book's change

No weights across lists. A region on two or more is flagged. Every candidate
is an Evidence record with its strata table and self-checks; one whose checks
fail is never listed. Estimates, not recoverable cash or avoidable loss.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field

import pandas as pd

from config import (
    INVESTIGATION_NOISE_SE, REVIEW_FDR, REVIEW_MAX_UNMATCHED_PCT, REVIEW_MIN_ACCOUNTS, REVIEW_MIN_PTS, REVIEW_MIN_RS,
)
from investigation.evidence import Check, EvidenceStore, ToolError
from investigation.metrics import ELIGIBLE_BUCKETS, get_metric
from investigation.panel import PeriodPair, make_pair
from investigation.review.scorecard import region_scorecard
from investigation.tools.change import material_change, metric_change
from utils import loan_flags, unit_metrics

SHORTFALL, EXCESS, WORSE = "collection_shortfall", "excess_30plus", "worsening"
LISTS = (SHORTFALL, EXCESS, WORSE)
NEW_THIS_MONTH, NO_EMI_LAST_MONTH = "New this month", "No EMI due last month"

_LABELS = {
    SHORTFALL: "Collection% (Month Collection excl. reserve / Net Collection Demand Inst+Exp+BC), "
               "standardised by opening bucket",
    EXCESS: "30+ (SMA-1 and worse) among running loans with an EMI due, standardised by agreement vintage",
    WORSE: "Change in the 30+ rate (running loans with an EMI due), against the rest of the portfolio's change",
}
_FORMULAS = {
    SHORTFALL: "sum over opening buckets of region demand x rest-of-portfolio collection rate, minus region collected",
    EXCESS: "region 30+ SOH minus sum over vintages of region eligible SOH x rest-of-portfolio SOH-weighted 30+ rate",
    WORSE: "(region 30+ change - rest-of-portfolio 30+ change) x region eligible loans this month",
}
# Shown once in a brief; every other caveat is about one region and says which.
SHORTFALL_DISCLAIMER = "Collection shortfalls are estimates against the rest of the portfolio, not cash that can be collected."
EXCESS_DISCLAIMER = "Excess 30+ exposure is measured against the rest of the portfolio, not avoidable loss."
WORSE_DISCLAIMER = "Getting worse compares each region's change with the rest of the portfolio's change, both months as filed."
DISCLAIMERS = (SHORTFALL_DISCLAIMER, EXCESS_DISCLAIMER, WORSE_DISCLAIMER)

_IMPACT_LABEL = {
    SHORTFALL: "estimated collection shortfall against a like-for-like book (not guaranteed recoverable cash)",
    EXCESS: "benchmark-relative excess 30+ exposure (not avoidable loss)",
    WORSE: "30+ loans more than the rest of the portfolio's change implies",
}


@dataclass
class Candidate:
    id: str
    list: str
    region: str
    metric: str
    formula: str
    period: dict
    value: float                  # the region's own rate, %
    benchmark: float              # the like-for-like rate it is compared with, %
    gap_pts: float                # value vs benchmark, % points, in the bad direction
    impact: dict                  # {"amount", "unit", "label"}
    raw_impact: float             # the same impact without standardising (transparency)
    mix_part: float | None        # raw - standardised: how much of the raw gap is book mix
    numerator: float
    denominator: float
    eligible: int
    band: float                   # noise band on the impact, same unit
    z: float
    q_value: float = float("nan")
    unmatched_pct: float = 0.0    # share of the base with no like-for-like benchmark (fallback strata)
    material: bool = False
    reasons_not_listed: list = field(default_factory=list)
    evidence_id: str = ""
    data_version: str = ""
    caveats: list = field(default_factory=list)
    se_pts: float | None = None   # one standard error of gap_pts, in % points (outcome tracking compares months)

    def baseline(self) -> dict:
        """The figures a decision on this candidate keeps, so next month's
        review can say whether the gap moved beyond noise (outcomes.py)."""
        return {"month": self.period.get("curr", ""), "gap_pts": round(float(self.gap_pts), 4),
                "se_pts": round(float(self.se_pts or 0.0), 4), "value": round(float(self.value), 4),
                "benchmark": round(float(self.benchmark), 4), "impact": round(float(self.impact["amount"]), 2),
                "unit": self.impact["unit"]}


@dataclass
class ReviewResult:
    scorecard: pd.DataFrame
    lists: dict                   # list name -> listed candidates, ranked by impact
    assessed: dict                # list name -> every ranked region's candidate, listed or not
    too_small: list
    on_lists: dict                # region -> lists it appears on, for regions on 2+
    data_version: str
    store: EvidenceStore
    curr_joined: pd.DataFrame     # this month with last month's bucket carried (the app's month join)
    caveats: list


# ── Pure tables (one row per region; the formulas of spec 5.3) ───────────────

def _rest_by_stratum(cells: pd.DataFrame, s: str, region) -> pd.DataFrame:
    """Every stratum's sums for the loans NOT in `region`."""
    totals = cells.groupby(s).sum(numeric_only=True)
    own = cells[cells["region"] == region].set_index(s).drop(columns="region")
    return totals.sub(own.reindex(totals.index, fill_value=0), fill_value=0)


def shortfall_table(df: pd.DataFrame, g: str, s: str, d: str, c: str) -> pd.DataFrame:
    """Per region: expected collection E = sum_s D_gs * R_rs (R the rest's
    rate in stratum s), shortfall E - C, the raw shortfall against the rest's
    overall rate, the book-mix part (raw - standardised), and the delta-method
    standard error of the shortfall."""
    w = pd.DataFrame({"region": df[g].to_numpy(), s: df[s].to_numpy(), "D": df[d].to_numpy(float),
                      "C": df[c].to_numpy(float)})
    w = w.assign(C2=w["C"] ** 2, CD=w["C"] * w["D"], D2=w["D"] ** 2)
    cells = w.groupby(["region", s], sort=True).sum().reset_index()
    rows = []
    for region in sorted(cells["region"].unique()):
        own = cells[cells["region"] == region].set_index(s)
        rest = _rest_by_stratum(cells, s, region)
        r_all = rest["C"].sum() / rest["D"].sum() if rest["D"].sum() else float("nan")
        rate = (rest["C"] / rest["D"].where(rest["D"] > 0)).reindex(own.index)
        fallback = sorted(map(str, rate.index[rate.isna()]))
        rate = rate.fillna(r_all)
        expected = float((own["D"] * rate).sum())
        D_g, C_g = float(own["D"].sum()), float(own["C"].sum())
        # Var(C_g - sum_s D_gs R_rs): the region's own residuals, plus the
        # uncertainty of each benchmark rate (ratio estimator, delta method).
        var_own = float((own["C2"] - 2 * rate * own["CD"] + rate ** 2 * own["D2"]).sum())
        rr = rest.reindex(own.index)
        rest_res = (rr["C2"] - 2 * rate * rr["CD"] + rate ** 2 * rr["D2"]).fillna(0)
        var_rate = (rest_res / rr["D"].where(rr["D"] > 0) ** 2).fillna(0)
        var = var_own + float((own["D"] ** 2 * var_rate).sum())
        raw = D_g * r_all - C_g
        unmatched = float(own.loc[own.index.astype(str).isin(fallback), "D"].sum())
        rows.append({"region": region, "demand": D_g, "collected": C_g, "expected": expected,
                     "unmatched_pct": 100 * unmatched / D_g if D_g else 0.0,
                     "shortfall": expected - C_g, "raw_shortfall": raw, "mix_part": raw - (expected - C_g),
                     "rate_pct": 100 * C_g / D_g if D_g else float("nan"),
                     "benchmark_pct": 100 * expected / D_g if D_g else float("nan"),
                     "gap_pts": 100 * (expected - C_g) / D_g if D_g else float("nan"),
                     "se": math.sqrt(max(var, 0.0)), "fallback_strata": fallback,
                     "strata": own[["D", "C"]].assign(rest_rate=rate).reset_index()})
    return pd.DataFrame(rows)


def excess_table(df: pd.DataFrame, g: str, s: str, y: str, h: str) -> pd.DataFrame:
    """Per region: expected 30+ loans X = sum_s N_gs * q_rs, excess loans
    Y - X, excess SOH (SOH-weighted rate x SOH, one unit), raw versions
    against the rest's overall rate, and the binomial standard error."""
    w = pd.DataFrame({"region": df[g].to_numpy(), s: df[s].to_numpy(), "N": 1,
                      "Y": df[y].to_numpy(int), "H": df[h].to_numpy(float)})
    w["H30"] = w["H"] * w["Y"]
    cells = w.groupby(["region", s], sort=True).sum().reset_index()
    rows = []
    for region in sorted(cells["region"].unique()):
        own = cells[cells["region"] == region].set_index(s)
        rest = _rest_by_stratum(cells, s, region).reindex(own.index)
        rest_all = _rest_by_stratum(cells, s, region)
        q_all = rest_all["Y"].sum() / rest_all["N"].sum() if rest_all["N"].sum() else float("nan")
        qh_all = rest_all["H30"].sum() / rest_all["H"].sum() if rest_all["H"].sum() else float("nan")
        q = (rest["Y"] / rest["N"].where(rest["N"] > 0))
        qh = (rest["H30"] / rest["H"].where(rest["H"] > 0))
        fallback = sorted(map(str, q.index[q.isna()]))
        q, qh = q.fillna(q_all), qh.fillna(qh_all)
        N_g, Y_g, H_g, H30_g = (float(own[k].sum()) for k in ("N", "Y", "H", "H30"))
        expected = float((own["N"] * q).sum())
        p = own["Y"] / own["N"]
        var = float((own["N"] * p * (1 - p)).sum()
                    + (own["N"] ** 2 * q * (1 - q) / rest["N"].where(rest["N"] > 0)).fillna(0).sum())
        unmatched = float(own.loc[own.index.astype(str).isin(fallback), "N"].sum())
        rows.append({"region": region, "eligible": int(N_g), "thirty_plus": int(Y_g), "soh": H_g,
                     "unmatched_pct": 100 * unmatched / N_g if N_g else 0.0,
                     "expected_loans": expected, "excess_loans": Y_g - expected,
                     "excess_soh": H30_g - float((own["H"] * qh).sum()),
                     "raw_excess_loans": Y_g - N_g * q_all, "raw_excess_soh": H30_g - H_g * qh_all,
                     "rate_pct": 100 * Y_g / N_g, "benchmark_pct": 100 * expected / N_g,
                     "gap_pts": 100 * (Y_g - expected) / N_g, "se": math.sqrt(max(var, 0.0)),
                     "fallback_strata": fallback,
                     "strata": own[["N", "Y", "H", "H30"]].assign(rest_rate=q, rest_soh_rate=qh).reset_index()})
    return pd.DataFrame(rows)


# ── Statistics ───────────────────────────────────────────────────────────────

def _p_one_sided(z: float) -> float:
    """P(Z >= z): a large positive z is a gap in the bad direction."""
    if math.isinf(z):
        return 0.0 if z > 0 else 1.0
    return 0.5 * math.erfc(z / math.sqrt(2))


def _bh(p_values: list[float]) -> list[float]:
    """Benjamini-Hochberg q-values, in the input order."""
    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i])
    q, running = [0.0] * m, 1.0
    for rank in range(m, 0, -1):
        i = order[rank - 1]
        running = min(running, p_values[i] * m / rank)
        q[i] = running
    return q


def _z(gap: float, se: float) -> float:
    if se > 0:
        return gap / se
    return math.inf if gap > 0 else (-math.inf if gap < 0 else 0.0)


# ── The review ───────────────────────────────────────────────────────────────

def _data_version(pair: PeriodPair) -> str:
    """Hash of both months' key columns, independent of row order."""
    cols = ["Loan No", "RegionName", "curr_bucket", "SOH", "Ag_Date", "Net Collection Demand Inst+Exp+BC",
            "Month Collection (Excluding Reserve Collection)"]
    h = hashlib.sha256()
    for df in (pair.prev, pair.curr):
        part = df[[c for c in cols if c in df.columns]].sort_values("Loan No", kind="stable")
        h.update(pd.util.hash_pandas_object(part, index=False).to_numpy().tobytes())
    h.update(f"{pair.prev_month}|{pair.curr_month}".encode())
    return h.hexdigest()[:16]


def _joined(curr: pd.DataFrame, prev: pd.DataFrame) -> pd.DataFrame:
    """The app's month join (app._cached_filter): last month's bucket per loan."""
    if "prev_bucket" in curr.columns:
        return curr
    carry = prev[["Loan No", "curr_bucket"]].drop_duplicates("Loan No").rename(columns={"curr_bucket": "prev_bucket"})
    return curr.merge(carry, on="Loan No", how="left")


def _opening_stratum(pair: PeriodPair) -> pd.Series:
    """Each of this month's loans' opening bucket: last month's closing bucket."""
    last = pair.curr["Loan No"].map(dict(zip(pair.prev["Loan No"], pair.prev["curr_bucket"].astype(str))))
    out = last.where(last.isin(ELIGIBLE_BUCKETS), NO_EMI_LAST_MONTH)
    return out.mask(last.isna(), NEW_THIS_MONTH)


def _check_totals(name: str, got: float, want: float | None, tol: float) -> Check:
    if want is None:
        return Check(name, False, "the scorecard has no figure for this region")
    return Check(name, abs(got - want) <= tol, f"{got:,.2f} vs {want:,.2f}")


def _qualify(assessed: dict[str, list[Candidate]], too_small: set) -> None:
    """Benjamini-Hochberg across every ranked test of the review (regions x
    lists: one family, since they are read together), then every floor
    (spec 5.5). Per-list correction let about a quarter of pure-noise reviews
    list something (Phase 1 noise eval). Appends to reasons_not_listed."""
    ranked = [c for cands in assessed.values() for c in cands if c.region not in too_small]
    for c, q in zip(ranked, _bh([_p_one_sided(c.z) for c in ranked])):
        c.q_value = q
    for name, cands in assessed.items():
        for c in cands:
            why = c.reasons_not_listed
            if c.region in too_small:
                why.append(f"fewer than {REVIEW_MIN_ACCOUNTS} eligible loans")
            if c.unmatched_pct > REVIEW_MAX_UNMATCHED_PCT:
                why.append(f"no like-for-like benchmark for {c.unmatched_pct:.0f}% of its base")
            if not c.impact["amount"] > c.band:
                why.append("within its noise band")
            if not c.q_value <= REVIEW_FDR:
                why.append(f"not significant after correcting for {len(ranked)} tests (q {c.q_value:.2f})")
            if name != WORSE and c.impact["amount"] < REVIEW_MIN_RS:
                why.append(f"under Rs {REVIEW_MIN_RS:,.0f}")
            if not c.gap_pts >= REVIEW_MIN_PTS:
                why.append(f"gap under {REVIEW_MIN_PTS} % points")


def shortfall_frame(pair: PeriodPair, groups: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    """(this month's loans with an EMI due as g / s / d / c for shortfall_table,
    cash collected per group from loans with no EMI due). groups: one label per
    row of pair.curr (region, or branch for a deep-dive). Strata: opening bucket.

    Loans with no EMI due this month (e.g. matured loans clearing arrears)
    still collect, and the Dashboard's Collection% counts that cash against
    zero demand (dummy file: a region at 139.99%). The shortfall compares
    loans with demand only; the rest is reported and reconciled separately."""
    flags = loan_flags(pair.curr)
    work = pd.DataFrame({"g": groups, "s": _opening_stratum(pair), "d": flags["demand"], "c": flags["collected"]})
    return work[work["d"] > 0], work[work["d"] <= 0].groupby("g")["c"].sum()


def excess_frame(pair: PeriodPair, groups: pd.Series) -> pd.DataFrame:
    """This month's eligible loans as g / s / y / h for excess_table. Strata:
    agreement vintage ("All" when the files have no usable agreement dates)."""
    metric = get_metric("thirty_plus_rate")
    strata = (pair.labels("vintage", "curr") if pair.unavailable("vintage") is None
              else pd.Series("All", index=pair.curr.index))
    work = pd.DataFrame({"g": groups, "s": strata, "y": metric.numerator(pair.curr),
                         "h": loan_flags(pair.curr)["soh"]})
    return work[metric.eligible(pair.curr)]


def review_regions(curr: pd.DataFrame, prev: pd.DataFrame, curr_month: str, prev_month: str,
                   pair: PeriodPair | None = None) -> ReviewResult:
    """The deterministic review: scorecard, three lists, overlap. No model call.
    pair: the same two months already paired, to share its work (the runner)."""
    pair = pair if pair is not None else make_pair(curr, prev, curr_month, prev_month)
    store, version = EvidenceStore(), _data_version(pair)
    period = {"prev": pair.prev_month, "curr": pair.curr_month}
    joined = _joined(curr, prev)
    scorecard = region_scorecard(joined, prev, curr_month)
    totals = unit_metrics(joined, ["RegionName"])
    totals.index = totals["RegionName"].astype(str).str.strip()

    metric = get_metric("thirty_plus_rate")
    region = pair.labels("region", "curr")
    eligible = metric.eligible(pair.curr)
    sizes = eligible.groupby(region).sum()
    too_small = sorted(r for r, n in sizes.items() if n < REVIEW_MIN_ACCOUNTS)
    caveats = []
    assessed: dict[str, list[Candidate]] = {}

    # L1: collection shortfall, by opening bucket.
    new_share = 100 * (_opening_stratum(pair) == NEW_THIS_MONTH).mean()
    work, no_demand = shortfall_frame(pair, region)
    out = []
    for r in shortfall_table(work, "g", "s", "d", "c").to_dict("records"):
        want = totals["Demand"].get(r["region"]), totals["Collected"].get(r["region"])
        r["collected_no_demand"] = float(no_demand.get(r["region"], 0.0))
        ev = store.add("review_collection_shortfall", {"region": r["region"]},
                       {k: v for k, v in r.items() if k != "strata"}, tables={"strata": r["strata"]},
                       checks=[Check("strata_add_up", abs(r["strata"]["D"].sum() - r["demand"]) < 0.5
                                     and abs(r["strata"]["C"].sum() - r["collected"]) < 0.5),
                               _check_totals("demand_matches_scorecard", r["demand"], want[0], 1.0),
                               _check_totals("collected_matches_scorecard",
                                             r["collected"] + r["collected_no_demand"], want[1], 1.0)])
        cav = [SHORTFALL_DISCLAIMER]
        if want[1] and r["collected_no_demand"] > 0.05 * want[1]:
            cav.append(f"Rs {r['collected_no_demand']:,.0f} ({100 * r['collected_no_demand'] / want[1]:.0f}% of "
                       "the region's collection) came from loans with no EMI due this month: counted in the "
                       "Dashboard's Collection%, left out here.")
        if r["fallback_strata"]:
            cav.append(f"No other region has demand in {', '.join(r['fallback_strata'])}: "
                       "the rest of the portfolio's overall rate is used there.")
        out.append(Candidate(
            f"{SHORTFALL}:{r['region']}", SHORTFALL, r["region"], _LABELS[SHORTFALL], _FORMULAS[SHORTFALL],
            period, round(r["rate_pct"], 2), round(r["benchmark_pct"], 2), round(r["gap_pts"], 2),
            {"amount": r["shortfall"], "unit": "Rs", "label": _IMPACT_LABEL[SHORTFALL]},
            r["raw_shortfall"], r["mix_part"], r["collected"], r["demand"], int(sizes.get(r["region"], 0)),
            INVESTIGATION_NOISE_SE * r["se"], _z(r["shortfall"], r["se"]), unmatched_pct=r["unmatched_pct"],
            se_pts=100 * r["se"] / r["demand"] if r["demand"] else None,
            evidence_id=ev.result_id, data_version=version, caveats=cav))
    assessed[SHORTFALL] = out

    # L2: excess 30+ exposure, by agreement vintage.
    if pair.unavailable("vintage") is not None:
        caveats.append("No usable agreement dates: excess 30+ is not standardised for loan age.")
    work = excess_frame(pair, region)
    out = []
    for r in excess_table(work, "g", "s", "y", "h").to_dict("records"):
        want = int((metric.numerator(pair.curr) & (region == r["region"])).sum())
        ev = store.add("review_excess_30plus", {"region": r["region"]},
                       {k: v for k, v in r.items() if k != "strata"}, tables={"strata": r["strata"]},
                       checks=[Check("strata_add_up", int(r["strata"]["N"].sum()) == r["eligible"]
                                     and int(r["strata"]["Y"].sum()) == r["thirty_plus"]),
                               _check_totals("thirty_plus_matches_metric", r["thirty_plus"], want, 0)])
        cav = [EXCESS_DISCLAIMER]
        if r["fallback_strata"]:
            cav.append(f"No other region has loans agreed in {', '.join(r['fallback_strata'])}: "
                       "the rest of the portfolio's overall rate is used there.")
        se_soh = r["se"] * (r["soh"] / r["eligible"]) if r["eligible"] else 0.0   # loans' noise in SOH terms
        out.append(Candidate(
            f"{EXCESS}:{r['region']}", EXCESS, r["region"], _LABELS[EXCESS], _FORMULAS[EXCESS], period,
            round(r["rate_pct"], 2), round(r["benchmark_pct"], 2), round(r["gap_pts"], 2),
            {"amount": r["excess_soh"], "unit": "Rs", "label": _IMPACT_LABEL[EXCESS]},
            r["raw_excess_soh"], r["raw_excess_soh"] - r["excess_soh"], r["thirty_plus"], r["eligible"],
            r["eligible"], INVESTIGATION_NOISE_SE * se_soh, _z(r["excess_loans"], r["se"]),
            se_pts=100 * r["se"] / r["eligible"] if r["eligible"] else None,
            unmatched_pct=r["unmatched_pct"], evidence_id=ev.result_id, data_version=version, caveats=cav))
    assessed[EXCESS] = out

    # L3: getting worse, against the rest of the portfolio's change.
    out = []
    # Each loan's region (its label in both months), worked out once: a
    # per-region isin on Loan No was most of a 90k-loan review's time.
    labels = pair.loan_label("region").astype(str).str.upper()
    prev_region = labels.reindex(pair.prev["Loan No"].to_numpy()).to_numpy()
    curr_region = labels.reindex(pair.curr["Loan No"].to_numpy()).to_numpy()
    for name in sorted(sizes.index):
        try:
            ev = metric_change(store, pair, scope={"region": name})
            want = str(name).upper()
            rest = pair.narrowed(prev_region != want, curr_region != want)
            rest_change, rest_noise, _ = material_change(rest, metric)
        except ToolError as e:
            caveats.append(f"{name}: change not tested ({e}).")
            continue
        s = ev.summary
        se = math.hypot((s["noise_pp"] or 0) / INVESTIGATION_NOISE_SE, (rest_noise or 0) / INVESTIGATION_NOISE_SE)
        diff = s["change_pp"] - rest_change
        n1 = s["curr"]["eligible"]
        cand = Candidate(
            f"{WORSE}:{name}", WORSE, name, _LABELS[WORSE], _FORMULAS[WORSE], period,
            round(s["curr"]["rate"], 2), round(s["prev"]["rate"] + rest_change, 2), round(diff, 2),
            {"amount": diff / 100 * n1, "unit": "loans", "label": _IMPACT_LABEL[WORSE]},
            s["change_pp"] / 100 * n1, None, s["curr"]["numerator"], n1, n1,
            INVESTIGATION_NOISE_SE * se / 100 * n1, _z(diff, se), se_pts=se,
            evidence_id=ev.result_id, data_version=version,
            caveats=[WORSE_DISCLAIMER])
        if not s["material"]:
            cand.reasons_not_listed.append("the region's own change is not material")
        out.append(cand)
    assessed[WORSE] = out

    _qualify(assessed, set(too_small))
    for cands in assessed.values():
        for c in cands:
            if not store.get(c.evidence_id).checks_passed:
                c.reasons_not_listed.append("its self-checks failed")
            c.material = not c.reasons_not_listed

    lists = {name: sorted((c for c in cands if c.material), key=lambda c: (-c.impact["amount"], c.region))
             for name, cands in assessed.items()}
    seen: dict[str, list[str]] = {}
    for name, cands in lists.items():
        for c in cands:
            seen.setdefault(c.region, []).append(name)
    if new_share:
        caveats.append(f"{new_share:.1f}% of this month's loans are new: their collection is compared "
                       "with other new loans only.")
    return ReviewResult(scorecard, lists, assessed, too_small, {r: v for r, v in seen.items() if len(v) > 1},
                        version, store, joined, caveats)
