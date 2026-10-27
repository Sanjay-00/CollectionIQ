"""
The evaluation cases (spec 12.2, adjusted for the 2026-10-24 like-for-like
decision): two-month synthetic books with one planted cause each, and the
answer an investigation must give, written from the plant before any run.
Case ids are stable. HOLDOUT_IDS are never used to tune prompts (Phase 4).

recoverable(case) proves a case is fair: the plant can be seen by running
the Phase 1 tools directly, so a baseline or agent that misses it has failed,
not the case.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd

from config import INVESTIGATION_CONCENTRATION_MIN_PCT, INVESTIGATION_DRIVER_MIN_SHARE_PCT
from investigation.evaluation import Expected
from investigation.evidence import EvidenceStore
from investigation.panel import make_pair
from investigation.prepare import prepare
from investigation.synthetic import make_book
from investigation.tools import flow_bridge, metric_change, mix_rate, scan_dimensions

PREV_MONTH, CURR_MONTH = "2026-08", "2026-09"
N = 30000
NOISE = 0.06                   # share of loans that redraw their bucket each month, as in a real book


@dataclass
class Case:
    id: int
    name: str
    build: Callable[[], tuple[pd.DataFrame, pd.DataFrame]]
    expected: Expected
    holdout: bool = False


def _book(seed: int, **plants) -> Callable[[], tuple[pd.DataFrame, pd.DataFrame]]:
    plants.setdefault("churn", NOISE)
    return lambda: make_book(N, seed=seed, **plants)


def _without(columns: list[str], build):
    def inner():
        prev, curr = build()
        return prev.drop(columns=columns), curr.drop(columns=columns)
    return inner


TRACTOR = ("product", "TRACTOR")
# A material change no testable slice concentrates: from one month's bridge a
# fall in cures can't be told from a rise in roll-ins, so "concluded" with the
# flow mechanism and "inconclusive" are both right; naming a driver is not
# (review finding 4; spec 12.2).
NO_SLICE = ("concluded", "inconclusive")

CASES: list[Case] = [
    Case(1, "one vintage rolls into 30+ much faster", _book(101, vintage_rate_up=("2025-01-01", "2025-03-31", 15)),
         Expected("concluded", drivers=[("vintage", "2025-Q1")], mechanism="rate")),
    Case(2, "a risky product's share grows", _book(102, product_share=3.0),
         Expected("concluded", drivers=[TRACTOR], mechanism="mix")),
    Case(3, "a large branch is missing this month", _book(103, drop_branch="BR3", rate_up=("TRACTOR", 5)),
         Expected("concluded", drivers=[TRACTOR], flags=["units_missing"], excluded_units=["BR3"])),
    Case(4, "headers drifted in this month's file",
         _book(104, rate_up=("TRACTOR", 5), attrs={"header_renames": {"DelinquencyDay s": "DelinquencyDays"}}),
         Expected("concluded", drivers=[TRACTOR], flags=["header_renames"]), holdout=True),
    Case(5, "duplicate loan rows", _book(105, rate_up=("TRACTOR", 5), duplicates=50,
                                         attrs={"dropped_duplicate_loans": 50}),
         Expected("concluded", drivers=[TRACTOR], flags=["duplicates_dropped"])),
    Case(6, "nothing material changed", _book(106),
         Expected("no_material_change", no_drivers=True)),
    Case(7, "the cause is in a column the files lack",
         _without(["FUEL_TYPE"], _book(107, segment_swing=("FUEL_TYPE", "CNG", 9000, 6))),
         Expected(NO_SLICE, untestable=["fuel"], no_drivers=True)),
    Case(8, "cures dry up while roll-ins hold", _book(108, churn=0.12, cure_block=0.4),
         Expected(NO_SLICE, mechanism="flow", no_drivers=True)),
    Case(9, "offsetting branch moves under a flat total", _book(109, shift={"BR1": 250, "BR2": -250}),
         Expected("no_material_change", drivers=[("branch", "BR1"), ("branch", "BR2")]), holdout=True),
    Case(10, "a vintage pattern that is really a product effect",
         _book(110, product_by_vintage=("TRACTOR", "2025-04-01"), rate_up=("TRACTOR", 8)),
         Expected("concluded", drivers=[TRACTOR], forbidden=[("vintage", "2025-Q2"), ("vintage", "2025-Q3")],
                  mechanism="rate")),
    Case(11, "two causes at once", _book(111, rate_up=("TRACTOR", 6), shift={"BR2": 400}),
         Expected("concluded", drivers=[TRACTOR, ("branch", "BR2")])),
    Case(12, "a small segment's real but minor swing beside the main cause",
         _book(112, rate_up=("TRACTOR", 6), segment_swing=("Make", "RAREMAKE", 260, 3)),
         Expected("concluded", drivers=[TRACTOR], forbidden=[("make", "RAREMAKE")]), holdout=True),
    Case(13, "the cause is one executive in one branch", _book(113, exec_rate_up=("BR3", "EXEC7", 40, 400)),
         Expected("concluded", drivers=[("branch", "BR3")], deep=[("executive", "EXEC7 (BR3)")], mechanism="rate")),
    Case(14, "only the base changed", _book(114, churn=0.0, base_grow=600),
         Expected("concluded", mechanism="base", no_drivers=True)),
    Case(15, "many loans reassigned between executives", _book(115, rate_up=("TRACTOR", 5), reassign=3000),
         Expected("concluded", drivers=[TRACTOR], flags=["executives_changed"]), holdout=True),
    Case(16, "the days column disagrees with the buckets", _book(116, rate_up=("TRACTOR", 5), days_off=0.04),
         Expected("concluded", drivers=[TRACTOR], flags=["days_column_agreement"]), holdout=True),
    Case(17, "a branch is missing from last month's file (real Sept 2026 fault)",
         _book(117, rate_up=("TRACTOR", 5), add_branch=("HINGL", 0.03, "2024-05-01")),
         Expected("concluded", drivers=[TRACTOR], flags=["units_new"], excluded_units=["HINGL"])),
    Case(18, "eligible loans jump by a quarter", _book(118, rate_up=("TRACTOR", 5), grow=0.25),
         Expected("halted_data", flags=["population_change"], no_drivers=True)),
    Case(19, "a truncated extract: six of ten branches missing",
         _book(119, rate_up=("TRACTOR", 5), drop_branch=[f"BR{i}" for i in range(6)]),
         Expected("halted_data", flags=["units_missing"], no_drivers=True)),
]

HOLDOUT_IDS = frozenset(c.id for c in CASES if c.holdout)


def recoverable(case: Case) -> dict[str, bool]:
    """Each thing the case expects, checked by running the tools directly."""
    prev, curr = case.build()
    store = EvidenceStore()
    prep = prepare(store, make_pair(curr, prev, CURR_MONTH, PREV_MONTH))
    exp, out = case.expected, {}
    halts = exp.status == "halted_data"
    out["status"] = (prep.status == "halted_data") == halts
    if exp.flags:
        out["flags"] = set(exp.flags) <= set(prep.flags)
    if halts:
        return out
    if exp.excluded_units:
        left_out = {x["unit"] for x in (prep.adjustment.summary["excluded"] if prep.adjustment else [])}
        out["excluded_units"] = set(exp.excluded_units) <= left_out
    change = metric_change(store, prep.pair).summary
    material = change["material"]                     # above the line AND outside the noise band (spec 9.3)
    allowed = (exp.status,) if isinstance(exp.status, str) else exp.status
    out["material"] = material == bool(set(allowed) & {"concluded", "inconclusive"})
    scan = scan_dimensions(store, prep.pair).summary
    rank = {r["dimension"]: i for i, r in enumerate(scan["ranked"])}
    top_conc = max((r["concentration_pct"] or 0.0 for r in scan["ranked"]), default=0.0)

    def notable(dim, seg, scope=None):
        """Cited and beyond its fair share of the change (and noise), pushing
        the same way as a material total: a big segment that only rose with
        everything else, or one whose share fell, isn't what drove the rise."""
        s = mix_rate(store, prep.pair, dim, scope=scope).summary
        big_enough = lambda t: not material or \
            100 * t["contribution_pp"] / s["total_change_pp"] >= INVESTIGATION_DRIVER_MIN_SHARE_PCT
        return any(t["segment"].upper() == seg.upper() and t["beyond_share"] and big_enough(t) for t in s["top"])

    if exp.drivers:
        out["drivers"] = all(notable(d, s) for d, s in exp.drivers)
    if exp.deep:
        parent = exp.drivers[0]
        out["deep"] = all(notable(d, s, scope={parent[0]: parent[1]}) for d, s in exp.deep)
    if exp.forbidden and exp.drivers:
        first = rank.get(exp.drivers[0][0], 99)
        out["forbidden"] = all(rank.get(d, 99) > first for d, _ in exp.forbidden)
    if exp.untestable:
        out["untestable"] = set(exp.untestable) <= {u["dimension"] for u in scan["unavailable"]}
    if "inconclusive" in allowed:
        out["nothing_concentrated"] = top_conc < INVESTIGATION_CONCENTRATION_MIN_PCT
    if exp.mechanism in ("mix", "rate") and exp.drivers:
        top = mix_rate(store, prep.pair, exp.drivers[0][0]).summary["top"][0]
        bigger = abs(top["mix_pp"]) > abs(top["rate_pp"])
        out["mechanism"] = bigger if exp.mechanism == "mix" else not bigger
    if exp.mechanism == "base":
        out["mechanism"] = abs(change["base_effect_pp"]) > abs(change["numerator_effect_pp"])
    if exp.mechanism == "flow":
        lines = flow_bridge(store, prep.pair).summary["lines"]
        ins = lines["in_from_std"] + lines["in_from_1_30"]
        cures = lines["cured_to_std"] + lines["cured_to_1_30"]
        out["mechanism"] = ins > cures and top_conc < INVESTIGATION_CONCENTRATION_MIN_PCT
    out["checks_passed"] = all(e.checks_passed for e in store.all())
    return out
