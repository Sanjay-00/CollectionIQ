"""Two-month books for the Portfolio Review evaluations (no customer data).

make_review_book(seed, **plants) -> (prev, curr). Every region has the same
behaviour unless a plant changes it, so anything a list flags is the plant's:

  Opening bucket (last month's closing bucket): 30+ odds grow with loan age
  (seasoning), then a 30+ loan is SMA-1 / SMA-2 / NPA and the rest STD / 1-30 DPD.
  Collection: a loan pays with odds set by its opening bucket, and an overdue
  loan that pays clears arrears too (more than the month's EMI), so a book with
  more overdue loans collects a different share of demand without anyone
  collecting worse.
  This month: a loan keeps its 30+ status, except a `churn` share that redraws
  it at the same odds (ordinary month-to-month noise).

Plants (region name -> setting):
  mix={g: {bucket: share}}   g's opening buckets drawn from this mix instead (bucket-mix trap)
  shortfall={g: f}           g's loans pay with f times the usual odds, in every bucket
  old_book={g: years}        g's loans all agreed `years` or more ago (seasoning trap)
  risk={g: f}                g's 30+ odds times f at every loan age
  worsen={g: pts}            pts % of g's loans more at 30+ this month
  sizes={g: n}               g has n loans instead of `n_per_region`
"""
from __future__ import annotations

import numpy as np
import pandas as pd

REVIEW_PREV_MONTH, REVIEW_CURR_MONTH = "2026-08", "2026-09"
REGIONS = ("NORTH", "SOUTH", "EAST", "WEST", "CENTRAL", "COAST")

_PAY_ODDS = {"STD": 0.95, "1-30 DPD": 0.75, "SMA-1": 0.55, "SMA-2": 0.40, "NPA": 0.20, "NEW": 0.90}
_PAID_TIMES_EMI = {"STD": 1.0, "1-30 DPD": 1.3, "SMA-1": 1.6, "SMA-2": 1.8, "NPA": 2.0, "NEW": 1.0}
_ARREARS_EMI = {"STD": 0.0, "1-30 DPD": 0.5, "SMA-1": 1.5, "SMA-2": 2.5, "NPA": 4.0}
_DEEP, _DEEP_P = ["SMA-1", "SMA-2", "NPA"], [0.4, 0.3, 0.3]
_LIGHT, _LIGHT_P = ["STD", "1-30 DPD"], [0.8, 0.2]
_AS_OF = pd.Timestamp("2026-09-30")


def _thirty_plus_odds(age_years: np.ndarray) -> np.ndarray:
    return np.clip(0.08 + 0.05 * age_years, 0.0, 0.6)


def _bucket(rng, is_30: np.ndarray) -> np.ndarray:
    deep = rng.choice(_DEEP, len(is_30), p=_DEEP_P)
    light = rng.choice(_LIGHT, len(is_30), p=_LIGHT_P)
    return np.where(is_30, deep, light).astype(object)


def make_review_book(seed: int = 0, n_per_region: int = 2500, new_share: float = 0.05, churn: float = 0.05,
                     regions=REGIONS, mix=None, shortfall=None, old_book=None, risk=None, worsen=None,
                     sizes=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    mix, shortfall, old_book = mix or {}, shortfall or {}, old_book or {}
    risk, worsen, sizes = risk or {}, worsen or {}, sizes or {}
    frames, start = [], 0
    for g in regions:
        n = int(sizes.get(g, n_per_region))
        age = rng.uniform(old_book.get(g, 0.2), old_book.get(g, 0.0) + 4.0, n)
        new = rng.random(n) < new_share
        age[new] = 0.0
        p30 = np.clip(_thirty_plus_odds(age) * risk.get(g, 1.0), 0.0, 0.95)
        if g in mix:
            names, p = zip(*mix[g].items())
            opening = rng.choice(names, n, p=np.array(p) / sum(p)).astype(object)
        else:
            opening = _bucket(rng, rng.random(n) < p30)
        was_30 = np.isin(opening, _DEEP)
        redraw = rng.random(n) < churn
        now_30 = np.where(redraw, rng.random(n) < p30, was_30)
        now_30[new] = False
        closing = np.where(now_30 == was_30, opening, _bucket(rng, now_30)).astype(object)
        closing[new] = "STD"
        if g in worsen:
            pool = np.flatnonzero(~np.isin(closing, _DEEP) & ~new)
            closing[rng.permutation(pool)[: int(worsen[g] / 100 * n)]] = "SMA-1"
        stratum = np.where(new, "NEW", opening)
        demand = rng.integers(8, 13, n) * 1000.0
        pays = rng.random(n) < np.clip(np.vectorize(_PAY_ODDS.get)(stratum) * shortfall.get(g, 1.0), 0, 1)
        collected = np.where(pays, demand * np.vectorize(_PAID_TIMES_EMI.get)(stratum), 0.0)
        frames.append(pd.DataFrame({
            "Loan No": [f"L{start + i}" for i in range(n)], "Loan Status": "RUN", "RegionName": g,
            "Unit": [f"{g[:3]}-BR{i % 4}" for i in range(n)], "Zone": "Z1",
            "Ag_Date": (_AS_OF - pd.to_timedelta(age * 365.25, unit="D")).normalize(),
            "SOH": rng.integers(2, 9, n) * 1_00_000.0, "Net Collection Demand Inst+Exp+BC": demand,
            "Month Collection (Excluding Reserve Collection)": collected,
            "_new": new, "_opening": opening, "_closing": closing,
        }))
        start += n
    book = pd.concat(frames, ignore_index=True)
    curr = book.assign(curr_bucket=book["_closing"], **{"Arrears / EMI": book["_closing"].map(_ARREARS_EMI)})
    prev = book[~book["_new"]].assign(curr_bucket=lambda d: d["_opening"],
                                      **{"Arrears / EMI": lambda d: d["_opening"].map(_ARREARS_EMI)})
    drop = ["_new", "_opening", "_closing"]
    return prev.drop(columns=drop).reset_index(drop=True), curr.drop(columns=drop).reset_index(drop=True)
