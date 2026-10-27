"""Two-month test books with a known cause planted (no customer data).

make_book(n, seed, **plants) -> (prev, curr). Last month's loans get random
attributes over every column the dimensions read, and a bucket from a fixed
30+ chance per product (TRACTOR 60%, the rest 20%). This month is the same
loans in the same buckets unless a plant changes it, so any movement is the
plant's:
  product_share=f   TRACTOR loans multiplied by f this month (new loans, same 60%)
  drop_branch=b     branch b missing this month
  shift={b: k}      k > 0: k STD loans of branch b become SMA-1; k < 0: |k| 30+ loans become STD
  grow=f            f x n new STD loans this month
  days_off=f        fraction f of this month's eligible loans get days that disagree with their bucket
  reassign=k        k loans change executive this month
  relabel=f         CHANNEL holds a file name for fraction f of this month's loans (real Sept 2026 fault)
  add_branch=(b, f, agreed)  f x n loans of a branch b only this month, agreed on `agreed` (an old date
                    = the branch was missing from last month's file, real Sept 2026: HINGL, WASHM)
  n_branches=k      k equal branches instead of the default ten (real file: 146)
  churn=f           fraction f of loans redraw their bucket this month at the same odds: pure noise,
                    as in any real month
  rate_up=(p, pts)  product p's 30+ rate up by pts % points (STD loans of p become SMA-1)
  vintage_rate_up=(start, end, pts)   loans agreed in [start, end]: pts % points more 30+
  product_by_vintage=(p, start)       product p's loans are all agreed from `start` (confounds p with vintage)
  exec_rate_up=(b, e, pts, size)      `size` loans of branch b belong to executive e (both months); pts more 30+
  cure_block=f      fraction f of loans that would have cured this month stay where they were
  base_grow=k       k loans with no EMI due last month (NA) are STD this month: the base grows, 30+ doesn't
  segment_swing=(col, value, n, pts)  n loans get `value` in `col` (both months); pts more 30+
  duplicates=k      k duplicate rows in this month's frame
  attrs=dict        set on this month's frame (header renames, duplicates dropped at upload)

Each "pts more 30+" plant moves STD loans of the group to SMA-1, pts % of the group's size.
"""
import numpy as np
import pandas as pd

BRANCHES = [f"BR{i}" for i in range(9)] + ["TINYBR"]
BRANCH_P = [0.995 / 9] * 9 + [0.005]
REGION_OF = {b: f"R{min(i // 3, 2)}" for i, b in enumerate(BRANCHES)}
PRODUCTS = ["CAR", "TRUCK", "BUS", "TRACTOR"]
PRODUCT_P = [0.4, 0.3, 0.2, 0.1]
_DAYS = {"STD": 0, "1-30 DPD": 15, "SMA-1": 45, "SMA-2": 75, "NPA": 120, "NA": 0}


def _buckets(rng, products):
    risky = rng.random(len(products)) < np.where(products == "TRACTOR", 0.6, 0.2)
    deep = rng.choice(["SMA-1", "SMA-2", "NPA"], len(products), p=[0.4, 0.25, 0.35])
    light = rng.choice(["STD", "1-30 DPD"], len(products), p=[0.8, 0.2])
    out = np.where(risky, deep, light).astype(object)
    out[rng.random(len(products)) < 0.03] = "NA"
    return out


def _loans(rng, n, start, n_branches=None):
    products = rng.choice(PRODUCTS, n, p=PRODUCT_P)
    if n_branches:
        branch = rng.choice([f"BR{i}" for i in range(n_branches)], n)
        region = [f"R{int(b[2:]) % 16}" for b in branch]
    else:
        branch = rng.choice(BRANCHES, n, p=BRANCH_P)
        region = [REGION_OF[b] for b in branch]
    df = pd.DataFrame({
        "Loan No": [f"L{start + i}" for i in range(n)], "Loan Status": "RUN",
        "Zone": rng.choice(["Z1", "Z2", "Z3"], n), "RegionName": region,
        "Unit": branch, "MNT NAME": np.char.add("EXEC", rng.integers(0, 40, n).astype(str)),
        "SegmentName": products, "CHANNEL": rng.choice(["DIRECT", "DEALER"], n),
        "scheme": rng.choice([f"SCH{i}" for i in range(5)], n), "Make": rng.choice(["TATA", "ASHOK", "EICHER"], n),
        "Ag_Date": pd.to_datetime("2024-01-01") + pd.to_timedelta(rng.integers(0, 600, n), unit="D"),
        "Tenure": rng.choice([12, 24, 36, 48, 60], n), "Loan Amount": rng.integers(100_000, 2_000_000, n),
        "Year Of Manufacture": rng.integers(2012, 2026, n), "Due Dt": rng.choice([5, 10, 15, 20], n),
        "Customer Loan Count": rng.choice([1, 1, 1, 2, 6], n), "NACHStatus": rng.choice(["ACTIVE", "INACTIVE"], n),
        "FUEL_TYPE": rng.choice(["DIESEL", "CNG"], n), "SRC Name": rng.choice(["D1", "D2", "D3"], n),
    })
    df["curr_bucket"] = _buckets(rng, products)
    return df


def _days(df):
    return df["curr_bucket"].map(_DAYS)


def _pick(rng, index, k: int):
    """k of these loans at random: plants never take the first rows, which
    tied unrelated plants together (review: 56% of a product plant sat in the
    10% of loans another plant reassigned)."""
    return rng.permutation(np.asarray(index))[:max(int(k), 0)]


def _worse(rng, curr: pd.DataFrame, mask: pd.Series, pts: float) -> None:
    """pts % of the group's loans more at 30+: STD loans in the group become SMA-1."""
    pool = curr.index[mask & (curr["curr_bucket"] == "STD")]
    curr.loc[_pick(rng, pool, pts / 100 * mask.sum()), "curr_bucket"] = "SMA-1"


def make_book(n=6000, seed=0, product_share=None, drop_branch=None, shift=None, grow=None, days_off=None,
              reassign=0, relabel=None, add_branch=None, n_branches=None, churn=None, rate_up=None,
              vintage_rate_up=None, product_by_vintage=None, exec_rate_up=None, cure_block=None, base_grow=None,
              segment_swing=None, duplicates=0, attrs=None):
    rng = np.random.default_rng(seed)
    prev = _loans(rng, n, 0, n_branches)
    # Attribute plants: set on last month's loans, so both months carry them.
    if product_by_vintage:
        product, start = product_by_vintage
        mine = prev["SegmentName"] == product
        span = (pd.Timestamp("2025-08-20") - pd.Timestamp(start)).days
        prev.loc[mine, "Ag_Date"] = pd.Timestamp(start) + pd.to_timedelta(rng.integers(0, span, mine.sum()), unit="D")
    if exec_rate_up:
        branch, executive, _, size = exec_rate_up
        prev.loc[_pick(rng, prev.index[prev["Unit"] == branch], size), "MNT NAME"] = executive
    if segment_swing:
        col, value, size, _ = segment_swing
        prev.loc[_pick(rng, prev.index, size), col] = value
    curr = prev.copy()
    if churn:
        redraw = rng.random(len(curr)) < churn
        curr.loc[redraw, "curr_bucket"] = _buckets(rng, curr.loc[redraw, "SegmentName"].to_numpy())
    if cure_block:
        cured = prev["curr_bucket"].isin(["SMA-1", "SMA-2", "NPA"]) & curr["curr_bucket"].isin(["STD", "1-30 DPD"])
        back = _pick(rng, curr.index[cured], cure_block * cured.sum())
        curr.loc[back, "curr_bucket"] = prev.loc[back, "curr_bucket"]
    if base_grow:
        na = curr.index[(prev["curr_bucket"] == "NA") & (curr["curr_bucket"] == "NA")]
        curr.loc[_pick(rng, na, base_grow), "curr_bucket"] = "STD"
    if rate_up:
        product, pts = rate_up
        _worse(rng, curr, curr["SegmentName"] == product, pts)
    if vintage_rate_up:
        start, end, pts = vintage_rate_up
        _worse(rng, curr, pd.to_datetime(curr["Ag_Date"]).between(start, end), pts)
    if exec_rate_up:
        branch, executive, pts, _ = exec_rate_up
        _worse(rng, curr, (curr["Unit"] == branch) & (curr["MNT NAME"] == executive), pts)
    if segment_swing:
        col, value, _, pts = segment_swing
        _worse(rng, curr, curr[col] == value, pts)
    if product_share:
        extra = _loans(rng, int((product_share - 1) * (prev["SegmentName"] == "TRACTOR").sum() * 3), n)
        extra = extra[extra["SegmentName"] == "TRACTOR"]
        curr = pd.concat([curr, extra], ignore_index=True)
    if grow:
        extra = _loans(rng, int(grow * n), n * 10)
        extra["curr_bucket"] = "STD"
        curr = pd.concat([curr, extra], ignore_index=True)
    if drop_branch:
        dropped = [drop_branch] if isinstance(drop_branch, str) else list(drop_branch)
        curr = curr[~curr["Unit"].isin(dropped)].reset_index(drop=True)
    if add_branch:
        name, frac, agreed = add_branch
        extra = _loans(rng, int(frac * n), n * 20).assign(Unit=name, Ag_Date=pd.Timestamp(agreed))
        curr = pd.concat([curr, extra], ignore_index=True)
    for branch, k in (shift or {}).items():
        source = ["STD"] if k > 0 else ["SMA-1", "SMA-2", "NPA"]
        pool = curr.index[(curr["Unit"] == branch) & curr["curr_bucket"].isin(source)]
        curr.loc[_pick(rng, pool, abs(k)), "curr_bucket"] = "SMA-1" if k > 0 else "STD"
    if reassign:
        curr.loc[_pick(rng, curr.index, reassign), "MNT NAME"] = "NEWEXEC"
    if relabel:
        curr.loc[_pick(rng, curr.index, relabel * len(curr)), "CHANNEL"] = "ALL CLOSING LCC.xlsx :: SATARA"
    prev["DelinquencyDays"], curr["DelinquencyDays"] = _days(prev), _days(curr)
    if days_off:
        sma1 = curr.index[curr["curr_bucket"] == "SMA-1"]
        k = int(days_off * curr["curr_bucket"].isin(["STD", "1-30 DPD", "SMA-1", "SMA-2", "NPA"]).sum())
        curr.loc[_pick(rng, sma1, k), "DelinquencyDays"] = 20
    if duplicates:
        curr = pd.concat([curr, curr.loc[_pick(rng, curr.index, duplicates)]], ignore_index=True)
    if attrs:
        curr.attrs.update(attrs)
    return prev, curr
