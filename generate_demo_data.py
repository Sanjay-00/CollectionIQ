"""Generate the public demo dataset (sample_data/*_Demo*.xlsx).

Fully SYNTHETIC: no row comes from a real extract. Distributions (status mix,
bucket mix by vintage, segment ticket sizes, tenures, collection rate, loans
per customer, flag rates) were calibrated once, as aggregates only, against a
real monthly LCC; every name, mobile, loan number and vehicle number here is
made up (mobiles are sequential 900xxxxxxx / 910xxxxxxx, vehicle numbers use
the non-existent RTO code MH00).

Each region carries a deliberate "story" so every dashboard tab has something
to show (see REGIONS). Months are simulated loan by loan -- end-June state ->
July payments -> August payments -> September -- so the previous-month file,
bucket migration, roll rates and the due-date-missed list all follow from the
same customer behaviour instead of being drawn independently.

    python generate_demo_data.py

Writes Current_Month_Demo.xlsx (Aug 2026), Previous_Month_Demo.xlsx (Jul
2026) and Demo_Due_Date_Missed_List.xlsx (29 Sep 2026) into sample_data/.
Deterministic: the same seed always produces the same files.
"""
import datetime as dt
import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from utils import REQUIRED_COLS

SEED = 20260930
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_data")
CURR, PREV, JUNE = dt.date(2026, 8, 1), dt.date(2026, 7, 1), dt.date(2026, 6, 1)
SEPT = dt.date(2026, 9, 1)
LIST_DATE = dt.date(2026, 9, 29)
COHORT_START = dt.date(2025, 11, 1)

rng = np.random.default_rng(SEED)


# ── Regions and their stories ────────────────────────────────────────────────
# pay: base monthly probability a running customer pays; trend: change in
# August vs July (+ improving / - worsening). Other knobs raise one root-cause
# driver so each story region shows a different "dominant driver".
_DEFAULT = dict(pay=0.80, trend=0.0, insurance=0.08, chronic=0.30, fleet=0.05, run_share=0.70,
                recent_risk=1.0, shock=0.0, colending=0.01, easy=0.03, non_starter=0.03)

REGIONS = [
    ("PUNE", "WEST ZONE", 1200, ["HADAPSAR", "CHAKAN", "BARAMATI", "SHIRUR", "DAUND"], dict(pay=0.86, trend=0.06, run_share=0.76)),
    ("THANE", "WEST ZONE", 900, ["BHIWANDI", "KALYAN", "DOMBIVLI", "SHAHAPUR", "MURBAD"], dict(colending=0.22, easy=0.22)),
    ("RATNAGIRI", "WEST ZONE", 550, ["CHIPLUN", "KHED", "DAPOLI", "SANGAMESHWAR", "LANJA"], dict(pay=0.85)),
    ("KOLHAPUR", "WEST ZONE", 800, ["ICHALKARANJI", "KAGAL", "GADHINGLAJ", "PANHALA", "SHIROL"], dict(insurance=0.60)),
    ("SATARA", "WEST ZONE", 700, ["KARAD", "WAI", "PHALTAN", "KOREGAON", "MAN"], dict(pay=0.78, trend=0.32, chronic=0.10)),
    ("SANGLI", "WEST ZONE", 600, ["MIRAJ", "TASGAON", "ISLAMPUR", "VITA", "JATH"], dict(pay=0.70, shock=0.90, chronic=0.10, recent_risk=0.7, trend=-0.06)),
    ("NASHIK", "NORTH ZONE", 1000, ["MALEGAON", "SINNAR", "IGATPURI", "NIPHAD", "YEOLA"], dict(pay=0.76, trend=-0.40)),
    ("JALGAON", "NORTH ZONE", 800, ["BHUSAWAL", "CHOPDA", "AMALNER", "PACHORA", "RAVER"], dict(fleet=0.45)),
    ("NANDED", "NORTH ZONE", 700, ["DEGLUR", "KINWAT", "BILOLI", "HADGAON", "MUKHED"], dict(recent_risk=3.0, non_starter=0.60)),
    ("SOLAPUR", "NORTH ZONE", 750, ["PANDHARPUR", "BARSHI", "AKKALKOT", "MOHOL", "SANGOLA"], dict(pay=0.56, chronic=0.95, recent_risk=0.4, run_share=0.62)),
]
WEAK_BRANCH = {"NASHIK": "MALEGAON"}           # the worsening region's problem branch
STAR_VS_WEAK = {("PUNE", "HADAPSAR"): (0.15, -0.30)}
N_CLOSED_IN_AUG = 120   # July loans fully closed before August (absent from the August file)

# Calibrated from the real extract (aggregate shares only).
# Real extracts carry ~45% matured/sold accounts with leftover dues; the demo
# keeps ~30% so the running-book stories aren't buried under legacy accounts.
BUCKETS_OLDER = {"STD": 0.573, "1-30": 0.131, "SMA1": 0.092, "SMA2": 0.072, "NPA": 0.132}
BUCKETS_RECENT = {"STD": 0.836, "1-30": 0.101, "SMA1": 0.042, "SMA2": 0.015, "NPA": 0.007}
# segment: (share, code, loan amount q50, q90, tenures, makes)
SEGMENTS = {
    "Heavy Goods Vehicle": (0.19, "MHGV", 660_000, 2_100_000, [48, 60], ["TATA", "ASHOK LEYLAND", "EICHER", "BHARATBENZ"]),
    "Passenger Commercial": (0.17, "PSGCMPSG3W", 320_000, 950_000, [36, 48], ["TATA", "MAHINDRA", "FORCE", "MARUTI"]),
    "Private Car": (0.13, "PVTCAAR", 365_000, 970_000, [36, 48, 60], ["MARUTI", "HYUNDAI", "TATA", "MAHINDRA"]),
    "Small Goods Vehicle": (0.10, "SGVLGVIGV", 255_000, 600_000, [36, 48], ["TATA", "MAHINDRA", "ASHOK LEYLAND"]),
    "Farm Equipment": (0.10, "FRMVFRMEQP", 300_000, 655_000, [36, 48], ["MAHINDRA", "SWARAJ", "SONALIKA", "JOHN DEERE"]),
    "Construction Vehicle": (0.09, "CME", 1_600_000, 4_500_000, [48, 60], ["JCB", "TATA HITACHI", "CASE"]),
    "Business Loan": (0.08, "BL", 60_000, 210_000, [12, 24], ["BUSINESS LOAN"]),
    "Light Goods Vehicle": (0.06, "SGVLGVIGV", 450_000, 1_780_000, [36, 48], ["TATA", "MAHINDRA", "EICHER"]),
    "Machinery": (0.03, "CME", 1_600_000, 3_400_000, [48], ["JCB", "CASE", "ESCORTS"]),
    "Cargo Three Wheeler": (0.03, "PSGCMPSG3W", 180_000, 320_000, [36], ["BAJAJ", "PIAGGIO", "MAHINDRA"]),
    "Passenger 3wheeler": (0.02, "PSGCMPSG3W", 170_000, 300_000, [36], ["BAJAJ", "PIAGGIO", "TVS"]),
}
VEHICLE_DESC = {
    "TATA": ["SIGNA 4825", "ULTRA 1918", "ACE GOLD", "NEXON XZ"], "ASHOK LEYLAND": ["ECOMET 1615", "BOSS 1115", "DOST PLUS"],
    "EICHER": ["PRO 3015", "PRO 2049", "SKYLINE"], "BHARATBENZ": ["1917R", "2823C"], "MAHINDRA": ["BOLERO PICKUP", "SUPRO", "ARJUN 605", "XUV 300"],
    "MARUTI": ["ERTIGA", "SWIFT DZIRE", "EECO CARGO"], "HYUNDAI": ["CRETA", "AURA", "VENUE"], "FORCE": ["TRAVELLER 26"],
    "SWARAJ": ["744 FE", "855 FE"], "SONALIKA": ["DI 745", "TIGER 55"], "JOHN DEERE": ["5310", "5050D"],
    "JCB": ["3DX", "JS 205"], "TATA HITACHI": ["EX 200", "SHINRAI"], "CASE": ["770 EX", "851 FX"], "ESCORTS": ["HYDRA 14"],
    "BAJAJ": ["MAXIMA CARGO", "RE COMPACT"], "PIAGGIO": ["APE XTRA", "APE CITY"], "TVS": ["KING DURAMAX"], "BUSINESS LOAN": [""],
}
FIRST = ["RAHUL", "SURESH", "VIJAY", "AMIT", "RAVI", "PRIYA", "NITIN", "DEEPAK", "SACHIN", "ARUN", "SUNITA", "MANOJ",
         "GANESH", "SANJAY", "ANIL", "KAVITA", "RAJESH", "SWATI", "MAHESH", "POOJA", "SANDEEP", "VIKAS", "ASHOK", "NEHA",
         "PRAKASH", "YOGESH", "SHUBHAM", "SNEHA", "TUSHAR", "DILIP", "RAMESH", "SUNIL", "KIRAN", "VAISHALI", "HEMANT", "ROHIT"]
LAST = ["PATIL", "MORE", "JADHAV", "SHINDE", "KULKARNI", "DESAI", "KADAM", "PAWAR", "GAIKWAD", "SALVE", "DESHMUKH", "BHOSALE",
        "CHAVAN", "SAWANT", "JOSHI", "WAGH", "KALE", "THORAT", "NIKAM", "MANE", "GHARAT", "SONAWANE", "LOKHANDE", "DHOLE"]


def _mi(d: dt.date) -> int:
    return d.year * 12 + d.month


def _add_months(d: dt.date, n: int) -> dt.date:
    y, m = divmod(d.year * 12 + d.month - 1 + n, 12)
    return dt.date(y, m + 1, min(d.day, 28))


def _pick(weights: dict):
    keys = list(weights)
    p = np.array([weights[k] for k in keys], dtype=float)
    return keys[rng.choice(len(keys), p=p / p.sum())]


def _lognormal(q50: float, q90: float) -> float:
    sigma = np.log(q90 / q50) / 1.2816
    return float(np.exp(rng.normal(np.log(q50), sigma)))


def _emi(amount: float, tenure: int, annual_rate: float) -> float:
    r = annual_rate / 12
    return amount * r / (1 - (1 + r) ** -tenure)


def _pos_after(amount: float, tenure: int, annual_rate: float, paid: int) -> float:
    r = annual_rate / 12
    emi = _emi(amount, tenure, annual_rate)
    return max(amount * (1 + r) ** paid - emi * ((1 + r) ** paid - 1) / r, 0.0)


# ── The portfolio ─────────────────────────────────────────────────────────────

@dataclass
class Loan:
    loan_no: str
    region: str
    zone: str
    branch: str
    exec_code: str
    exec_name: str
    exec_q: float
    cust_id: int
    fleet: bool
    segment: str
    ag_date: dt.date
    tenure: int
    amount: float
    rate: float
    status_aug: str               # status in the August (current) file
    habit: float = 0.9            # this customer's monthly probability of paying
    insurance_only: bool = False
    non_starter: bool = False
    colending: bool = False
    nach: bool = True
    due_day: int = 5
    closes_in_aug: bool = False   # present in July, fully closed before August
    params: dict = field(default_factory=dict)
    states: dict = field(default_factory=dict)   # month -> end-of-month state

    @property
    def emi(self) -> float:
        return _emi(self.amount, self.tenure, self.rate)

    @property
    def premium(self) -> float:
        return 0.0 if self.segment == "Business Loan" else round(max(0.012 * self.amount, 3000), -2)

    def status(self, month: dt.date) -> str:
        if self.status_aug == "RUN" and _mi(month) - _mi(self.ag_date) > self.tenure:
            return "MAT"
        return self.status_aug

    def exists(self, month: dt.date) -> bool:
        if _mi(self.ag_date) > _mi(month):
            return False
        return not (self.closes_in_aug and _mi(month) >= _mi(CURR))


def _disbursal_month_weights(max_age: int) -> np.ndarray:
    """Growing book with festive (Oct/Nov) and year-end (Mar) peaks."""
    ages = np.arange(max_age + 1)
    months = [(_add_months(CURR, -a)).month for a in ages]
    season = np.array([1.35 if m in (10, 11) else 1.25 if m == 3 else 1.0 for m in months])
    return (1.025 ** -ages) * season


def build_portfolio() -> list[Loan]:
    loans: list[Loan] = []
    exec_names_used: set = set()
    seq = 0
    cust_seq = 0
    for r_idx, (region, zone, n_loans, branches, story) in enumerate(REGIONS):
        p = {**_DEFAULT, **story}
        n_loans += N_CLOSED_IN_AUG // len(REGIONS)
        # 4 executives per branch; quality spread, with deliberate star/weak pairs.
        execs = {}
        for b in branches:
            qs = list(STAR_VS_WEAK.get((region, b), (0.06, -0.06))) + [0.0, 0.0]
            if WEAK_BRANCH.get(region) == b:
                qs = [q - 0.25 for q in qs]
            rng.shuffle(qs)
            execs[b] = []
            for q in qs:
                while True:
                    name = f"{FIRST[rng.integers(len(FIRST))]} {LAST[rng.integers(len(LAST))]}"
                    if name not in exec_names_used:
                        exec_names_used.add(name)
                        break
                execs[b].append((f"E{r_idx + 1:02d}{len(exec_names_used):03d}", name, q))

        # Customers: most hold one loan; fleet operators hold 3-8 (story: JALGAON).
        remaining = n_loans
        fleet_loans_target = int(n_loans * p["fleet"])
        while remaining > 0:
            cust_seq += 1
            if fleet_loans_target > 0:
                size = int(min(rng.integers(3, 9), remaining))
                fleet_loans_target -= size
            else:
                size = int(min(_pick({1: 0.84, 2: 0.14, 3: 0.015, 4: 0.005}), remaining))
            remaining -= size
            home = branches[rng.integers(len(branches))]
            fleet_seg = "Heavy Goods Vehicle" if size >= 3 else None
            for _ in range(size):
                seq += 1
                branch = home if rng.random() < 0.85 else branches[rng.integers(len(branches))]
                ecode, ename, eq = execs[branch][rng.integers(4)]
                rest = (1 - p["run_share"]) / 2
                status = _pick({"RUN": p["run_share"], "MAT": rest, "S&S": rest})
                seg = fleet_seg or _pick({k: v[0] for k, v in SEGMENTS.items()})
                _, _, q50, q90, tenures, _ = SEGMENTS[seg]
                amount = round(min(max(_lognormal(q50, q90), 5000), q90 * 3), -3)
                rate = 0.24 if seg == "Business Loan" else 0.16
                if status == "RUN":
                    w = _disbursal_month_weights(59)
                    age = int(rng.choice(len(w), p=w / w.sum()))
                    fits = [t for t in tenures if t > age] or [max(age + 12, 60)]
                    tenure = int(fits[rng.integers(len(fits))])
                elif status == "MAT":
                    tenure = int(tenures[rng.integers(len(tenures))])
                    age = tenure + int(rng.integers(3, 150))
                else:  # S&S: repossessed and sold part-way through
                    tenure = int(tenures[rng.integers(len(tenures))])
                    age = int(rng.integers(max(tenure // 3, 12), 170))
                ag = _add_months(CURR, -age).replace(day=int(rng.integers(1, 29)))
                loan = Loan(
                    loan_no=f"DEMO{seq:06d}", region=region, zone=zone, branch=branch,
                    exec_code=ecode, exec_name=ename, exec_q=eq, cust_id=cust_seq, fleet=size >= 3,
                    segment=seg, ag_date=ag, tenure=tenure, amount=amount, rate=rate, status_aug=status,
                    nach=rng.random() < 0.62, due_day=int(_pick({5: 44, 10: 20, 15: 17, 20: 15, 1: 2, 7: 2})), params=p,
                )
                recent = ag >= COHORT_START
                loan.colending = status == "RUN" and ag >= dt.date(2024, 1, 1) and rng.random() < p["colending"] * 1.6
                loan.non_starter = status == "RUN" and _mi(CURR) - _mi(ag) in (1, 2, 3) and rng.random() < p["non_starter"]
                loan.insurance_only = status == "RUN" and not recent and rng.random() < p["insurance"] * 0.35
                loans.append(loan)

    # Loans present in July but fully closed (zero dues) before August.
    eligible = [l for l in loans if l.status_aug == "RUN" and not l.fleet and _mi(CURR) - _mi(l.ag_date) > 6]
    for loan in rng.choice(np.array(eligible, dtype=object), size=N_CLOSED_IN_AUG, replace=False):
        loan.closes_in_aug = True
    return loans


# ── Month-by-month simulation ────────────────────────────────────────────────

def _initial_state(loan: Loan) -> dict:
    """Arrears at end-June, drawn from the calibrated bucket mix."""
    p = loan.params
    emi = loan.emi
    age = _mi(JUNE) - _mi(loan.ag_date)
    state = dict(inst=0.0, exp=0.0, bc=0.0, pc=0.0, streak=0, paid_ever=age > 1, cum_bc=0.0, cum_pc=0.0)
    if age < 1:
        return state
    status = loan.status(JUNE)
    if status == "RUN":
        base = BUCKETS_RECENT if loan.ag_date >= COHORT_START else BUCKETS_OLDER
        worse = (1 - p["pay"] - loan.exec_q) / (1 - 0.80)
        if loan.ag_date >= COHORT_START:
            worse *= p["recent_risk"]
        if loan.fleet:
            worse *= 1.8
        w = {k: (v if k == "STD" else v * max(worse, 0.1)) for k, v in base.items()}
        bucket = _pick(w)
        a = {"STD": 0.0, "1-30": rng.uniform(0.1, 1), "SMA1": rng.uniform(1, 2), "SMA2": rng.uniform(2, 3),
             "NPA": min(float(np.exp(rng.normal(np.log(7), 0.9))), 60)}[bucket]
        shock = bucket == "NPA" and rng.random() < p["shock"]
        if shock:
            a = rng.uniform(6, 10)            # sudden hard-bucket slip, still paying now and then
        a = min(a, age)
        habit = {"STD": 0.94, "1-30": 0.80, "SMA1": 0.64, "SMA2": 0.50, "NPA": 0.25}[bucket]
        habit += (p["pay"] - 0.80) * 0.6 + loan.exec_q * 0.6 - (0.08 if loan.fleet else 0.0)
        if loan.ag_date >= COHORT_START:
            habit = 1 - (1 - habit) * p["recent_risk"] ** 0.5
        loan.habit = float(np.clip(0.45 if shock else habit, 0.02, 0.99))
    else:
        a = 0.0 if rng.random() < 0.03 else min(float(np.exp(rng.normal(np.log(10), 0.9))), 80)
    if loan.insurance_only:
        state["exp"] = float(rng.integers(6, 26) * 1000)
        a = 0.0
    state["inst"] = round(a * emi, 0)
    if a > 0 and rng.random() < 0.15:
        state["exp"] = float(rng.integers(1, 9) * 1000)
    if a >= 3:
        chronic = status == "RUN" and rng.random() < p["chronic"] and not (p["shock"] and a < 10)
        state["streak"] = int(rng.integers(3, 9)) if chronic else int(rng.integers(0, 2))
        if chronic:
            loan.habit = 0.03
            a = min(max(a, rng.uniform(6.5, 14)), age)   # a no-collection streak this long means > 6 EMIs behind
            state["inst"] = round(a * emi, 0)
    elif a > 0:
        state["streak"] = int(rng.integers(0, 2))
    state["pc"] = round(0.03 * state["inst"], 0)
    state["cum_pc"] = state["pc"] * 1.5
    if loan.non_starter:
        state["paid_ever"] = False
    return state


def _step(loan: Loan, prev: dict, month: dt.date) -> tuple[dict, dict]:
    """One month: dues arise, the customer pays (or not), arrears roll."""
    p = loan.params
    emi = loan.emi
    age = _mi(month) - _mi(loan.ag_date)
    status = loan.status(month)
    running = status == "RUN" and 1 <= age <= loan.tenure
    due_inst = emi if running else 0.0
    due_exp = loan.premium if running and age % 12 == 0 else 0.0
    due_bc = 500.0 if running and loan.nach and prev["streak"] > 0 else 0.0
    due_pc = round(0.02 * prev["inst"], 0) if prev["inst"] > 0 else 0.0
    open_ = dict(inst=prev["inst"], exp=prev["exp"], bc=prev["bc"], pc=prev["pc"])
    coll = dict(inst=0.0, exp=0.0, bc=0.0, pc=0.0)
    a = prev["inst"] / emi if emi else 0.0

    if running:
        if loan.non_starter:
            pass
        elif loan.insurance_only:
            coll["inst"] = open_["inst"] + due_inst
            coll["bc"], coll["pc"] = open_["bc"] + due_bc, open_["pc"] + due_pc
        else:
            trend = p["trend"] if month == CURR else (p["trend"] * 0.5 if month == SEPT else 0.0)
            pay = float(np.clip(loan.habit + trend * (1 if a >= 1 else 0.3), 0.02, 0.99))
            u = rng.random()
            if u < pay:
                k = 1.0 + (1.0 if a >= 1 and rng.random() < 0.35 else 0.0) + (a if 0 < a < 1 and rng.random() < 0.6 else 0.0)
                if trend > 0 and a >= 2 and rng.random() < trend * 1.2:
                    k = 1.0 + min(a - 1.5, 3.0) + rng.uniform(0, 0.5)   # improving region: catch-up payments
                coll["inst"] = min(k * emi, open_["inst"] + due_inst)
            elif u < pay + (1 - pay) * 0.5 * min(loan.habit / 0.5, 1.0):
                # Part-payments follow the habit too: a habitual non-payer rarely pays even half.
                coll["inst"] = min(0.5 * emi, open_["inst"] + due_inst)
            if coll["inst"] > 0:
                if rng.random() < 0.8:
                    coll["exp"] = open_["exp"] + due_exp
                if rng.random() < 0.7:
                    coll["bc"], coll["pc"] = open_["bc"] + due_bc, open_["pc"] + due_pc
    elif open_["inst"] + open_["exp"] > 0 and rng.random() < 0.03:
        coll["inst"] = round(open_["inst"] * rng.uniform(0.05, 0.2), 0)   # recovery on a closed account

    # Easy-settlement story: a small leftover after a near-full catch-up.
    if running and rng.random() < p["easy"] * 0.12 and open_["inst"] > 0:
        coll["inst"] = max(open_["inst"] + due_inst - float(rng.integers(150, 950)), 0.0)

    closing = {k: max(open_[k] + d - coll[k], 0.0) for k, d in
               zip(("inst", "exp", "bc", "pc"), (due_inst, due_exp, due_bc, due_pc))}
    paid_total = sum(coll.values())
    state = dict(
        inst=round(closing["inst"], 0), exp=closing["exp"], bc=closing["bc"], pc=closing["pc"],
        streak=0 if paid_total > 0 else prev["streak"] + 1,
        paid_ever=prev["paid_ever"] or paid_total > 0,
        cum_bc=prev["cum_bc"] + due_bc, cum_pc=prev["cum_pc"] + due_pc,
        last_paid=month if paid_total > 0 else prev.get("last_paid"),
        last_amt=paid_total if paid_total > 0 else prev.get("last_amt", 0.0),
    )
    flows = dict(age=age, status=status, due_inst=due_inst, due_exp=due_exp, due_bc=due_bc, due_pc=due_pc,
                 open=open_, coll=coll, closing=closing)
    return state, flows


def simulate(loans: list[Loan]) -> dict:
    """{month: {loan_no: (state, flows)}} for July, August and September."""
    out = {m: {} for m in (PREV, CURR, SEPT)}
    for loan in loans:
        state = _initial_state(loan)
        if not state.get("last_paid") and state["paid_ever"]:
            state["last_paid"] = _add_months(JUNE, -state["streak"])
            state["last_amt"] = loan.emi
        for month in (PREV, CURR, SEPT):
            if _mi(loan.ag_date) > _mi(month):
                continue
            if _mi(loan.ag_date) == _mi(month):
                state = dict(inst=0.0, exp=0.0, bc=0.0, pc=0.0, streak=0, paid_ever=False, cum_bc=0.0, cum_pc=0.0)
            state, flows = _step(loan, state, month)
            out[month][loan.loan_no] = (dict(state), flows)
    return out


# ── Rendering rows in the LCC layout ─────────────────────────────────────────

_customers: dict = {}


def _customer(cust_id: int) -> tuple[str, str, str, str]:
    if cust_id not in _customers:
        name = f"{FIRST[rng.integers(len(FIRST))]} {LAST[rng.integers(len(LAST))]}"
        guar = f"{FIRST[rng.integers(len(FIRST))]} {LAST[rng.integers(len(LAST))]}"
        _customers[cust_id] = (name, f"90{cust_id:08d}", guar, f"91{cust_id:08d}")
    return _customers[cust_id]


def _vehicle(loan: Loan) -> tuple[str, str, int, str]:
    make_pool = SEGMENTS[loan.segment][5]
    make = make_pool[int(loan.cust_id + int(loan.loan_no[4:])) % len(make_pool)]
    desc_pool = VEHICLE_DESC[make]
    desc = desc_pool[int(loan.loan_no[4:]) % len(desc_pool)]
    if loan.segment == "Business Loan":
        return make, desc, 0, ""
    n = int(loan.loan_no[4:])
    veh_id = f"MH00{chr(65 + n % 26)}{chr(65 + (n // 26) % 26)}{n % 10000:04d}"
    return make, desc, loan.ag_date.year - int(n % 4), veh_id


def row(loan: Loan, state: dict, f: dict, month: dt.date, sno: int) -> dict:
    emi = loan.emi
    age, status = f["age"], f["status"]
    accrued = int(np.clip(age, 0, loan.tenure))
    cust, mob, guar, gmob = _customer(loan.cust_id)
    make, desc, yom, veh = _vehicle(loan)
    cl, op, co = f["closing"], f["open"], f["coll"]
    arr_ie = cl["inst"] + cl["exp"]
    renewals = (1 + (accrued - 1) // 12) if accrued > 0 else 0
    cum_due_inst = round(emi * accrued, 0)
    cum_due_exp = loan.premium * renewals
    cum_coll_ie = max(cum_due_inst + cum_due_exp - arr_ie, 0.0)
    cum_due_ie = cum_due_inst + cum_due_exp
    lcc = 100.0 if cum_due_ie == 0 else round(min(cum_coll_ie / cum_due_ie * 100, 100), 2)
    total_cum = cum_coll_ie + max(state["cum_bc"] - cl["bc"], 0) + max(state["cum_pc"] - cl["pc"], 0)
    paid_now = sum(co.values())
    a = round(arr_ie / emi, 2) if emi else 0.0
    if status == "RUN":
        pos = _pos_after(loan.amount, loan.tenure, loan.rate, accrued)
    else:
        pos = 0.0
    strike_y = (f["due_inst"] > 0 and co["inst"] >= f["due_inst"]) or lcc >= 100 or cl["inst"] <= 0
    non_starter = loan.non_starter and accrued >= 1 and not state["paid_ever"]
    chronic = status == "RUN" and state["streak"] >= 3 and a > 6
    last_paid = state.get("last_paid")
    last_receipt = None
    if last_paid:
        last_receipt = dt.date(last_paid.year, last_paid.month, min(loan.due_day + int(loan.cust_id % 6), 28))
    npa_legal = status != "RUN" or a >= 6
    legal = npa_legal and (loan.cust_id % 10) < 7
    sale_type = ["Auction", "Physical", "Direct"][loan.cust_id % 3] if status == "S&S" else "N"
    return {
        "SNo": sno, "Loan No": loan.loan_no, "CHANNEL": "CV", "BU": "DEMO CV", "StateName": "MAHARASHTRA",
        "Zone": loan.zone, "RegionName": loan.region, "Unit": loan.branch, "Ag_Date": loan.ag_date,
        "SRC Code": f"SRC{loan.branch[:3]}{loan.cust_id % 4 + 1}", "SRC Name": f"DEMO MOTORS {loan.branch}",
        "MNT CODE": loan.exec_code, "MNT NAME": loan.exec_name, "Due Dt": loan.due_day, "Tenure": loan.tenure,
        "Loan Status": status, "Loan Amount": loan.amount, "Veh ID": veh, "Cust Name": cust, "Guar Name": guar,
        "Cust Mob No": mob, "Guar Mob No": gmob, "Segment": SEGMENTS[loan.segment][1], "SegmentName": loan.segment,
        "Make": make, "Vehicle Description": desc, "Year Of Manufacture": yom or None,
        "Arrear Opening": round(sum(op.values()), 0), "ARREARS OPEN AGAINST INST": round(op["inst"], 0),
        "ARREARS OPEN AGAINST EXP": op["exp"], "ARREARS OPEN AGAINST BC": op["bc"], "ARREARS OPEN AGAINST PC": op["pc"],
        "OPENING RESERVE COLLECTION": 0.0, "Month Due-Inst": round(f["due_inst"], 0), "Month Due-Exp": f["due_exp"],
        "MONTH DUE (BC)": f["due_bc"], "MONTH DUE PC": f["due_pc"], "Month Receipt Amount": round(paid_now, 0),
        "MONTHCOLL INST": round(co["inst"], 0), "MONTHCOLL EXP": co["exp"], "MONTHCOLL BC": co["bc"], "MONTHCOLL PC": co["pc"],
        "Month Collection (Excluding Reserve Collection)": round(paid_now, 0),
        "Closing Arrears": round(sum(cl.values()), 0),
        "UN-CLEARED CHEQUE FOR THE MONTH/Amount Not remitted by RE": 0.0,
        "Cum Due-Inst": cum_due_inst, "Cum Due-Exp": cum_due_exp, "CUM DUE (BC)": state["cum_bc"], "Cum Due PC": state["cum_pc"],
        "Cum Coll (Inst+Exp)": round(cum_coll_ie, 0), "Total Cum Collection": round(total_cum, 0),
        "ARREARS AGAINST INST": round(cl["inst"], 0), "ARREARS AGAINST EXP": cl["exp"], "ARREARS AGAINST BC": cl["bc"],
        "ARREARS AGAINST PC": cl["pc"], "CLOSING RESERVE COLLECTION": 0.0, "Arrears against Inst+Exp": round(arr_ie, 0),
        "Uncleared Cheque/Amount Not remitted by RE": 0.0, "LCC%": lcc, "Arrears / EMI": a,
        "DelinquencyDays": int(round(a * 30)), "VehEMI Accrued": accrued, "ClosingPC": cl["pc"], "POS": round(pos, 0),
        "scheme": ["STANDARD", "STANDARD", "STANDARD", "TOP-UP", "REFINANCE"][loan.cust_id % 5],
        "Non Starter": "Yes" if non_starter else "No",
        "Strike": ("Yes" if strike_y else "No") if loan.cust_id % 7 == 0 else ("Y" if strike_y else "N"),
        "RCEndors(>90Days)": 0, "RTO / INSURANCE": 0.0,
        "NET Collection Demand Inst+Exp": round(f["due_inst"] + f["due_exp"], 0),
        "Net Collection Demand Inst+Exp+BC": round(f["due_inst"] + f["due_exp"] + f["due_bc"], 0),
        "NET COLLECTION": round(paid_now, 0), "NET COLLECTION EXCLUDING RESERVE COLL": round(paid_now, 0),
        "Last Receipt Date": last_receipt, "Last Receipt Amount": round(state.get("last_amt") or 0.0, 0) if last_receipt else None,
        "ParentLDueDate": _add_months(loan.ag_date, loan.tenure),
        "No Coll 3 Months and >6 EMI": "Y" if chronic else "N",
        "NACHStatus": "Y" if loan.nach else "N", "SaleType": sale_type,
        "CoLending_Loans": "Y" if loan.colending else "N", "CUSTOMER_STATUS": "ALIVE",
        "LGL_FLAG": "Y" if legal else "N",
        "LGL_DESCRIPTION": ["Arbitration Numbering", "Pre arbitration", "Sec 138 Notice"][loan.cust_id % 3] if legal else "",
        "TyreFlag": "Y" if loan.cust_id % 12 == 0 else "N",
        "FUEL_TYPE": "" if loan.segment == "Business Loan" else
                     ("DIESEL" if loan.cust_id % 20 < 17 else ["PETROL", "CNG", "ELECTRIC"][loan.cust_id % 3]),
        "NPA Status": "Cust DPD" if loan.cust_id % 14 == 0 else "Loan DPD",
        "Paymethod": "NACH" if loan.nach else ("CASH" if loan.cust_id % 12 else "PDC"),
        "Security_Type": "UNSECURED" if loan.segment == "Business Loan" else "SECURED",
    }


def snapshot(loans: list[Loan], sim: dict, month: dt.date) -> pd.DataFrame:
    rows = []
    for loan in loans:
        if not loan.exists(month) or loan.loan_no not in sim[month]:
            continue
        state, flows = sim[month][loan.loan_no]
        rows.append(row(loan, state, flows, month, len(rows) + 1))
    df = pd.DataFrame(rows)
    extras = [c for c in df.columns if c not in REQUIRED_COLS]
    return df[[c for c in REQUIRED_COLS if c in df.columns] + extras]


def add_real_world_mess(df: pd.DataFrame) -> pd.DataFrame:
    """A little of what real extracts look like, so the app's cleaning shows:
    older agreement dates as dd/mm/yyyy text, and a few duplicate rows."""
    df = df.copy()
    ag = df["Ag_Date"].astype(object)
    old = pd.to_datetime(df["Ag_Date"]) < pd.Timestamp("2016-01-01")
    as_text = old & (rng.random(len(df)) < 0.6)
    ag[as_text] = [d.strftime("%d/%m/%Y") for d in df.loc[as_text, "Ag_Date"]]
    df["Ag_Date"] = ag
    dups = df.sample(n=12, random_state=SEED)
    df = pd.concat([df, dups], ignore_index=True)
    df["SNo"] = range(1, len(df) + 1)
    return df


def due_date_missed_list(loans: list[Loan], sim: dict) -> pd.DataFrame:
    """The 29 Sep 2026 list: running Nov'25+ loans that missed a due date,
    as a separate daily report would give it (its own column names), plus a
    few rows the Root Cause notes explain: loans from another zone, loans
    from a branch missing from the LCC, and executives changed since August."""
    rows = []
    for loan in loans:
        if loan.ag_date < COHORT_START or loan.loan_no not in sim[SEPT] or not loan.exists(CURR):
            continue
        state, flows = sim[SEPT][loan.loan_no]
        if flows["status"] != "RUN":
            continue
        a = round((flows["closing"]["inst"] + flows["closing"]["exp"]) / loan.emi, 2)
        if a <= 0:
            continue
        cust, *_ = _customer(loan.cust_id)
        re_code, re_name = loan.exec_code, loan.exec_name
        if loan.cust_id % 33 == 0:  # reassigned to a colleague in September
            re_code, re_name = f"E99{loan.cust_id % 1000:03d}", f"{FIRST[loan.cust_id % len(FIRST)]} {LAST[-1 - loan.cust_id % 5]}"
        rows.append(_list_row(loan.loan_no, loan.zone, loan.region, loan.branch, loan.segment, loan.ag_date,
                              cust, a, flows["closing"], re_code, re_name))
    # A new branch in JALGAON region that isn't in the August LCC extract yet.
    for i in range(40):
        rows.append(_list_row(f"DEMOX{i:05d}", "NORTH ZONE", "JALGAON", "SHIRPUR", "Small Goods Vehicle",
                              dt.date(2026, 1 + i % 8, 1 + i % 27), f"{FIRST[i % len(FIRST)]} {LAST[i % len(LAST)]}",
                              round(0.3 + (i % 9) * 0.4, 2), dict(inst=5000.0 + 800 * i, exp=0.0, bc=0.0, pc=0.0),
                              f"E99{i:03d}", "NEW BRANCH EXEC"))
    # Another zone's loans: the list covers the whole business unit.
    for i in range(120):
        rows.append(_list_row(f"DEMOG{i:05d}", "KONKAN ZONE", "GOA", ["MARGAO", "PONDA", "MAPUSA"][i % 3],
                              "Passenger Commercial", dt.date(2025, 11 + i % 2, 1 + i % 27),
                              f"{FIRST[-1 - i % len(FIRST)]} {LAST[i % len(LAST)]}", round(0.2 + (i % 12) * 0.3, 2),
                              dict(inst=4000.0 + 500 * i, exp=0.0, bc=0.0, pc=0.0), f"G{i:04d}", "OTHER ZONE EXEC"))
    return pd.DataFrame(rows)


def _list_row(loan_no, zone, region, unit, segment, ag, cust, a, closing, re_code, re_name) -> dict:
    return {
        "MISMONTH": int(LIST_DATE.strftime("%Y%m%d")), "BUNAME": "DEMO CV", "LOAN NO": loan_no, "ZONE": zone,
        "REGIONNAME": region, "UNIT": unit, "SEGMENT": SEGMENTS[segment][1], "AG DATE": ag, "CUST NAME": cust,
        "CLOSING ARREARS": round(sum(closing.values()), 0), "ARREARS / EMI": a, "RE CODE": re_code, "RE NAME": re_name,
    }


def main() -> None:
    loans = build_portfolio()
    sim = simulate(loans)
    curr = add_real_world_mess(snapshot(loans, sim, CURR))
    prev = add_real_world_mess(snapshot(loans, sim, PREV))
    lst = due_date_missed_list(loans, sim)
    os.makedirs(OUT_DIR, exist_ok=True)
    for name, frame in (("Current_Month_Demo.xlsx", curr), ("Previous_Month_Demo.xlsx", prev),
                        ("Demo_Due_Date_Missed_List.xlsx", lst)):
        frame.to_excel(os.path.join(OUT_DIR, name), index=False, sheet_name="Sheet1")
        print(f"wrote {name}: {len(frame):,} rows x {len(frame.columns)} cols")


if __name__ == "__main__":
    main()
