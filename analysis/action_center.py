"""
Action overview: what to fix first, where, and how much -- the top of the
Dashboard tab. Pure pandas, no LLM: every sentence is built from numbers the
rest of the app already shows (utils.unit_metrics, analysis/roll_flow.py), so
a manager can trace each one back to a table.

Movement ("new defaulters", "slipped", "recovered") needs last month's file;
without it, only today's position is used (current buckets, quick wins).
"""
import math

import pandas as pd

from analysis import roll_flow as rf
from config import (
    ACTION_COLLECTION_GAP_PP, ACTION_HEADLINES_MAX, ACTION_HOTSPOT_MULTIPLE, ACTION_MIN_ACCOUNTS,
    ACTION_MIN_ROLLED, ACTION_RANGE_Z, ACTION_RISING_PP, ACTION_TOP_N, INSURANCE_EXP_ARREARS_MIN,
)
from utils import BUCKET_SCORE, insurance_only_mask, is_yes, to_num, unit_metrics

_GRAIN_COLS = {"region": ["RegionName"], "branch": ["Unit"], "executive": ["MNT NAME", "Unit"]}
_WORSE_THAN = {b: [c for c in rf.VALID_BUCKETS if BUCKET_SCORE[c] > BUCKET_SCORE[b]] for b in rf.VALID_BUCKETS}


def _cr(x: float) -> str:
    return f"₹{x / 1e7:,.2f} Cr"


# ── This month's transition rates (the basis for "if ignored" and forecasts) ──

def transition_rates(df: pd.DataFrame, matched: pd.DataFrame | None = None) -> dict | None:
    """Per last-month bucket: share of loans (and of last month's SOH) that
    moved to a worse bucket, and to NPA. None without last month's file."""
    m = rf._matched(df) if matched is None else matched
    if m.empty:
        return None
    out = {}
    for b in rf.VALID_BUCKETS:
        base = m[m["prev_bucket"] == b]
        soh = base["_prev_soh"].sum()
        worse, npa = base["_worse"], base["curr_bucket"] == "NPA"
        out[b] = {
            "n": len(base),
            "worse": worse.mean() if len(base) else 0.0,
            "worse_soh": base.loc[worse, "_prev_soh"].sum() / soh if soh else 0.0,
            "npa": npa.mean() if len(base) else 0.0,
            "npa_soh": base.loc[npa, "_prev_soh"].sum() / soh if soh else 0.0,
        }
    return out


def _expected(df: pd.DataFrame, rates: dict, key: str, exclude: tuple = ()) -> dict:
    """Expected loans / SOH making the `key` move next month if this month's
    rates repeat, with a likely range (normal approximation of a binomial)."""
    n_exp = soh_exp = var = 0.0
    soh = to_num(df, "SOH", fill=0)
    for b, grp in df.groupby("curr_bucket"):
        if b not in rates or b in exclude:
            continue
        p, p_soh = rates[b][key], rates[b][f"{key}_soh"]
        n_exp += len(grp) * p
        soh_exp += soh.loc[grp.index].sum() * p_soh
        var += len(grp) * p * (1 - p)
    spread = ACTION_RANGE_Z * math.sqrt(var)
    scale = soh_exp / n_exp if n_exp else 0.0
    return {"n": round(n_exp), "low": max(0, round(n_exp - spread)), "high": round(n_exp + spread),
            "soh": soh_exp, "soh_low": max(0.0, (n_exp - spread) * scale), "soh_high": (n_exp + spread) * scale}


def next_month_outlook(df: pd.DataFrame, rates: dict | None = None) -> dict | None:
    """If this month's roll rates repeat: expected new NPAs (from every non-NPA
    bucket) and new defaulters (from today's STD loans)."""
    rates = rates if rates is not None else transition_rates(df)
    if rates is None or "curr_bucket" not in df.columns:
        return None
    std_now = df[df["curr_bucket"] == "STD"]
    return {
        "new_npa": _expected(df, rates, "npa", exclude=("NPA",)),
        "new_defaulters": _expected(std_now, rates, "worse"),
    }


# ── Where a set of loans is concentrated ─────────────────────────────────────

def _top_share(loans: pd.DataFrame, col: str) -> tuple[str, int, float] | None:
    if loans.empty or col not in loans.columns:
        return None
    counts = loans[col].value_counts()
    return str(counts.index[0]), int(counts.iloc[0]), round(counts.iloc[0] / len(loans) * 100, 1)


def _worst_branch(by: pd.DataFrame, step: str) -> dict | None:
    """Branch with the highest rate for a roll step (min size applies)."""
    if by.empty:
        return None
    by = by[by["Matched Accounts"] >= ACTION_MIN_ACCOUNTS["branch"]]
    by = by[by[f"{step} | Accounts"] >= ACTION_MIN_ROLLED]
    if by.empty:
        return None
    r = by.sort_values(f"{step} | %", ascending=False).iloc[0]
    return {"branch": r["Branch"], "pct": float(r[f"{step} | %"]),
            "n": int(r[f"{step} | Accounts"]), "base": int(r[f"{step} | Base"])}


# ── Headlines ────────────────────────────────────────────────────────────────

_PULSE = (  # (label, unit_metrics column, a rise is bad)
    ("Delinquency", "Delinquency%", True),
    ("NPA", "NPA%", True),
    ("NPA (SOH)", "NPA% (SOH)", True),
    ("SMA-2", "SMA-2%", True),
    ("Collection", "Collection%", False),
)


def pulse(df_curr: pd.DataFrame, df_prev: pd.DataFrame) -> list[dict]:
    """The headline rates now vs last month: {label, now, prev, change, worse}."""
    now = unit_metrics(df_curr, [])
    if now.empty:
        return []
    prev = unit_metrics(df_prev, []) if df_prev is not None and len(df_prev) else pd.DataFrame()
    n, p = now.iloc[0], (prev.iloc[0] if not prev.empty else None)
    out = []
    for label, col, rise_is_bad in _PULSE:
        change = None if p is None else round(float(n[col] - p[col]), 2)
        out.append({"label": label, "now": float(n[col]), "prev": None if p is None else float(p[col]),
                    "change": change, "worse": None if not change else (change > 0) == rise_is_bad})
    return out


def headlines(df_curr: pd.DataFrame, df_prev: pd.DataFrame, ctx: dict | None = None) -> list[dict]:
    """Plain sentences: {tone: bad/good/info, title, detail, soh}. Problems
    first, by the SOH involved; one slot each is kept for the best good-news
    item and the quick-win item, so they're never crowded out."""
    ctx = ctx or _context(df_curr)
    items: list[dict] = []
    summary = ctx["summary"]
    if summary:
        moved = ctx["moved"]
        for step, from_b in rf.ROLL_STEPS.items():
            s = summary[step]
            if not s["n"]:
                continue
            loans = moved[(moved["prev_bucket"] == from_b) & moved["_worse"]]
            where = []
            top = _top_share(loans, "RegionName")
            if top:
                where.append(f"{top[2]:.0f}% of them are in {top[0]}")
            wb = _worst_branch(ctx["roll_by"]["branch"], step)
            if wb and s["pct"] and wb["pct"] >= ACTION_HOTSPOT_MULTIPLE * s["pct"]:
                where.append(f"worst branch {wb['branch']} at {wb['pct']:.1f}% ({wb['n']} of {wb['base']})")
            soh_note = ""
            if s["soh_pct"] - s["pct"] >= 2:
                soh_note = f" By SOH it's {s['soh_pct']:.1f}%: the bigger loans are slipping."
            title = {
                "STD → Behind": f"{s['n']:,} new defaulters",
                "1-30 → SMA-1+": f"{s['n']:,} loans slipped from 1-30 DPD to SMA-1 or worse",
                "SMA-1 → SMA-2+": f"{s['n']:,} loans slipped from SMA-1 to SMA-2 or worse",
                "SMA-2 → NPA": f"{s['n']:,} new NPAs from SMA-2",
            }.get(step, f"{s['n']:,} loans: {step}")
            items.append({
                "tone": "bad", "soh": s["soh_cr"] * 1e7, "title": title,
                "detail": (f"{s['pct']:.1f}% of last month's {from_b} loans, ₹{s['soh_cr']:,.2f} Cr SOH."
                           + soh_note + (" " + _sentence("; ".join(where)) if where else "")),
            })
        cure = summary["Back to STD"]
        if cure["n"]:
            items.append({"tone": "good", "soh": cure["soh_cr"] * 1e7,
                          "title": f"{cure['n']:,} loans recovered to STD",
                          "detail": f"{cure['pct']:.1f}% of loans that were behind last month, ₹{cure['soh_cr']:,.2f} Cr SOH."})

    if not df_curr.empty:
        quick = df_curr[insurance_only_mask(df_curr)]
        if len(quick):
            items.append({"tone": "info", "soh": 0.0,
                          "title": f"{len(quick):,} quick wins: behind only on the insurance charge",
                          "detail": (f"EMI paid, insurance/expense arrears over ₹{INSURANCE_EXP_ARREARS_MIN:,}; "
                                     f"{_cr(to_num(quick, 'SOH', fill=0).sum())} SOH. A cash or WCL adjustment clears them.")})

    by_tone = {t: sorted([h for h in items if h["tone"] == t], key=lambda h: -h["soh"]) for t in ("bad", "good", "info")}
    extras = by_tone["good"][:1] + by_tone["info"][:1]
    return by_tone["bad"][:ACTION_HEADLINES_MAX - len(extras)] + extras


def _sentence(text: str) -> str:
    """First letter upper-case, the rest untouched (names keep their case)."""
    return (text[:1].upper() + text[1:] + ".") if text else ""


# ── Focus first: action groups on today's loans ──────────────────────────────

def _focus_groups(df: pd.DataFrame) -> list[tuple[str, str, pd.Series]]:
    """(name, what to do, mask) in priority order: cheapest and most urgent first."""
    bucket = df["curr_bucket"] if "curr_bucket" in df.columns else pd.Series("", index=df.index)
    prev = df["prev_bucket"] if "prev_bucket" in df.columns else pd.Series(None, index=df.index)
    behind = to_num(df, "Arrears / EMI") > 0
    new_def = (prev == "STD") & bucket.isin(_WORSE_THAN["STD"])
    return [
        ("Quick wins: insurance charge only", "Settle the charge by cash or WCL adjustment.", insurance_only_mask(df)),
        ("New defaulters", "Call before the next due date: they were paid up last month.", new_def),
        ("SMA-2: last chance before NPA", "Field visit this week.", (bucket == "SMA-2") & ~new_def),
        ("SMA-1: stop the slide", "Get at least one EMI to keep them out of SMA-2.", (bucket == "SMA-1") & ~new_def),
        ("1-30 DPD: bring them current", "Reminder call, collect the overdue EMI.", (bucket == "1-30 DPD") & ~new_def),
        ("Non-starters", "Never paid the first EMI: check the customer and the disbursement.",
         is_yes(df, "Non Starter") & behind),
    ]


def focus_first(df: pd.DataFrame, rates: dict | None = None) -> list[dict]:
    """Each action group with loans, SOH and what's likely to slip next month
    if nothing is done (this month's roll rate for each loan's bucket)."""
    rates = rates if rates is not None else transition_rates(df)
    out = []
    for i, (name, action, mask) in enumerate(_focus_groups(df), start=1):
        grp = df[mask.fillna(False)]
        if grp.empty:
            continue
        risk = _expected(grp, rates, "worse") if rates and "curr_bucket" in grp.columns else None
        out.append({"rank": i, "name": name, "action": action, "loans": len(grp),
                    "soh": float(to_num(grp, "SOH", fill=0).sum()), "risk": risk})
    return out


# The columns that prove why a loan is in each group, so a manager can check
# it on the spot (e.g. an insurance-only loan: installment arrears 0, expense
# arrears above the threshold). Columns missing from a file are skipped.
_COLLECTED = "Month Collection (Excluding Reserve Collection)"
_RECEIPT = ["Last Receipt Date", "Last Receipt Amount", "SOH"]
EVIDENCE_COLS = {
    "Quick wins: insurance charge only": [
        "Month Due-Inst", "Month Due-Exp", "ARREARS AGAINST INST", "ARREARS AGAINST EXP",
        "ARREARS AGAINST BC", "ARREARS AGAINST PC", "Closing Arrears", "Arrears / EMI", *_RECEIPT],
    "New defaulters": [
        "prev_bucket", "curr_bucket", "Month Due-Inst", "Month Due-Exp", _COLLECTED,
        "Closing Arrears", "Arrears / EMI", *_RECEIPT],
    "SMA-2: last chance before NPA": ["curr_bucket", "Arrears / EMI", "Closing Arrears", "Month Due-Inst", _COLLECTED, *_RECEIPT],
    "SMA-1: stop the slide": ["curr_bucket", "Arrears / EMI", "Closing Arrears", "Month Due-Inst", _COLLECTED, *_RECEIPT],
    "1-30 DPD: bring them current": ["curr_bucket", "Arrears / EMI", "Closing Arrears", "Month Due-Inst", _COLLECTED, *_RECEIPT],
    "Non-starters": ["Non Starter", "Ag_Date", "ParentLDueDate", "Arrears / EMI", "Closing Arrears", *_RECEIPT],
}
_IDENTITY = ["Loan No", "Cust Name", "Cust Mob No", "RegionName", "Unit", "MNT NAME"]
_RENAME = {"RegionName": "Region", "Unit": "Branch", "MNT NAME": "Executive",
           "prev_bucket": "Last Month Bucket", "curr_bucket": "Bucket Now", _COLLECTED: "Collected This Month"}


def call_lists(df: pd.DataFrame) -> list[dict]:
    """One table per focus group, in priority order: {name, loans}. Each table
    has who to call, the columns that prove the loan belongs in the group
    (EVIDENCE_COLS) and "Also In" (its other groups). Largest SOH first."""
    groups = [(name, mask.fillna(False)) for name, _, mask in _focus_groups(df)]
    out = []
    for name, mask in groups:
        hit = df[mask]
        if hit.empty:
            continue
        also = pd.Series("", index=hit.index)
        for other, omask in groups:
            if other != name:
                flag = omask.loc[hit.index]
                also = also.where(~flag, also + "; " + other)
        also = also.str.lstrip("; ")
        cols = [c for c in [*_IDENTITY, *EVIDENCE_COLS[name]] if c in hit.columns]
        table = hit[cols].assign(**{"Also In": also}).assign(SOH=to_num(hit, "SOH"))
        table = table.sort_values("SOH", ascending=False).rename(columns=_RENAME).reset_index(drop=True)
        ident = [_RENAME.get(c, c) for c in _IDENTITY if c in hit.columns]
        table = table[[*ident, "Also In", *[c for c in table.columns if c not in ident and c != "Also In"]]]
        out.append({"name": name, "loans": table})
    return out


def unique_call_count(lists: list[dict]) -> int:
    """Distinct loans across every call list (a loan can be in several)."""
    ids = [t["loans"]["Loan No"] for t in lists if "Loan No" in t["loans"].columns]
    return int(pd.concat(ids).nunique()) if ids else 0


# ── Needs attention: branches / executives, with reasons in words ────────────

def attention(df_curr: pd.DataFrame, df_prev: pd.DataFrame, grain: str, ctx: dict | None = None) -> dict:
    """{"worst": [...], "improving": [...]} for branches or executives. A unit
    is flagged for each reason that applies; most reasons first, then the SOH
    that slipped. Units below ACTION_MIN_ACCOUNTS aren't judged."""
    cols = _GRAIN_COLS[grain]
    if not set(cols) <= set(df_curr.columns):
        return {"worst": [], "improving": []}
    m = unit_metrics(df_curr, cols)
    if m.empty:
        return {"worst": [], "improving": []}
    port = unit_metrics(df_curr, []).iloc[0]
    prev = (unit_metrics(df_prev, cols).set_index(cols)["Delinquency%"].to_dict()
            if df_prev is not None and len(df_prev) and set(cols) <= set(df_prev.columns) else {})
    ctx = ctx or _context(df_curr)
    roll = ctx["roll_by"][grain]
    roll_port = ctx["summary"] or {}
    id_names = {"RegionName": "Region", "Unit": "Branch", "MNT NAME": "Executive"}
    roll_idx = roll.set_index([id_names[c] for c in cols]).to_dict("index") if not roll.empty else {}
    region = df_curr.groupby(cols)["RegionName"].first().to_dict() if "RegionName" in df_curr.columns else {}

    worst, improving = [], []
    for _, r in m.iterrows():
        key = tuple(r[c] for c in cols) if len(cols) > 1 else r[cols[0]]
        if r["Accounts"] < ACTION_MIN_ACCOUNTS[grain]:
            continue
        reasons, soh_risk = [], 0.0
        rr = roll_idx.get(key)
        if rr is not None:
            for step in list(rf.ROLL_STEPS)[:2] + ["SMA-2 → NPA"]:
                pct, n_, base = rr[f"{step} | %"], int(rr[f"{step} | Accounts"]), int(rr[f"{step} | Base"])
                avg = roll_port.get(step, {}).get("pct", 0)
                if n_ >= ACTION_MIN_ROLLED and avg and pct >= ACTION_HOTSPOT_MULTIPLE * avg:
                    reasons.append(f"{step} {pct:.1f}% ({n_} of {base}), {pct / avg:.1f}x the average")
                    soh_risk += rr[f"{step} | SOH (Cr)"] * 1e7
        prev_d = prev.get(key)
        if prev_d is not None and not pd.isna(prev_d):
            d = r["Delinquency%"] - prev_d
            if d >= ACTION_RISING_PP:
                reasons.append(f"delinquency up {d:.1f} pts to {r['Delinquency%']:.1f}%")
            elif d <= -ACTION_RISING_PP:
                improving.append({"name": _unit_name(key, cols), "region": str(region.get(key, "")),
                                  "accounts": int(r["Accounts"]),
                                  "why": f"delinquency down {abs(d):.1f} pts to {r['Delinquency%']:.1f}%",
                                  "score": d})
        if r["Collection%"] <= port["Collection%"] - ACTION_COLLECTION_GAP_PP:
            reasons.append(f"collection {r['Collection%']:.1f}% vs {port['Collection%']:.1f}% overall")
        if reasons:
            worst.append({"name": _unit_name(key, cols), "region": str(region.get(key, "")),
                          "accounts": int(r["Accounts"]), "reasons": reasons, "soh_risk": soh_risk})
    worst.sort(key=lambda u: (-len(u["reasons"]), -u["soh_risk"]))
    improving.sort(key=lambda u: u["score"])
    return {"worst": worst[:ACTION_TOP_N], "improving": improving[:3]}


def _unit_name(key, cols) -> str:
    return f"{key[0]} ({key[1]})" if len(cols) > 1 else str(key)


def _context(df: pd.DataFrame) -> dict:
    """The roll inputs every section shares, computed once."""
    has_roll = rf.has_roll_data(df)
    moved = rf._matched(df)
    return {
        "moved": moved,
        "summary": rf.roll_steps_summary(df),
        "rates": transition_rates(df, moved),
        "roll_by": {g: (rf.roll_steps_by(df, g) if has_roll else pd.DataFrame()) for g in ("branch", "executive")},
    }


def build(df_curr: pd.DataFrame, df_prev: pd.DataFrame) -> dict:
    """Everything the action overview shows, in one call."""
    ctx = _context(df_curr)
    return {
        "pulse": pulse(df_curr, df_prev),
        "headlines": headlines(df_curr, df_prev, ctx),
        "outlook": next_month_outlook(df_curr, ctx["rates"]),
        "focus": focus_first(df_curr, ctx["rates"]),
        "attention": {g: attention(df_curr, df_prev, g, ctx) for g in ("branch", "executive")},
        "calls": call_lists(df_curr),
        "roll_summary": ctx["summary"],
        "has_prev": rf.has_roll_data(df_curr) and df_prev is not None and len(df_prev) > 0,
    }
