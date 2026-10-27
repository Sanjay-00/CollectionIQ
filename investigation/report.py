"""
The answer in plain words (spec 11), built in code from a result and its
evidence: every number is read from a tool result, causal verbs are never
used ("sits in", "accounts for"), and a like-for-like adjustment is stated
before any finding. Wording rules: "delinquent", "MoM", "% points", no dashes.
"""
from __future__ import annotations

from config import INVESTIGATION_NO_CHANGE_PP
from investigation.dimensions import DIMENSIONS
from investigation.evaluation import Driver, InvestigationResult
from investigation.evidence import EvidenceStore

_LABEL = "30+"
_BRIDGE_WORDS = (("in_from_std", "+", "from STD"), ("in_from_1_30", "+", "from 1-30 DPD"),
                 ("in_from_no_emi", "+", "from no EMI yet"), ("new_to_book", "+", "new to the book"),
                 ("cured_to_std", "-", "cured to STD"), ("cured_to_1_30", "-", "cured to 1-30 DPD"),
                 ("became_no_emi", "-", "back to no EMI"), ("left_book", "-", "left the running book"))


def _first(store: EvidenceStore, tool: str):
    return next((e for e in store.all() if e.tool == tool), None)


def _dim(dim_id: str) -> str:
    return DIMENSIONS[dim_id].label if dim_id in DIMENSIONS else dim_id


def _row(store: EvidenceStore, d: Driver):
    seg = store.get(d.evidence_id).tables["segments"]
    hit = seg[seg["segment"].astype(str).str.upper() == d.segment.upper()]
    return None if hit.empty else hit.iloc[0]


def _pp(v: float) -> str:
    return f"{v:+.2f} % points"


def _adjustment(store: EvidenceStore) -> str:
    adj = _first(store, "like_for_like")
    if adj is None:
        return ""
    s = adj.summary
    left = "; ".join(f"{x['unit']} ({x['reason']}, {max(x['loans_prev'], x['loans_curr']):,} loans)" for x in s["excluded"])
    return (f"Compared like for like: left out of both months {left}. On the whole files the change is "
            f"{_pp(s['full_change_pp'])}; like for like it is {_pp(s['like_for_like_change_pp'])}.")


def _data_notes(store: EvidenceStore) -> str:
    notes, seen = [], set()
    for ev in (e for e in store.all() if e.tool == "data_health"):
        for f in ev.summary["findings"]:
            if f["status"] in ("warn", "halt", "info") and f["check"] not in seen:
                seen.add(f["check"])
                notes.append(f"- {f['detail']}")
    return "\n".join(notes)


def _rates(r) -> str:
    if r["r0"] is not None and r["r1"] is not None:
        return f"its {_LABEL} rate {r['r0']:.2f}% to {r['r1']:.2f}%"
    if r["r1"] is not None:
        return f"new this month, {_LABEL} rate {r['r1']:.2f}%"
    return f"gone this month, {_LABEL} rate was {r['r0']:.2f}%"


def _where(result: InvestigationResult, store: EvidenceStore, total: float | None) -> str:
    """total None: the change wasn't material, so no share of it is printed
    (review: dividing by a stand-in gave "87% of the change" from nothing)."""
    lines = []
    for d in result.drivers:
        r = _row(store, d)
        if r is None:
            continue
        share = f", {100 * r['contribution_pp'] / total:.0f}% of the change" if total else ""
        lines.append(f"- {_dim(d.dimension)} {d.segment}: {_pp(r['contribution_pp'])}{share}; "
                     f"{_rates(r)}; share of loans {r['s0'] * 100:.1f}% to {r['s1'] * 100:.1f}% "
                     f"(mix {_pp(r['mix_pp'] + r['new_pp'] + r['gone_pp'])}, rate {_pp(r['rate_pp'])}).")
    for d in result.deep:
        r = _row(store, d)
        if r is not None:
            lines.append(f"- Inside it, {_dim(d.dimension)} {d.segment}: {_pp(r['contribution_pp'])} of that "
                         f"segment's change; {_rates(r)}.")
    return "\n".join(lines)


def _coverage(result: InvestigationResult, store: EvidenceStore, total: float) -> float:
    rows = [_row(store, d) for d in result.drivers]
    return 100 * sum(r["contribution_pp"] for r in rows if r is not None) / total if total else 0.0


def _checks(store: EvidenceStore) -> str:
    out = []
    for ev in (e for e in store.all() if e.tool == "controlled_decompose"):
        s = ev.summary
        verdict = "keeps the same direction" if s["persists"] else "does not keep the same direction"
        out.append(f"- {_dim(s['dimension'])} {s['segment']} {verdict} inside {_dim(s['control'])} groups "
                   f"holding {s['persisting_loans']:,} of its {s['tested_loans']:,} tested loans. "
                   "This tests a rival explanation; it is not proof of cause.")
    for ev in (e for e in store.all() if e.tool == "rival_untestable"):
        a = ev.args
        out.append(f"- The rival explanation ({_dim(a['control'])}) couldn't be tested for {_dim(a['dimension'])} "
                   f"{a['segment']}: {ev.summary['reason']}.")
    return "\n".join(out)


def _next_steps(lines: dict) -> str:
    steps = []
    if lines["in_from_std"]:
        steps.append(f"- Action Lists, New defaulters: {lines['in_from_std']:,} loans went from STD to {_LABEL} this month.")
    if lines["in_from_1_30"]:
        steps.append(f"- Action Lists, 1-30 → SMA-1+: {lines['in_from_1_30']:,} loans slipped from 1-30 DPD to {_LABEL}.")
    return "\n".join(steps)


def build_report(result: InvestigationResult, store: EvidenceStore) -> dict:
    rep = {k: "" for k in ("headline", "adjustment", "what_changed", "where", "checks", "not_tested",
                           "data_notes", "next_steps")}
    rep["data_notes"] = _data_notes(store)
    rep["adjustment"] = _adjustment(store)
    if result.status == "halted_data":
        halt = next((f["detail"] for e in store.all() if e.tool == "data_health"
                     for f in e.summary["findings"] if f["status"] == "halt"), "the data checks failed")
        rep["headline"] = f"Stopped: these two files can't be compared month on month. {halt.rstrip('.')}."
        return _markdown(rep)

    c = _first(store, "metric_change").summary
    lines = _first(store, "flow_bridge").summary["lines"]
    total = c["change_pp"]
    moved = "rose" if total > 0 else "fell" if total < 0 else "did not move"
    base = (f"{_LABEL} {moved} from {c['prev']['rate']:.2f}% to {c['curr']['rate']:.2f}% ({_pp(total)})"
            + (" like for like" if rep["adjustment"] else ""))
    rep["what_changed"] = (
        f"- {_LABEL} loans {c['prev']['numerator']:,} of {c['prev']['eligible']:,} with an EMI due, then "
        f"{c['curr']['numerator']:,} of {c['curr']['eligible']:,}.\n"
        f"- More or fewer {_LABEL} loans: {_pp(c['numerator_effect_pp'])}; a bigger or smaller base: {_pp(c['base_effect_pp'])}.\n"
        + (f"- Bridge: {lines['opening']:,} last month, " + ", ".join(
            f"{sign}{lines[key]:,} {words}" for key, sign, words in _BRIDGE_WORDS if lines[key])
           + f": {lines['closing']:,} now." if any(lines[k] for k, _, _ in _BRIDGE_WORDS)
           else f"- No loans moved into or out of {_LABEL}: {lines['closing']:,} both months."))
    scan = _first(store, "scan_dimensions")
    if scan and scan.summary["unavailable"]:
        rep["not_tested"] = "\n".join(f"- {_dim(u['dimension'])}: {u['reason']}." for u in scan.summary["unavailable"])
    rep["next_steps"] = _next_steps(lines)

    if result.status == "no_material_change":
        noise = c.get("noise_pp") or 0.0
        if abs(total) > noise:
            rep["headline"] = f"{base}: under the {INVESTIGATION_NO_CHANGE_PP:.2f} % point line for a material change."
        else:
            rep["headline"] = f"{base}: within the normal month-to-month movement for this book (about ±{noise:.2f})."
        if result.drivers:
            rep["headline"] += " Some segments moved in opposite directions; see below."
            rep["where"] = _where(result, store, None)
        return _markdown(rep)

    if result.mechanism == "base":
        rep["headline"] = (f"{base}. The {_LABEL} loans barely changed ({_pp(c['numerator_effect_pp'])}); "
                           f"the base of loans with an EMI due changed ({_pp(c['base_effect_pp'])}).")
    elif not result.drivers:
        ins = lines["in_from_std"] + lines["in_from_1_30"]
        cures = lines["cured_to_std"] + lines["cured_to_1_30"]
        rep["headline"] = (f"{base}. No single slice of the book accounts for it: {ins:,} loans rolled into {_LABEL} "
                           f"and {cures:,} cured.")
    else:
        names = f"{_dim(result.drivers[0].dimension)} " + " and ".join(d.segment for d in result.drivers)
        up = total > 0
        kind = ((f"more loans from segments with a {'higher' if up else 'lower'} {_LABEL} rate")
                if result.mechanism == "mix" else f"the same loans with {'more' if up else 'fewer'} EMIs overdue")
        rep["coverage_pct"] = _coverage(result, store, total)
        lead = "Most of it sits in" if rep["coverage_pct"] >= 60 else "The largest parts sit in"
        rep["headline"] = f"{base}. {lead} {names}: {kind}."
        rep["where"] = _where(result, store, total)
        rep["checks"] = _checks(store)
    return _markdown(rep)


_TITLES = (("adjustment", "Like for like"), ("what_changed", "What changed"), ("where", "Where the change sits"),
           ("checks", "Checks on the explanation"), ("not_tested", "Not tested"), ("data_notes", "Data notes"),
           ("next_steps", "What to do next"))


def _markdown(rep: dict) -> dict:
    parts = [f"**{rep['headline']}**"]
    parts += [f"#### {title}\n{rep[key]}" for key, title in _TITLES if rep[key]]
    rep["markdown"] = "\n\n".join(parts)
    return rep
