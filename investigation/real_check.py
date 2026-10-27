"""
Phase 1 acceptance on real files (run locally; prints counts, rates and
dimension names only, never loan rows or customer fields):

    python -m investigation.real_check <last month file> <this month file> <YYYY-MM last> <YYYY-MM this>
"""
from __future__ import annotations

import sys
import warnings

from investigation.baseline import run_baseline
from investigation.evidence import EvidenceStore
from investigation.report import build_report
from investigation.panel import make_pair
from investigation.prepare import prepare
from investigation.tools import flow_bridge, metric_change, scan_dimensions
from utils import load_and_validate


def _load(path: str):
    with open(path, "rb") as f:
        df, errs = load_and_validate.__wrapped__(f)
    if errs:
        raise SystemExit(f"{path}: {errs}")
    return df


def main(prev_path: str, curr_path: str, prev_month: str, curr_month: str) -> int:
    warnings.filterwarnings("ignore")
    full = make_pair(_load(curr_path), _load(prev_path), curr_month, prev_month)
    store = EvidenceStore()

    prep = prepare(store, full)
    files = store.all()[0].summary
    print(f"DATA HEALTH (whole files): {files['status']}")
    for f in files["findings"]:
        print(f"  [{f['status']:<4}] {f['check']}: {f['detail']}")
    if prep.adjustment:
        a = prep.adjustment.summary
        print("LIKE FOR LIKE, left out of both months:",
              "; ".join(f"{x['unit']} ({x['reason']}, {x['loans_prev']:,} / {x['loans_curr']:,} loans)" for x in a["excluded"]))
        print(f"  change on the whole files {a['full_change_pp']:+.2f}, like for like {a['like_for_like_change_pp']:+.2f} % points")
    print(f"STATUS: {prep.status} (health of the pair compared: {prep.health.summary['status']})")
    if prep.status != "ok":
        return 1
    pair = prep.pair

    s = metric_change(store, pair).summary
    for side in ("prev", "curr"):
        v = s[side]
        print(f"{side}: running {v['running']:,}  eligible {v['eligible']:,}  30+ {v['numerator']:,}  "
              f"rate {v['rate']:.2f}%  all-running {v['rate_all_running']:.2f}%")
    print(f"change {s['change_pp']:+.2f} % points (relative {s['change_rel_pct']:+.1f}%); "
          f"30+ loans {s['numerator_effect_pp']:+.2f}, base {s['base_effect_pp']:+.2f}; "
          f"all-running {s['all_running_change_pp']:+.2f}")

    b = flow_bridge(store, pair).summary
    print("BRIDGE:", ", ".join(f"{k} {v:,}" for k, v in b["lines"].items()))
    print("BASE:  ", ", ".join(f"{k} {v:,}" for k, v in b["base_lines"].items()))
    print("left the book, last bucket:", b["left_book_last_bucket"])

    scan = scan_dimensions(store, pair).summary
    print(f"SCAN: material {scan['material']}; unavailable {[u['dimension'] + ': ' + u['reason'] for u in scan['unavailable']]}")
    for r in scan["ranked"][:6]:
        top = "; ".join(f"{t['segment']} {t['contribution_pp']:+.2f} (rate {t['rate_change_pp'] if t['rate_change_pp'] is None else round(t['rate_change_pp'], 2)}, "
                        f"mix {t['mix_pp']:+.2f})" for t in r["top"])
        fmt = lambda v: "     -" if v is None else f"{v:6.1f}"     # None when the change isn't material
        print(f"  {r['label']:<20} concentration {fmt(r['concentration_pct'])}  coverage {fmt(r['coverage_pct'])}  "
              f"mix {r['mix_pp']:+.2f} rate {r['rate_pp']:+.2f}  | {top}")

    failed = [(e.result_id, e.tool, [c.name for c in e.checks if not c.passed]) for e in store.all() if not e.checks_passed]
    print(f"CHECKS: {len(store.all())} results, failed: {failed or 'none'}")

    result, bstore = run_baseline(full.curr, full.prev, curr_month, prev_month)
    print(f"\n===== BASELINE ANSWER ({result.status}, {result.tool_calls} tool results, {result.seconds:.0f}s) =====")
    print(build_report(result, bstore)["markdown"])
    bad = [e.result_id for e in bstore.all() if not e.checks_passed]
    print(f"\nbaseline checks failed: {bad or 'none'}")
    return 1 if failed or bad else 0


if __name__ == "__main__":
    if len(sys.argv) != 5:
        raise SystemExit(__doc__)
    raise SystemExit(main(*sys.argv[1:]))
