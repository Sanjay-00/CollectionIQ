"""Local, append-only log of AI Query outcomes.

Purpose: find out what real users ask that the registry vocabulary
(registry/ontology.py's CONCEPTS/METRICS, registry/views.py's VIEWS) doesn't
cover yet, so it can grow from evidence instead of guesswork. This is the
concrete fix for "the system feels hardcoded and fails at unknown queries" -
every failure/clarification/fallback here is a candidate for a new concept,
metric, or view.

Local disk only. Never a database, never sent anywhere beyond this machine
(separate from, and in addition to, the LangSmith tracing already wired into
the Gemini calls themselves via @traceable). Logging must never be able to
break a real query, so every failure mode here is swallowed.
"""
import datetime
import json
from pathlib import Path

from config import QUERY_LOG_ENABLED, QUERY_LOG_PATH, QUERY_LOG_RETENTION_DAYS

# How often _prune_old_entries actually does the read+rewrite work, tracked via
# a sidecar marker file next to the log. Bounds the cost of retention to about
# once/day regardless of query volume, instead of paying a full file rewrite on
# every single logged query.
_PRUNE_CHECK_INTERVAL = datetime.timedelta(days=1)


def classify_outcome(state: dict) -> str:
    """One-word outcome label for a finished QueryState (graph.py).

    Checked in this order because a query can satisfy more than one condition
    at once (e.g. priority_mode queries never set ir1["view"]) - error and
    clarification are always the most specific/important signal when present.
    """
    if state.get("needs_clarification"):
        return "clarification"
    if state.get("error"):
        return "error"
    if state.get("priority_mode"):
        return "priority_mode"
    if (state.get("ir1") or {}).get("view"):
        return "view_hit"
    return "compiled_ok"


def _prune_old_entries(path: Path) -> None:
    """Drop lines older than QUERY_LOG_RETENTION_DAYS, at most once per
    _PRUNE_CHECK_INTERVAL. An unparseable line is dropped rather than kept
    (retention is a hygiene policy, not a correctness guarantee) or allowed to
    crash pruning for every other line."""
    marker = path.with_name(path.name + ".pruned_at")
    now = datetime.datetime.now()
    if marker.exists():
        last = datetime.datetime.fromisoformat(marker.read_text().strip())
        if now - last < _PRUNE_CHECK_INTERVAL:
            return

    if path.exists():
        cutoff = now - datetime.timedelta(days=QUERY_LOG_RETENTION_DAYS)
        kept = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    ts = datetime.datetime.fromisoformat(json.loads(line)["ts"])
                except Exception:
                    continue
                if ts >= cutoff:
                    kept.append(line)
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(kept) + ("\n" if kept else ""))

    marker.write_text(now.isoformat())


def log_query_outcome(state: dict) -> None:
    """Append one JSON line for a finished AI Query run. Never raises."""
    if not QUERY_LOG_ENABLED:
        return
    try:
        path = Path(QUERY_LOG_PATH)
        path.parent.mkdir(parents=True, exist_ok=True)
        _prune_old_entries(path)

        ir1 = state.get("ir1") or {}
        view = ir1.get("view")
        result_df = state.get("result_df")

        entry = {
            "ts":                      datetime.datetime.now().isoformat(timespec="seconds"),
            "run_id":                  state.get("run_id", ""),
            "query":                   state.get("query", ""),
            "outcome":                 classify_outcome(state),
            "intent":                  ir1.get("intent", ""),
            "view":                    view.get("name") if isinstance(view, dict) else view,
            "error":                   state.get("error", ""),
            "clarification_question":  state.get("clarification_question", ""),
            "result_rows":             int(len(result_df)) if result_df is not None else 0,
        }

        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        pass
