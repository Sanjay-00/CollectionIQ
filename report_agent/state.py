import threading
from typing import TypedDict, Any, Optional

# ── Large-object side channel ────────────────────────────────────────────────
# Same fix as graph.py's own _stash_large/_fetch_large (the AI Query pipeline
# hit this exact bug first): df_curr/df_prev are full DataFrames -- on a real
# ~60k-row file, pickling df_curr alone runs ~60MB, df_prev another ~60MB,
# ~120MB combined -- 6x LangSmith's 20MB trace-payload limit. LangGraph's own
# tracing serializes the COMPLETE state on every node transition, independent
# of and not controllable via any individual node's @traceable process_inputs/
# process_outputs -- so leaving these in ReportState meant every one of the
# report graph's node transitions produced an oversized trace that failed to
# upload and retried repeatedly, adding real wall-clock latency on top of the
# actual (and actually small -- ~1,200 tokens) Gemini calls. Fix: never put
# them IN the traced state. run_report() stashes them here instead, keyed per
# thread (one report runs synchronously per thread, so no cross-report
# collision risk -- same reasoning as graph.py's _tls). Only
# portfolio_analyzer_node reads them back, via _fetch_large; every other node
# in this graph never touched df_curr/df_prev even before this fix.
_tls = threading.local()


def _stash_large(key: str, value) -> None:
    if not hasattr(_tls, "large_data"):
        _tls.large_data = {}
    _tls.large_data[key] = value


def _fetch_large(state: dict, key: str, default=None):
    """Check the thread-local stash first (the real path via run_report()); if
    nothing was stashed (e.g. a node function called directly with a hand-built
    state dict, as tests do), fall back to reading the field straight out of
    state -- preserves backward compatibility for any caller that still puts
    the real object directly in state."""
    stash = getattr(_tls, "large_data", {})
    if key in stash:
        return stash[key]
    return state.get(key, default)


class ReportState(TypedDict):
    # Inputs
    df_curr: Any
    df_prev: Any
    curr_month: str
    prev_month: Optional[str]
    enabled_sections: list
    filters_applied: dict
    skip_ai: bool

    # Node 1  -  portfolio_analyzer
    section_data: dict

    # Node 2  -  risk_narrator
    executive_narrative: str
    action_plan: str
    ai_skipped: bool

    # Node 3  -  report_builder
    html_report: str

    # Node 4  -  email_dispatcher
    email_to: str
    email_sent: bool
    email_error: str

    # LangSmith trace ID
    run_id: str

    # Error
    error: str
