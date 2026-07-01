"""CollectionIQ compiler  -  lowers a logical query (IR-1) into the existing
physical step-plan (IR-2) that agents.plan_executor.execute_plan runs.

This is the deterministic layer that recovers correctness: the LLM emits a flat
logical query that only *references* business concepts/metrics/dimensions by name,
and the compiler expands those references  -  fully, every time  -  into engine
primitives. A multi-condition concept (e.g. colending_at_risk) is expanded to its
complete condition set here, so it can never silently lose a leg the way prompt-
driven planning could.

Phase 1 scope: filtered aggregations and pure filters over loan-grain concepts and
metrics, plus a grain-ambiguity detector. No optimizer, no new execution code  - 
the output is the existing plan format.
"""

from compiler.core import compile_logical, grain_issues

__all__ = ["compile_logical", "grain_issues"]
