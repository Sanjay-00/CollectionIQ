"""CollectionIQ knowledge registry  -  the deterministic source of truth the v2
compiler reads instead of the LLM re-deriving business logic from prose.

Three coherent layers (see the architecture design of record):

- semantic_model  -  STRUCTURE: entities, their key columns, and the grain lattice
  ("how the data is connected"). For this single denormalized table it is a grain
  lattice + entity-key map, NOT a join graph.
- ontology  -  VOCABULARY: named business concepts (multi-condition rules) and
  metrics, with deterministic definitions ("what" each concept means).
- views  -  FAST PATH: precomputed analysis/ answers the AI Query pipeline can
  return directly for a matching question, bypassing the compiler entirely.

Status: this is the live, sole vocabulary source for agents/logical_planner.py's
prompt (via build_catalog()/build_views_catalog()) and for compiler/core.py's
lowering -- not an additive/inert scaffold. PRIORITY_RULES is also re-exported
from agents.domain_expert for back-compat with that module's still-live helpers.
"""

from registry.semantic_model import (
    ENTITIES,
    DIMENSIONS,
    TIME_MODEL,
    entity_key,
    grain_level,
    is_coarser,
    resolve_dimension,
)
from registry.ontology import CONCEPTS, METRICS, PRIORITY_RULES, ENTITY_CONCEPTS
from registry.views import VIEWS

__all__ = [
    "ENTITIES",
    "DIMENSIONS",
    "TIME_MODEL",
    "entity_key",
    "grain_level",
    "is_coarser",
    "resolve_dimension",
    "CONCEPTS",
    "METRICS",
    "PRIORITY_RULES",
    "ENTITY_CONCEPTS",
    "VIEWS",
]
