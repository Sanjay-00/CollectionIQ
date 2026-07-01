"""CollectionIQ knowledge registry  -  the deterministic source of truth that the
v2 compiler will read instead of the LLM re-deriving business logic from prose.

Two coherent layers (see the architecture design of record):

- semantic_model  -  STRUCTURE: entities, their key columns, and the grain lattice
  ("how the data is connected"). For this single denormalized table it is a grain
  lattice + entity-key map, NOT a join graph.
- ontology  -  VOCABULARY: named business concepts (multi-condition rules) and
  metrics, with deterministic definitions ("what" each concept means).

Phase 0 status: these structures are the canonical home for business logic.
PRIORITY_RULES has been migrated here and is re-exported from agents.domain_expert
for back-compat. The broader CONCEPTS/METRICS/semantic-model structures are
additive and not yet wired into the LLM prompt, so behavior is unchanged.
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
]
