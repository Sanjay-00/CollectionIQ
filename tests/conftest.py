import pytest


@pytest.fixture(autouse=True)
def _empty_plan_cache():
    """The AI Query plan cache is process wide (agents/plan_cache.py), so a
    plan kept by one test would be reused by the next test asking the same
    question with a different scripted planner. Every test starts empty."""
    from agents.plan_cache import PLAN_CACHE
    PLAN_CACHE.clear()
    yield
    PLAN_CACHE.clear()
