"""
One noise band for a rate change, used for a whole scope (metric_change,
scan materiality) and for each segment (mix_rate). Paired: most loans are the
same loans seen twice, so a loan in both months adds noise only if it crossed
into or out of the numerator (b + c); loans in one month only are
independent draws. Treating the months as independent samples overstated
noise so much that a planted 2.5-point rise in 4,000 loans read as noise.
"""
from __future__ import annotations

import math

from config import INVESTIGATION_NOISE_SE


def rate_noise_pp(n0: int, k0: int, n1: int, k1: int, m: int, b: int, c: int) -> float | None:
    """INVESTIGATION_NOISE_SE standard errors of r1 - r0, in % points.
    n, k: eligible loans and numerator each month; m: loans in both months
    (and the same segment); b, c: of those, left / entered the numerator.
    None when either month has no loans."""
    if not n0 or not n1:
        return None
    p0, p1 = k0 / n0, k1 / n1
    u0, u1 = n0 - m, n1 - m
    var = (b + c) / (n0 * n1) + u0 * p0 * (1 - p0) / n0 ** 2 + u1 * p1 * (1 - p1) / n1 ** 2
    return INVESTIGATION_NOISE_SE * 100 * math.sqrt(var)
