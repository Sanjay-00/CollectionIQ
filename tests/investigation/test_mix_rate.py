"""mix_rate: a change split into mix (more loans in riskier segments) and rate
(the same segments got worse), plus new and gone segments (spec 7.4)."""
import numpy as np
import pandas as pd
import pytest

from investigation.evidence import EvidenceStore, ToolError
from investigation.panel import make_pair
from investigation.tools import mix_rate


def _month(spec, prefix):
    """spec: {segment: (loans, of which 30+)} -> one month's loans (SegmentName = product)."""
    rows = []
    for seg, (n, k) in spec.items():
        rows += [{"SegmentName": seg, "curr_bucket": "NPA" if i < k else "STD"} for i in range(n)]
    df = pd.DataFrame(rows)
    df["Loan No"] = [f"{prefix}{i}" for i in range(len(df))]
    df["Loan Status"] = "RUN"
    return df


def _run(prev_spec, curr_spec, min_size=1, **kw):
    pair = make_pair(_month(curr_spec, "C"), _month(prev_spec, "P"), "2026-09", "2026-08")
    return mix_rate(EvidenceStore(), pair, "product", min_size=min_size, **kw)


def test_pure_mix_case():
    e = _run({"A": (80, 8), "B": (20, 10)}, {"A": (60, 6), "B": (40, 20)})
    s = e.summary
    assert s["total_change_pp"] == pytest.approx(8.0)
    assert s["mix_pp"] == pytest.approx(8.0) and s["rate_pp"] == pytest.approx(0.0)
    assert e.tool == "mix_rate" and e.checks_passed


def test_pure_rate_case():
    s = _run({"A": (50, 5), "B": (50, 25)}, {"A": (50, 5), "B": (50, 35)}).summary
    assert s["mix_pp"] == pytest.approx(0.0) and s["rate_pp"] == pytest.approx(10.0) == s["total_change_pp"]
    assert s["top"][0]["segment"] == "B" and s["top"][0]["rate_change_pp"] == pytest.approx(20.0)


def test_new_and_gone_segments_are_their_own_terms():
    e = _run({"A": (50, 5), "OLD": (50, 25)}, {"A": (50, 5), "NEW": (50, 40)})
    seg = e.tables["segments"].set_index("segment")
    assert seg.loc["NEW", "r0"] is None and seg.loc["OLD", "r1"] is None      # undefined, never 0
    # Measured against last month's book rate (30%): NEW (80%) adds 0.5 x 50,
    # OLD (50%) leaving takes 0.5 x 20 off.
    assert e.summary["new_pp"] == pytest.approx(25.0) and e.summary["gone_pp"] == pytest.approx(-10.0)
    assert e.checks_passed


def test_a_growing_segment_below_the_book_rate_pulls_the_rate_down():
    # Review C2: CAR's share grew while its 30+ rate (20%) sat below the
    # book's (30%); its mix term was +, so it was named a driver of a rise.
    e = _run({"CAR": (500, 100), "TRACTOR": (500, 200)}, {"CAR": (700, 140), "TRACTOR": (500, 250)})
    seg = e.tables["segments"].set_index("segment")
    assert seg.loc["CAR", "mix_pp"] < 0 and seg.loc["CAR", "contribution_pp"] < 0
    assert seg.loc["TRACTOR", "contribution_pp"] > 0 and e.checks_passed


@pytest.mark.parametrize("seed", range(20))
def test_reconciles_on_random_books(seed):
    rng = np.random.default_rng(seed)
    segs = [f"S{i}" for i in range(rng.integers(3, 13))]

    def spec():
        out = {}
        for s in segs:
            if rng.random() < 0.15:
                continue                                    # this segment absent this month
            n = int(rng.integers(1, 400))
            out[s] = (n, int(rng.integers(0, n + 1)))
        return out or {segs[0]: (10, 3)}
    assert _run(spec(), spec()).checks_passed


def test_small_segments_are_grouped():
    e = _run({"A": (30, 3), "B": (30, 6), "C": (30, 9)}, {"A": (30, 6), "B": (30, 6), "C": (30, 9)}, min_size=100)
    seg = e.tables["segments"]
    assert list(seg["segment"]) == ["Smaller segments"] and int(seg["n1"].iloc[0]) == 90
    assert e.checks_passed and e.summary["top"] == []


def test_tiny_segment_swing_is_inside_the_noise_band():
    e = _run({"TINY": (10, 1), "BIG": (5000, 1000)}, {"TINY": (10, 4), "BIG": (5000, 1050)})
    seg = e.tables["segments"].set_index("segment")
    assert not bool(seg.loc["TINY", "outside_noise"])


def test_a_pure_mix_move_counts_as_notable():
    # B's rate never changes, but its share doubles: that is the change, and it
    # must be citable (a rate-only noise test missed it).
    e = _run({"A": (8000, 800), "B": (2000, 1000)}, {"A": (6000, 600), "B": (4000, 2000)})
    b = e.tables["segments"].set_index("segment").loc["B"]
    assert b["rate_change_pp"] == pytest.approx(0.0) and bool(b["share_outside_noise"]) and bool(b["outside_noise"])
    assert e.summary["top"][0]["segment"] == "B"


def test_the_other_group_is_never_cited():
    # "Other" (makes outside the top 25) is a leftover group, like "Smaller segments".
    def month(prefix, worse):
        rows = [{"Make": f"M{i}", "curr_bucket": "STD"} for i in range(25) for _ in range(40)]
        rows += [{"Make": f"X{i}", "curr_bucket": "NPA" if worse and i < 30 else "STD"} for i in range(60)]
        df = pd.DataFrame(rows)
        return df.assign(**{"Loan No": [f"{prefix}{i}" for i in range(len(df))], "Loan Status": "RUN"})
    pair = make_pair(month("C", True), month("P", False), "2026-09", "2026-08")
    e = mix_rate(EvidenceStore(), pair, "make", min_size=1)
    assert "Other" in set(e.tables["segments"]["segment"])
    assert all(t["segment"] != "Other" for t in e.summary["top"])


def test_unavailable_dimension_is_refused():
    pair = make_pair(_month({"A": (5, 1)}, "C"), _month({"A": (5, 1)}, "P"), "2026-09", "2026-08")
    with pytest.raises(ToolError, match="no FUEL_TYPE column"):
        mix_rate(EvidenceStore(), pair, "fuel")
