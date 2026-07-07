"""Regression: report_agent/nodes/report_builder.py's NPA & SMA-2 Movement
renderer used to crash the ENTIRE report generation on real data whenever a
region/branch/executive existed in one period's file but not the other (a
real case for two independently-uploaded monthly extracts -- e.g. a branch
opened or closed between months). The missing side comes back as NaN (from
compute_npa_sma2_comparison's merge), not None, and the renderer's own
`val is None` checks didn't catch that -- int(float('nan')) raises
ValueError: cannot convert float NaN to integer.
"""
import math

from report_agent.nodes.report_builder import (
    _movement_fmt, _movement_fmt_delta, _movement_delta_color, _render_npa_sma2_movement,
)


class TestMovementFmtHandlesNaN:
    def test_movement_fmt_nan_renders_placeholder_not_crash(self):
        assert _movement_fmt(float("nan")) == "&#8212;"

    def test_movement_fmt_none_still_renders_placeholder(self):
        assert _movement_fmt(None) == "&#8212;"

    def test_movement_fmt_real_value_still_formats(self):
        assert _movement_fmt(1234) == "1,234"

    def test_movement_fmt_delta_nan_renders_placeholder_not_crash(self):
        assert _movement_fmt_delta(float("nan")) == "&#8212;"

    def test_movement_fmt_delta_real_value_still_signed(self):
        assert _movement_fmt_delta(5) == "+5"
        assert _movement_fmt_delta(-3) == "-3"

    def test_movement_delta_color_nan_is_neutral_not_crash(self):
        assert _movement_delta_color(float("nan")) == "#6b7280"

    def test_movement_delta_color_none_is_neutral(self):
        assert _movement_delta_color(None) == "#6b7280"


class TestRenderNpaSma2MovementWithMissingSideData:
    def test_region_row_with_nan_prev_values_does_not_crash(self):
        # Simulates a region present in df_curr but absent from df_prev --
        # compute_npa_sma2_comparison's merge leaves the prev-side columns
        # as NaN for that row.
        data = {
            "portfolio": {
                "npa_current": 10, "npa_prev": None, "sma2_current": 5, "sma2_prev": None,
                "npa_delta": None, "sma2_delta": None, "npa_pct_change": None, "sma2_pct_change": None,
            },
            "region": [
                {
                    "RegionName": "NEWREGION", "Accounts": 50,
                    "NPA (Curr)": 3, "NPA (Prev)": float("nan"),
                    "SMA-2 (Curr)": 2, "SMA-2 (Prev)": float("nan"),
                    "NPA Δ": float("nan"), "SMA-2 Δ": float("nan"),
                    "NPA Δ%": float("nan"), "SMA-2 Δ%": float("nan"),
                },
            ],
            "branch_worst": [], "branch_best": [], "exec_worst": [], "exec_best": [],
            "has_prev": True,
        }
        html_out = _render_npa_sma2_movement(data)
        assert "NEWREGION" in html_out
        assert "&#8212;" in html_out  # the NaN cells render as a placeholder
