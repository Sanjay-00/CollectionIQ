"""Risk columns are shaded pale pink -> red by rank in the column, never green."""
import pandas as pd

from ui.components import heat_bg, heat_range, heat_style

GREENS = ("#16a34a", "#dcfce7", "#86efac", "#f0fdf4", "63BE7B")


def test_lowest_is_palest_highest_is_reddest():
    rng = heat_range([1.0, 5.0, 9.0])
    assert heat_bg(1.0, rng) == "#fef2f2" and heat_bg(9.0, rng) == "#f87171"
    mid = heat_bg(5.0, rng)
    assert mid not in (heat_bg(1.0, rng), heat_bg(9.0, rng))


def test_zero_and_missing_get_no_shade():
    rng = heat_range([0, None, float("nan"), 3.0])
    assert rng == (3.0, 3.0)
    assert heat_style(0, rng) == "" and heat_style(None, rng) == "" and heat_style(float("nan"), rng) == ""
    assert heat_bg(3.0, rng)                      # a lone value still shows
    assert heat_range([0, 0]) is None


def test_root_cause_status_table_never_green_and_total_unshaded():
    from ui.tabs.root_cause import _html_table, _pct_and
    df = pd.DataFrame({
        "Region": ["A", "B", "Grand Total"], "Running Loans": [100, 100, 200],
        "NPA": [0, 4, 4], "NPA %": [0.0, 4.0, 2.0],
        "Delinquent": [10, 30, 40], "Delinquent %": [10.0, 30.0, 20.0],
    })
    cols = [{"key": "Region", "align": "left"}, {"key": "Running Loans"}]
    cols += [{"key": f"{c} %", "label": c, "fmt": _pct_and(c), "heat": True} for c in ("NPA", "Delinquent")]
    html = _html_table(df, cols, last_row_is_total=True)
    assert "30.0% <span" in html and "(30)</span>" in html   # % first, count in brackets
    assert not any(g in html for g in GREENS)
    assert "#f87171" in html                                  # worst region is red
    total_row = html.split("<tr")[-1]
    assert "background:#f" not in total_row.split(">", 1)[1]  # no shaded cell in Grand Total


def test_unit_table_and_product_table_have_no_green_on_risk(monkeypatch):
    import ui.tabs.portfolio_intelligence as pi
    captured = []
    monkeypatch.setattr(pi.st, "markdown", lambda html, **_: captured.append(html))
    region = pd.DataFrame({
        "Region": ["A", "B"], "Accounts": [100, 100], "SMA-2": [0, 5], "SMA-2%": [0.0, 5.0], "NPA": [1, 9],
        "NPA%": [1.0, 9.0], "NPA% (SOH)": [1.0, 12.0], "Delinquent": [3, 20], "Delinquency%": [3.0, 20.0],
        "Hard Bucket": [1, 4], "Hard Bucket%": [1.0, 4.0], "Roll Fwd%": [2.0, 30.0], "Slipped": [2, 30],
        "Collection%": [95.0, 80.0], "Strike%": [90.0, 70.0],
    })
    html = pi.units_table_html(region, "Region")
    pi._render_product_table(pd.DataFrame({"Segment": ["X", "Y"], "Accounts": [10, 20],
                                           "NPA%": [0.5, 3.0], "SMA-2%": [1.0, 2.0]}))
    html += "".join(captured)
    assert "#f87171" in html and not any(g in html for g in GREENS)


def test_executive_scorecard_risk_columns_not_green():
    from analysis.executive_scorecard import build_scorecard_table_html
    df = pd.DataFrame({
        "Executive (Branch)": ["E1", "E2"], "Accounts": [10, 10], "Demand (L)": [1.0, 1.0],
        "Collected (L)": [1.0, 0.9], "Collection %": [100.0, 90.0], "Strike Rate %": [80.0, 70.0],
        "NPA": [0, 2], "NPA %": [0.0, 20.0], "SMA-2": [0, 1], "SMA-2 %": [0.0, 10.0],
        "Delinquent": [0, 5], "Delinquency %": [0.0, 50.0], "Roll Fwd %": [0.0, 25.0], "Tier": ["top", "bottom"],
    })
    html = build_scorecard_table_html(df)
    # E2's four risk cells (counts sit inside their % cells now); lone value = mid shade
    assert html.count("background:#fbb2b2") == 4
    assert "20.0% <span" in html and "(2)</span>" in html     # NPA % with its count in one cell
    assert 'color:#16a34a;">0<' not in html                   # zeros are plain, not green


