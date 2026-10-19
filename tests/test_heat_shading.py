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
    from ui.tabs.root_cause import _table
    df = pd.DataFrame({
        "Region": ["A", "B", "Grand Total"], "Running Loans": [100, 100, 200],
        "NPA": [0, 4, 4], "NPA %": [0.0, 4.0, 2.0],
        "Delinquent": [10, 30, 40], "Delinquent %": [10.0, 30.0, 20.0],
    })
    cols = [{"key": "Region", "align": "left"}, {"key": "Running Loans"}]
    cols += [{"key": f"{c} %", "label": c, "fmt": "pct_count", "count": c, "heat": True} for c in ("NPA", "Delinquent")]
    html = _table(df, cols, last_row_is_total=True)
    assert "30.0% <span" in html and "(30)</span>" in html   # % first, count in brackets
    assert not any(g in html for g in GREENS)
    assert "#f87171" in html                                  # worst region is red
    total_row = html.split("<tr")[-1]
    assert "background:#f" not in total_row.split(">", 1)[1]  # no shaded cell in Grand Total


def test_unit_table_and_vintage_table_have_no_green_on_risk(monkeypatch):
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
    from ui.components import html_table          # the Business tab's vintage table spec
    html += html_table(pd.DataFrame({"Disbursement Month": ["X", "Y"], "Accounts": [10, 20], "NPA%": [0.5, 3.0],
                                     "NPA Count": [1, 6], "SMA-2%": [1.0, 2.0], "SMA-2 Count": [1, 4]}),
                       [{"key": "Disbursement Month"},
                        {"key": "SMA-2%", "fmt": "pct_count", "count": "SMA-2 Count", "heat": True},
                        {"key": "NPA%", "fmt": "pct_count", "count": "NPA Count", "heat": True}])
    assert "#f87171" in html and not any(g in html for g in GREENS)


