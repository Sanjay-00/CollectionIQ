"""The ⓘ definitions shown beside the app's own analytical terms."""
import pandas as pd

from analysis.root_cause import _DRIVER_LABELS
from config import FLEET_MIN_LOANS, HARD_BUCKET_ARREARS_EMI_MIN
from ui.glossary import DRIVER_HELP, GLOSSARY, help_for, info_icon


def test_every_root_cause_driver_has_a_definition():
    # A renamed driver in analysis/root_cause.py must not silently lose its ⓘ.
    assert set(_DRIVER_LABELS.values()) == set(DRIVER_HELP)


def test_chronic_matches_the_lcc_flag():
    # "No Coll 3 Months and >6 EMI": 3 months without payment, over 6 EMIs overdue.
    text = GLOSSARY["Chronic"]
    assert "3 months" in text and f"more than {HARD_BUCKET_ARREARS_EMI_MIN} EMIs" in text


def test_thresholds_come_from_config():
    assert f"{HARD_BUCKET_ARREARS_EMI_MIN} or more EMIs" in GLOSSARY["Hard Bucket"]
    assert f"{FLEET_MIN_LOANS} or more loans" in GLOSSARY["Fleet operator"]


def test_percent_columns_share_their_base_definition():
    assert help_for("Insurance-Only %") == GLOSSARY["Insurance-Only"]
    assert help_for("Hard + Not Paying %") == GLOSSARY["Hard + Not Paying"]
    assert help_for("SMA-2 %") is None


def test_info_icon_escapes_and_is_empty_without_text():
    assert info_icon(None) == ""
    html = info_icon('say "hi" <b>')
    assert 'title="say &quot;hi&quot; &lt;b&gt;"' in html and "&#9432;" in html


def test_root_cause_table_headers_carry_definitions():
    from ui.tabs.root_cause import _html_table
    df = pd.DataFrame({"Unit": ["A"], "Insurance-Only %": [40.0], "SMA-2": [1]})
    html = _html_table(df, [{"key": "Unit"}, {"key": "Insurance-Only %"}, {"key": "SMA-2"}])
    assert html.count("&#9432;") == 1                      # only the app-specific term
    assert "cash or WCL adjustment" in html


def test_arrears_split_uses_the_four_plain_group_names():
    from analysis.root_cause import ARREARS_GROUPS, HARD_NOT_PAYING, compute_chronic_shock_split
    df = pd.DataFrame({
        "Unit": ["A"] * 4, "Arrears / EMI": [8.0, 7.0, 2.0, 0.5],
        "No Coll 3 Months and >6 EMI": ["Y", "N", "Y", "N"],
    })
    out = compute_chronic_shock_split(df).iloc[0]
    assert [out[g] for g in ARREARS_GROUPS] == [1, 1, 1, 1]      # one loan per group
    assert HARD_NOT_PAYING == "Hard + Not Paying"
    assert all(help_for(g) for g in ARREARS_GROUPS)                # every group has an ⓘ


def test_no_em_or_en_dashes_in_screen_code():
    # Product rule: on-screen text uses ":" or "," instead of em/en dashes.
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    files = [root / "app.py", *sorted((root / "ui").rglob("*.py"))]
    offenders = [f"{f.relative_to(root)}:{i}" for f in files
                 for i, line in enumerate(f.read_text(encoding="utf-8-sig").splitlines(), 1)
                 if "\u2014" in line or "\u2013" in line]
    assert offenders == []
