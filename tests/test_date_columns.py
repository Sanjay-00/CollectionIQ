"""Every date column becomes a real date on upload, whatever the file stored:
Excel serials (46236), text (15/08/2026) or dates."""
from io import BytesIO

import pandas as pd
import pytest

from utils import is_date_column_name, parse_date_columns


@pytest.mark.parametrize("name", ["Ag_Date", "Last Receipt Date", "Last Received Date", "NPA_DATE",
                                  "NPA Date", "NpaDate", "AG DATE", "NPA DT", "npa_dt"])
def test_date_columns_are_found_by_name(name):
    assert is_date_column_name(name)


@pytest.mark.parametrize("name", ["Due Dt", "UPDATED_BY", "Update Flag", "Mandate No", "Candidate",
                                  "Validated", "Last Receipt Amount", "DelinquencyDays"])
def test_other_columns_are_not(name):
    assert not is_date_column_name(name)       # "Due Dt" is the EMI due day (5, 10, 15)


def test_serials_text_and_dates_all_become_dates():
    df = pd.DataFrame({
        "Last Received Date": [46236, "15/08/2026", pd.Timestamp("2026-08-15"), None],
        "NPA_DATE": [45500.0, "45600", "", 999_999_999],     # last one: not a plausible date
        "Due Dt": [5, 10, 15, 20],
    })
    out, warns = parse_date_columns(df)
    assert pd.api.types.is_datetime64_any_dtype(out["Last Received Date"])
    assert out["Last Received Date"].tolist()[:3] == [pd.Timestamp("2026-08-02"), pd.Timestamp("2026-08-15"),
                                                      pd.Timestamp("2026-08-15")]
    assert out["NPA_DATE"].iloc[0] == pd.Timestamp("2024-07-27") and pd.isna(out["NPA_DATE"].iloc[3])
    assert out["Due Dt"].tolist() == [5, 10, 15, 20]          # untouched
    assert any("NPA_DATE" in w for w in warns)                 # the unreadable value is reported


def test_a_mislabelled_column_is_left_alone_not_blanked():
    df = pd.DataFrame({"Remark Date": ["ok", "n/a", "pending"]})
    out, warns = parse_date_columns(df)
    assert out["Remark Date"].tolist() == ["ok", "n/a", "pending"]
    assert warns and "left as-is" in warns[0]


def test_due_date_list_dates_are_converted():
    # A .xlsb list stores dates as serial numbers.
    from analysis.root_cause import load_daily_missed_feed
    buf = BytesIO()
    pd.DataFrame({"LOAN NO": ["L1", "L2"], "AG DATE": [46236, 46000], "ARREARS / EMI": [0.5, 2.0]}).to_excel(buf, index=False)
    buf.seek(0)
    buf.name = "list.xlsx"
    df, err = load_daily_missed_feed(buf)
    assert err is None
    assert df["AG DATE"].tolist() == [pd.Timestamp("2026-08-02"), pd.Timestamp("2025-12-09")]
