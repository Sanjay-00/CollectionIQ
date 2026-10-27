"""Header drift (real Sept 2026 Mumbai extract): "DelinquencyDay s" and
"CoLending_Loan s" carried a stray space, so the app didn't find the columns
and showed "Co-lending Loans at Risk: 0" for the whole month (real: 500)."""
import io

import pandas as pd

from smart_alerts import run_all_alerts
from utils import REQUIRED_COLS, _normalize_columns, header_renames_note, load_and_validate


def _upload(renames: dict, values: dict | None = None, n: int = 4) -> io.BytesIO:
    """A minimal valid LCC upload with some headers renamed as a drifted file has them."""
    data = {c: ["x"] * n for c in REQUIRED_COLS}
    data.update({
        "Loan No": [f"L{i}" for i in range(n)], "Arrears / EMI": [2.0, 0.0, 1.0, 0.0],
        "Month Receipt Amount": [100.0] * n, "Month Collection (Excluding Reserve Collection)": [100.0] * n,
        "Net Collection Demand Inst+Exp+BC": [100.0] * n, "POS": [1000.0] * n, "LCC%": [100.0] * n,
        "Closing Arrears": [0.0] * n, "Month Due-Inst": [100.0] * n, "Month Due-Exp": [0.0] * n,
        "Total Cum Collection": [1000.0] * n, "Strike": ["Y"] * n, "Due Dt": [5] * n,
        "CoLending_Loans": ["Y", "Y", "N", "N"], "CUSTOMER_STATUS": ["ALIVE"] * n, "DelinquencyDays": [45, 0, 20, 0],
    })
    data.update(values or {})
    buf = io.BytesIO()
    pd.DataFrame(data).rename(columns=renames).to_excel(buf, index=False, engine="openpyxl")
    buf.seek(0)
    buf.name = "drift.xlsx"
    return buf


SEPT_DRIFT = {"DelinquencyDays": "DelinquencyDay s", "CoLending_Loans": "CoLending_Loan s"}


class TestSeptemberHeaders:
    def test_spaced_headers_are_read_under_their_standard_names(self):
        df, errs = load_and_validate.__wrapped__(_upload(SEPT_DRIFT))
        assert errs == []
        assert {"DelinquencyDays", "CoLending_Loans"} <= set(df.columns)
        assert not {"DelinquencyDay s", "CoLending_Loan s"} & set(df.columns)
        assert not {"DelinquencyDays", "CoLending_Loans"} & set(df.attrs["missing_optional_cols"])

    def test_co_lending_alert_counts_the_loans_instead_of_a_false_zero(self):
        df, _ = load_and_validate.__wrapped__(_upload(SEPT_DRIFT))
        alerts = {a["title"]: a["count"] for a in run_all_alerts(df)}
        assert alerts["Co-lending Loans at Risk"] == 1          # L0: co-lending and delinquent

    def test_data_checks_say_which_headers_were_renamed(self):
        df, _ = load_and_validate.__wrapped__(_upload(SEPT_DRIFT))
        note = df.attrs["data_fixes"][0]
        assert "'DelinquencyDay s' read as DelinquencyDays" in note
        assert "'CoLending_Loan s' read as CoLending_Loans" in note


class TestRenameRules:
    def test_capital_letters_alone_are_not_reported(self):
        # Real extracts differ in case on ~25 headers every month: harmless, and noise.
        assert header_renames_note({"MNT Name": "MNT NAME"}) is None
        assert "read as" in header_renames_note({"MNT Name": "MNT NAME", "CoLending_Loan s": "CoLending_Loans"})

    def test_a_standard_header_is_never_overwritten_by_a_lookalike(self):
        df = _normalize_columns(pd.DataFrame(columns=["CoLending_Loans", "CoLending_Loan s"]))
        assert list(df.columns) == ["CoLending_Loans", "CoLending_Loan s"]     # no two columns with one name
        assert df.attrs["header_renames"] == {}

    def test_every_rename_is_recorded_for_data_health(self):
        df = _normalize_columns(pd.DataFrame(columns=["MNT Name", "DelinquencyDay s", "EMI Accrued"]))
        assert df.attrs["header_renames"] == {"MNT Name": "MNT NAME", "DelinquencyDay s": "DelinquencyDays",
                                              "EMI Accrued": "VehEMI Accrued"}

    def test_a_clean_file_has_no_rename_note(self):
        df, _ = load_and_validate.__wrapped__(_upload({}))
        assert not any("read as" in n for n in df.attrs["data_fixes"])
