"""Source-extract faults repaired when a file loads (utils.repair_upload)."""
import pandas as pd

from tests.test_utils import _build_upload
from utils import canonical_region, load_and_validate, repair_upload


class TestRegionSpellings:
    def test_variants_map_to_one_name(self):
        s = pd.Series(["CS NAGAR", "csn", "C.S. Nagar", "Chhatrapati Sambhaji Nagar", "AKOLA"])
        assert list(canonical_region(s)) == ["CHHATRAPATI SAMBHAJI NAGAR"] * 4 + ["AKOLA"]

    def test_unknown_and_missing_values_are_untouched(self):
        s = pd.Series(["Pune Region", None])
        out = canonical_region(s)
        assert out[0] == "Pune Region" and pd.isna(out[1])

    def test_repair_reports_the_merge(self):
        df, notes = repair_upload(pd.DataFrame({"RegionName": ["CS NAGAR", "CSN", "AKOLA"]}))
        assert set(df["RegionName"]) == {"CHHATRAPATI SAMBHAJI NAGAR", "AKOLA"}
        assert len(notes) == 1 and "CS NAGAR" in notes[0] and "CSN" in notes[0]


class TestSwappedColendingColumns:
    def _df(self):
        # Rows 2-3 have the two columns traded places (real Aug LCC: 1,948 rows).
        return pd.DataFrame({
            "CoLending_Loans": ["N", "Y", "ALIVE", "ALIVE", "DEAD"],
            "CUSTOMER_STATUS": ["ALIVE", "ALIVE", "Y", "N", "ALIVE"],
        })

    def test_swaps_back_only_when_both_sides_prove_it(self):
        df, notes = repair_upload(self._df())
        assert list(df["CoLending_Loans"]) == ["N", "Y", "Y", "N", "DEAD"]      # last row: no proof, left alone
        assert list(df["CUSTOMER_STATUS"]) == ["ALIVE", "ALIVE", "ALIVE", "ALIVE", "ALIVE"]
        assert "Fixed 2 row(s)" in notes[0] and "1 of them are co-lending" in notes[0]

    def test_clean_file_has_no_notes(self):
        df, notes = repair_upload(pd.DataFrame({"CoLending_Loans": ["Y", "N"], "CUSTOMER_STATUS": ["ALIVE", "DEAD"]}))
        assert notes == []


def test_load_and_validate_applies_repairs_and_records_them():
    upload = _build_upload({
        "RegionName": ["CS NAGAR", "AKOLA", "AKOLA"],
        "CoLending_Loans": ["ALIVE", "N", "N"],
        "CUSTOMER_STATUS": ["Y", "ALIVE", "ALIVE"],
    })
    df, errs = load_and_validate.__wrapped__(upload)
    assert errs == []
    assert df.loc[df["Loan No"] == "L0", "RegionName"].iloc[0] == "CHHATRAPATI SAMBHAJI NAGAR"
    assert df.loc[df["Loan No"] == "L0", "CoLending_Loans"].iloc[0] == "Y"
    assert len(df.attrs["data_fixes"]) == 2


def test_multi_file_upload_keeps_each_files_notes(monkeypatch):
    import ui.components as components
    a = pd.DataFrame({"Loan No": ["A1"], "RegionName": ["X"]})
    b = pd.DataFrame({"Loan No": ["B1"], "RegionName": ["Y"]})
    a.attrs["data_fixes"], b.attrs["data_fixes"] = ["fix in a"], ["fix in b"]
    files = {"a.xlsx": a, "b.xlsx": b}

    class F(str):
        @property
        def name(self):
            return str(self)

    monkeypatch.setattr(components, "load_and_validate", lambda f: (files[str(f)].copy(), []))
    combined, _ = components._load_and_concat([F("a.xlsx"), F("b.xlsx")])
    assert combined.attrs["data_fixes"] == ["a.xlsx: fix in a", "b.xlsx: fix in b"]
