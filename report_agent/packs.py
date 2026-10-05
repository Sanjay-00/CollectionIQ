"""
Branch packs: one Branch report per branch in the current view, as a ZIP with
a PDF (and an Excel annex with the call lists) per branch -- ready for a
regional manager to send to each branch manager.
"""
from __future__ import annotations

import io
import re
import zipfile

import pandas as pd

from utils import _unit_key


def _safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", str(text)).strip("_") or "branch"


def branches_in(df: pd.DataFrame) -> list[str]:
    return sorted(df["Unit"].dropna().astype(str).unique()) if "Unit" in df.columns else []


def build_branch_packs(df_curr: pd.DataFrame, df_prev: pd.DataFrame, curr_month: str, prev_month: str | None,
                       filters: dict | None = None, extras: list[str] | tuple = (), with_excel: bool = True,
                       progress=None) -> tuple[bytes, list[str]]:
    """(zip bytes, list of branches packed). progress(done, total, branch) is
    called after each branch. Last month's rows are matched to the branch by
    its name, ignoring case and spaces."""
    from report_agent import render
    from report_agent.story import build_report
    branches = branches_in(df_curr)
    curr_key = df_curr["Unit"].map(_unit_key) if "Unit" in df_curr.columns else None
    prev_key = df_prev["Unit"].map(_unit_key) if len(df_prev) and "Unit" in df_prev.columns else None
    buf, done = io.BytesIO(), []
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for i, branch in enumerate(branches, start=1):
            key = _unit_key(branch)
            c = df_curr[curr_key == key]
            p = df_prev[prev_key == key] if prev_key is not None else df_prev.iloc[0:0]
            model = build_report(c, p, curr_month, prev_month, "Branch",
                                 {**(filters or {}), "Branch": branch}, extras=extras)
            name = f"{_safe_name(branch)}_{curr_month}"
            zf.writestr(f"{name}.pdf", render.to_pdf(model))
            if with_excel:
                zf.writestr(f"{name}.xlsx", render.to_excel(model))
            done.append(branch)
            if progress:
                progress(i, len(branches), branch)
    return buf.getvalue(), done
