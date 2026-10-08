"""
One report file per unit in view, in the one format picked, zipped when
there's more than one: a Zone report per zone, a Regional report per region,
a Branch report per branch. Leadership is always one file for everything in
view (so a regional manager filtered to their own region gets their region's
leadership view).

Last month's rows are matched to each unit by its name, ignoring case and
spaces (utils._unit_key), as the sidebar filter does.
"""
from __future__ import annotations

import io
import re
import zipfile

import pandas as pd

from utils import _unit_key

# level -> (column it splits by, the filter label it sets)
LEVEL_SPLIT = {"Zone": ("Zone", "Zone"), "Regional": ("RegionName", "Region"), "Branch": ("Unit", "Branch")}
FORMATS = {"PDF": "pdf", "HTML": "html", "Excel": "xlsx"}
MIME = {"pdf": "application/pdf", "html": "text/html",
        "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "zip": "application/zip"}


def _safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", str(text)).strip("_") or "unit"


def units_in(df: pd.DataFrame, level: str) -> list[str]:
    """The units this level makes a file for ([] for Leadership: one file)."""
    col = LEVEL_SPLIT.get(level, (None,))[0]
    if not col or col not in df.columns:
        return []
    return sorted(df[col].dropna().astype(str).unique())


def _render(model: dict, ext: str) -> bytes:
    from report_agent import render
    if ext == "pdf":
        return render.to_pdf(model)
    if ext == "xlsx":
        return render.to_excel(model)
    return render.to_html(model).encode("utf-8")


def build_files(df_curr: pd.DataFrame, df_prev: pd.DataFrame, curr_month: str, prev_month: str | None,
                level: str, filters: dict | None = None, sections: list[str] | None = None,
                fmt: str = "PDF", use_ai: bool = False, progress=None, only: list[str] | None = None) -> dict:
    """{"files": [(file name, bytes)], "preview": HTML of the first file,
    "notes": [AI messages]}. progress(done, total, unit) after each file.
    only: make files just for these units (None: every unit in view)."""
    from report_agent.story import build_report
    ext = FORMATS[fmt]
    filters = dict(filters or {})
    units = units_in(df_curr, level)
    if only is not None and units:
        units = [u for u in units if u in set(only)]
        if not units:                              # none picked: no files (never the whole book by mistake)
            return {"files": [], "preview": "", "notes": []}
    if units:
        col, label = LEVEL_SPLIT[level]
        curr_key = df_curr[col].map(_unit_key)
        prev_key = df_prev[col].map(_unit_key) if len(df_prev) and col in df_prev.columns else None
        jobs = [(u, df_curr[curr_key == _unit_key(u)],
                 df_prev[prev_key == _unit_key(u)] if prev_key is not None else df_prev.iloc[0:0],
                 {**filters, label: u}) for u in units]
    else:
        jobs = [(None, df_curr, df_prev, filters)]

    out = {"files": [], "preview": "", "notes": []}
    for i, (unit, c, p, f) in enumerate(jobs, start=1):
        model = build_report(c, p, curr_month, prev_month, level, f, sections=sections, unit=unit)
        if use_ai:
            from report_agent.ai_summary import polish
            model["ai_summary"], note = polish(model)
            if note:
                out["notes"].append(f"{unit}: {note}" if unit else note)
        if i == 1:
            from report_agent import render
            out["preview"] = render.to_html(model)
        name = f"{level.lower()}_{_safe_name(unit)}_{curr_month}" if unit else f"{level.lower()}_report_{curr_month}"
        out["files"].append((f"{name}.{ext}", _render(model, ext)))
        if progress:
            progress(i, len(jobs), unit or level)
    return out


def zip_files(files: list[tuple[str, bytes]]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in files:
            zf.writestr(name, data)
    return buf.getvalue()
