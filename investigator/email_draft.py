"""
Turns a priority-style DataFrame into a downloadable .eml file.

This NEVER sends anything -- it only builds a downloadable file the user
opens and sends themselves from their own mail client (Outlook, Gmail,
etc), the exact same "downloadable, human-in-the-loop" pattern every other
result in this app already uses (ui/components.py::_dl_btn's Excel
downloads). There is no SMTP, no Outlook automation, no live delivery path
here at all.

Reuses the same stdlib-only approach (email.mime, nothing OS-specific) as
the sibling "Excel Automation" project's core/outlook.py::create_eml_file,
so this works identically whether CollectionIQ runs locally or in a cloud
deployment. That project's OTHER delivery path -- creating a live Outlook
draft via win32com.client.Dispatch("Outlook.Application") -- is
deliberately NOT ported here: it only works on a Windows machine with a
local Outlook install, which a cloud deployment (CLAUDE.md discusses
Streamlit Cloud health-check concerns for this exact app) would never have.
"""
from __future__ import annotations

import html as _html
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import pandas as pd

_CELL_STYLE = "border: 1px solid #ccc; padding: 6px 12px; font-family: Calibri, Arial, sans-serif; font-size: 10pt;"
_HEADER_STYLE = f"{_CELL_STYLE} background: #1a3660; color: white; font-weight: bold;"
_ODD_ROW_STYLE = f"{_CELL_STYLE} background: #ffffff;"
_EVEN_ROW_STYLE = f"{_CELL_STYLE} background: #f9fafb;"


def _esc(value) -> str:
    return _html.escape(str(value)) if value is not None and not pd.isna(value) else ""


def _table_html(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "<p>No priority cases found.</p>"
    parts = ['<table style="border-collapse: collapse; font-family: Calibri, Arial, sans-serif; font-size: 10pt;">']
    parts.append("<tr>" + "".join(f'<th style="{_HEADER_STYLE}">{_esc(c)}</th>' for c in df.columns) + "</tr>")
    for i, row in enumerate(df.itertuples(index=False)):
        style = _ODD_ROW_STYLE if i % 2 == 0 else _EVEN_ROW_STYLE
        parts.append("<tr>" + "".join(f'<td style="{style}">{_esc(v)}</td>' for v in row) + "</tr>")
    parts.append("</table>")
    return "\n".join(parts)


def build_priority_email_html(
    priority_df: pd.DataFrame, branch_name: str, window_label: str = "this week",
    category_label: str = "priority cases",
) -> str:
    """A plain, self-contained HTML email body. Every manually-typed LCC
    field (Cust Name, MNT NAME, branch_name itself) is escaped via _esc --
    same discipline report_agent/nodes/report_builder.py's _esc() convention
    already follows, since an unescaped "&"/"<"/">" in a free-typed field
    breaks the surrounding table markup (CLAUDE.md documents this exact
    class of issue for the report's own HTML rendering). category_label
    lets a per-category drill (e.g. "NPA Accounts", "High Closing Arrears")
    say what it actually is instead of the generic "priority cases" --
    defaults to the old wording so the one existing call site (the full
    priority_accounts dump) is unaffected."""
    safe_branch = _esc(branch_name)
    safe_category = _esc(category_label)
    return f"""<html>
<body style="font-family: Calibri, Arial, sans-serif; font-size: 11pt;">
<p>Dear Sir,</p>
<p>Please find below the {safe_category} {safe_branch} branch should focus on {_esc(window_label)}.</p>
{_table_html(priority_df)}
<p>Regards,<br>CollectionIQ Investigator</p>
</body>
</html>"""


def build_eml_bytes(subject: str, html_body: str, to_list: list[str], cc_list: list[str]) -> bytes:
    """A standalone .eml file's raw bytes -- opens directly as an unsent
    draft (X-Unsent: 1) in Outlook or any other mail client, reusing the
    exact mechanism Excel Automation's core/outlook.py::create_eml_file
    already uses in production, minus the local-disk write (this returns
    bytes directly so Streamlit can serve it via st.download_button with
    nothing touching disk)."""
    msg = MIMEMultipart()
    msg["Subject"] = subject
    msg["To"] = "; ".join(to_list)
    msg["CC"] = "; ".join(cc_list)
    msg["X-Unsent"] = "1"
    msg.attach(MIMEText(html_body, "html"))
    return msg.as_bytes()


def draft_priority_email(
    df: pd.DataFrame, branch_name: str, to_list: list[str], cc_list: list[str],
    category_label: str = "priority cases", window_label: str = "this week",
) -> tuple[str, bytes]:
    """One-call "tool" wrapping build_priority_email_html + build_eml_bytes:
    every caller (the full priority_accounts dump, and each individual
    priority_menu category drill) wants the exact same subject-line shape
    and HTML/eml assembly, so this is the single place that decides it,
    rather than duplicating "compose subject, build html, build eml" at
    each call site. Returns (subject, eml_bytes); callers still choose
    their own file_name/key for st.download_button."""
    subject = f"{category_label} for {branch_name} -- {window_label}"
    html_body = build_priority_email_html(
        df, branch_name, window_label=window_label, category_label=category_label.lower(),
    )
    return subject, build_eml_bytes(subject, html_body, to_list, cc_list)
