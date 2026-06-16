"""Shared UI helpers used across multiple tab modules."""

import os
from io import BytesIO

import pandas as pd
import streamlit as st

from utils import load_and_validate


def _safe_df(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce mixed-type object columns to string for safe st.dataframe display."""
    df = df.copy()
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).replace({"nan": "", "None": ""})
    return df


def _excel_bytes(df: pd.DataFrame) -> bytes:
    buf = BytesIO()
    df.to_excel(buf, index=False, engine="openpyxl")
    return buf.getvalue()


def _dl_btn(df: pd.DataFrame, filename: str, key: str) -> None:
    """Right-aligned compact Excel download button."""
    _, col = st.columns([5, 1])
    with col:
        st.download_button(
            "⬇ Excel", data=_excel_bytes(df), file_name=filename,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key=key, width='stretch',
        )


def _send_feedback(run_id: str, score: float) -> None:
    """Post thumbs-up (1.0) or thumbs-down (0.0) feedback to LangSmith."""
    try:
        from langsmith import Client as _LSClient
        _LSClient().create_feedback(run_id=run_id, key="result_quality", score=score)
    except Exception:
        pass


def _send_report_email(html_report: str, email_to: str, curr_month: str) -> tuple[bool, str]:
    """Send the HTML report via SMTP. Returns (success, error_msg)."""
    import smtplib
    from email import encoders as _enc
    from email.mime.base import MIMEBase
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText

    smtp_host = os.environ.get("SMTP_HOST", "")
    smtp_port = int(os.environ.get("SMTP_PORT", "587"))
    smtp_user = os.environ.get("SMTP_USER", "")
    smtp_pass = os.environ.get("SMTP_PASS", "")
    try:
        msg = MIMEMultipart("mixed")
        msg["Subject"] = f"Shriram Finance - Portfolio Intelligence Report {curr_month}"
        msg["From"]    = smtp_user
        msg["To"]      = email_to
        msg.attach(MIMEText(html_report, "html", "utf-8"))
        att = MIMEBase("application", "octet-stream")
        att.set_payload(html_report.encode("utf-8"))
        _enc.encode_base64(att)
        att.add_header("Content-Disposition", f'attachment; filename="portfolio_report_{curr_month}.html"')
        msg.attach(att)
        with smtplib.SMTP(smtp_host, smtp_port, timeout=30) as srv:
            srv.ehlo(); srv.starttls(); srv.login(smtp_user, smtp_pass)
            srv.sendmail(smtp_user, [r.strip() for r in email_to.split(",") if r.strip()], msg.as_string())
        return True, ""
    except Exception as e:
        return False, str(e)


def _load_and_concat(files) -> tuple[pd.DataFrame | None, list[str]]:
    """Load one or more Excel files and concatenate into a single DataFrame."""
    if not files:
        return None, ["No file uploaded"]
    if not isinstance(files, list):
        files = [files]

    dfs, errors = [], []
    for f in files:
        df, errs = load_and_validate(f)
        if errs:
            errors.append(f"{getattr(f, 'name', 'file')}: {errs[0]}")
        else:
            dfs.append(df)

    if not dfs:
        return None, errors

    if len(dfs) == 1:
        return dfs[0], errors

    def _normalize_dt(df):
        for col in df.select_dtypes(include="datetime64").columns:
            try:
                df[col] = df[col].astype("datetime64[ms]")
            except Exception:
                df[col] = pd.to_datetime(df[col], errors="coerce")
        return df

    dfs = [_normalize_dt(d) for d in dfs]
    combined = pd.concat(dfs, ignore_index=True)
    if "Loan No" in combined.columns:
        combined = combined.drop_duplicates(subset=["Loan No"], keep="first")
    return combined, errors
