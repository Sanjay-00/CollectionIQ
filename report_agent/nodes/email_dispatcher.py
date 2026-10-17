import os
import re
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders


_ADDRESS = re.compile(r"^[^@\s,;<>]+@[^@\s,;<>]+\.[^@\s,;<>]+$")


def send_report_email(html_report: str, to_addr: str, curr_month: str,
                      title: str = "Portfolio Report", file_name: str | None = None) -> tuple[bool, str]:
    """Send the HTML report via SMTP as both body and attachment. Returns
    (success, error_message). title / file_name: the report's own ("Regional
    Report: PUNE", "regional_PUNE_2026-08.html"), so forwarded reports for
    different regions don't all look the same. Every address is checked, and
    a line break anywhere is refused (it could add a header to the email)."""
    smtp_host = os.environ.get("SMTP_HOST", "")
    if not smtp_host:
        return False, "SMTP not configured"

    to_addr = (to_addr or "").strip()
    if not to_addr:
        return False, "No recipient email provided"
    recipients = [r.strip() for r in to_addr.split(",") if r.strip()]
    bad = [r for r in recipients if not _ADDRESS.match(r)]
    if bad or "\n" in to_addr or "\r" in to_addr:
        return False, f"Not a valid email address: {', '.join(bad) or to_addr!r}"
    if not html_report:
        return False, "No report to send"

    smtp_port = int(os.environ.get("SMTP_PORT", "587"))
    smtp_user = os.environ.get("SMTP_USER", "")
    smtp_pass = os.environ.get("SMTP_PASS", "")

    try:
        msg = MIMEMultipart("mixed")
        msg["Subject"] = f"CollectionIQ: {title} ({curr_month})"
        msg["From"]    = smtp_user
        msg["To"]      = ", ".join(recipients)

        # HTML body
        msg.attach(MIMEText(html_report, "html", "utf-8"))

        # HTML file attachment
        attach = MIMEBase("application", "octet-stream")
        attach.set_payload(html_report.encode("utf-8"))
        encoders.encode_base64(attach)
        attach.add_header(
            "Content-Disposition",
            f'attachment; filename="{file_name or f"portfolio_report_{curr_month}.html"}"'
        )
        msg.attach(attach)

        with smtplib.SMTP(smtp_host, smtp_port, timeout=30) as server:
            server.ehlo()
            server.starttls()
            server.login(smtp_user, smtp_pass)
            server.sendmail(smtp_user, recipients, msg.as_string())

        return True, ""
    except Exception as e:
        return False, str(e)
