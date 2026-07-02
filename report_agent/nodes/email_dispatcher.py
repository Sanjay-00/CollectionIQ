import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from report_agent.state import ReportState


def send_report_email(html_report: str, to_addr: str, curr_month: str) -> tuple[bool, str]:
    """Send the HTML report via SMTP as both body and attachment.

    Returns (success, error_message). Used by both the auto-send-on-generate
    LangGraph node below and the tab's standalone "resend" button, so the
    two entry points can't drift out of sync with each other.
    """
    smtp_host = os.environ.get("SMTP_HOST", "")
    if not smtp_host:
        return False, "SMTP not configured"

    to_addr = (to_addr or "").strip()
    if not to_addr:
        return False, "No recipient email provided"
    if not html_report:
        return False, "No report to send"

    smtp_port = int(os.environ.get("SMTP_PORT", "587"))
    smtp_user = os.environ.get("SMTP_USER", "")
    smtp_pass = os.environ.get("SMTP_PASS", "")

    try:
        msg = MIMEMultipart("mixed")
        msg["Subject"] = f"Shriram Finance - Portfolio Intelligence Report {curr_month}"
        msg["From"]    = smtp_user
        msg["To"]      = to_addr

        # HTML body
        msg.attach(MIMEText(html_report, "html", "utf-8"))

        # HTML file attachment
        attach = MIMEBase("application", "octet-stream")
        attach.set_payload(html_report.encode("utf-8"))
        encoders.encode_base64(attach)
        attach.add_header(
            "Content-Disposition",
            f'attachment; filename="portfolio_report_{curr_month}.html"'
        )
        msg.attach(attach)

        with smtplib.SMTP(smtp_host, smtp_port, timeout=30) as server:
            server.ehlo()
            server.starttls()
            server.login(smtp_user, smtp_pass)
            recipients = [r.strip() for r in to_addr.split(",") if r.strip()]
            server.sendmail(smtp_user, recipients, msg.as_string())

        return True, ""
    except Exception as e:
        return False, str(e)


def email_dispatcher_node(state: ReportState) -> ReportState:
    """Send the HTML report via SMTP. Skips silently if SMTP_HOST is not configured."""
    sent, error = send_report_email(
        state.get("html_report", ""), state.get("email_to", ""), state.get("curr_month", ""),
    )
    return {**state, "email_sent": sent, "email_error": error}
