import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from report_agent.state import ReportState


def email_dispatcher_node(state: ReportState) -> ReportState:
    """Send the HTML report via SMTP. Skips silently if SMTP_HOST is not configured."""
    smtp_host = os.environ.get("SMTP_HOST", "")
    if not smtp_host:
        return {**state, "email_sent": False, "email_error": "SMTP not configured"}

    smtp_port = int(os.environ.get("SMTP_PORT", "587"))
    smtp_user = os.environ.get("SMTP_USER", "")
    smtp_pass = os.environ.get("SMTP_PASS", "")
    to_addr   = os.environ.get("REPORT_EMAIL_TO", "")

    if not to_addr:
        return {**state, "email_sent": False, "email_error": "REPORT_EMAIL_TO not set"}
    if not state.get("html_report"):
        return {**state, "email_sent": False, "email_error": "No report to send"}

    try:
        msg = MIMEMultipart("mixed")
        msg["Subject"] = f"Shriram Finance -- Portfolio Intelligence Report {state['curr_month']}"
        msg["From"]    = smtp_user
        msg["To"]      = to_addr

        # HTML body
        msg.attach(MIMEText(state["html_report"], "html", "utf-8"))

        # HTML file attachment
        attach = MIMEBase("application", "octet-stream")
        attach.set_payload(state["html_report"].encode("utf-8"))
        encoders.encode_base64(attach)
        attach.add_header(
            "Content-Disposition",
            f'attachment; filename="portfolio_report_{state["curr_month"]}.html"'
        )
        msg.attach(attach)

        with smtplib.SMTP(smtp_host, smtp_port, timeout=30) as server:
            server.ehlo()
            server.starttls()
            server.login(smtp_user, smtp_pass)
            recipients = [r.strip() for r in to_addr.split(",") if r.strip()]
            server.sendmail(smtp_user, recipients, msg.as_string())

        return {**state, "email_sent": True, "email_error": ""}
    except Exception as e:
        return {**state, "email_sent": False, "email_error": str(e)}
