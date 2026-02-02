import os
import smtplib
from abc import abstractmethod
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText


class BaseNotifier:
    """Email, slack message, desktop alert, etc"""

    @abstractmethod
    def notify(self):
        pass


class EmailNotifier:
    def notify(
        self,
        subject: str = "Example subject",
        body_html: str = "Example email body",
    ):
        self._send_email(subject=subject, body_html=body_html)

    def _send_email(
        self,
        subject: str,
        body_html: str,
    ):
        sender = os.getenv("EMAIL_USERNAME")
        password = os.getenv("EMAIL_PASSWORD")
        recipient = os.getenv("NOTIFY_RECIPIENT")

        msg = MIMEMultipart()
        msg["Subject"] = subject
        msg["From"] = f"ArXiv Rec <{sender}>"
        msg["To"] = recipient
        msg.attach(MIMEText(body_html, "html"))

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender, password)
            server.sendmail(sender, recipient, msg.as_string())
