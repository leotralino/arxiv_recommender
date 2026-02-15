import os
import smtplib
from abc import ABC, abstractmethod
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText


class BaseNotifier(ABC):
    """Email, slack message, desktop alert, etc"""

    @abstractmethod
    def notify(self, **kwargs):
        pass


class EmailNotifier(BaseNotifier):
    def __init__(self, host: str = "smtp.gmail.com", port: int = 465):
        self.host = host
        self.port = port
        self.sender = os.getenv("EMAIL_USERNAME")
        self.password = os.getenv("EMAIL_PASSWORD")
        self.recipient = os.getenv("NOTIFY_RECIPIENT")

        if not all([self.sender, self.password, self.recipient]):
            missing = [
                k
                for k, v in {
                    "EMAIL_USERNAME": self.sender,
                    "EMAIL_PASSWORD": self.password,
                    "NOTIFY_RECIPIENT": self.recipient,
                }.items()
                if not v
            ]
            raise ValueError(
                f"Missing required environment variables for email:\n"
                f"{', '.join(missing)}\n"
                f"Please set on your Github's forked repo:\n"
                f"Settings -> Secrets and variables -> Actions"
            )

    def _send_email(
        self,
        subject: str,
        body_html: str,
    ):
        msg = MIMEMultipart()
        msg["Subject"] = subject
        msg["From"] = f"ArXiv Rec <{self.sender}>"
        msg["To"] = self.recipient
        msg.attach(MIMEText(body_html, "html"))

        with smtplib.SMTP_SSL(self.host, self.port) as server:
            server.login(self.sender, self.password)
            server.sendmail(self.sender, self.recipient, msg.as_string())

    def notify(
        self,
        subject: str = "Example subject",
        body_html: str = "Example email body",
        **kwargs,
    ):
        self._send_email(subject=subject, body_html=body_html)


class SlackNotifier(BaseNotifier):
    def notify(self):
        raise NotImplementedError("Please implement this feeder!")


class RSSNotifier(BaseNotifier):
    def notify(self):
        raise NotImplementedError("Please implement this feeder!")
