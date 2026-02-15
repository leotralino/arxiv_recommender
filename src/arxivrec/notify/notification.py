import os
import smtplib
from abc import abstractmethod
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText


class BaseNotifier:
    """Email, slack message, desktop alert, etc"""

    @abstractmethod
    def notify(self, **kwargs):
        pass


class GMailNotifier(BaseNotifier):
    def notify(
        self,
        subject: str = "Example subject",
        body_html: str = "Example email body",
        **kwargs,
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

        if not all([sender, password, recipient]):
            missing = [
                k
                for k, v in {
                    "EMAIL_USERNAME": sender,
                    "EMAIL_PASSWORD": password,
                    "NOTIFY_RECIPIENT": recipient,
                }.items()
                if not v
            ]
            raise ValueError(
                f"Missing required environment variables for email:"
                f"{', '.join(missing)}"
                f"Please set on your Github's forked repo:"
                f"Settings -> Secrets and variables -> Actions"
            )

        msg = MIMEMultipart()
        msg["Subject"] = subject
        msg["From"] = f"ArXiv Rec <{sender}>"
        msg["To"] = recipient
        msg.attach(MIMEText(body_html, "html"))

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender, password)
            server.sendmail(sender, recipient, msg.as_string())


class RSSNotifier(BaseNotifier):
    def notify(self):
        raise NotImplementedError("Please implement this feeder!")
