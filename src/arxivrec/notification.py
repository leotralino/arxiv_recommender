import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

EMAIL_TEMPLATE = """
<html>
<body>
    <h2>Today's ArXiv Recommendations 🐈</h2>
    <p>Based on your interest: {my_interest}, here are the top 5 papers:</p>
    {input_html_table}
</body>
</html>
"""


def send_email(
    subject="Daily ArXiv Recommendations",
    body_html="<h1>Your daily ArXiv paper recommendations are here!</h1>",
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
