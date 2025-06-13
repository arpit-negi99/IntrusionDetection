import smtplib
from email.message import EmailMessage
import time

EMAIL_RECEIVER = "kanaking1999@gmail.com"
EMAIL_PASSWORD = "gfesdwwqypsmkweq"
EMAIL_SENDER = "negiarpit2005@gmail.com"

last_alert_time = 0  # Global cooldown tracker (in seconds)
ALERT_COOLDOWN = 60  # Seconds between alerts

def send_alert_email():
    global last_alert_time
    current_time = time.time()

    if current_time - last_alert_time < ALERT_COOLDOWN:
        print("⏳ Alert suppressed — cooldown active.")
        return

    msg = EmailMessage()
    msg['Subject'] = '🚨 Intruder Detected in Classroom'
    msg['From'] = EMAIL_SENDER
    msg['To'] = EMAIL_RECEIVER
    msg.set_content('An outsider has been detected in the classroom. Please check immediately.')

    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(EMAIL_SENDER, EMAIL_PASSWORD)
            smtp.send_message(msg)

        last_alert_time = current_time
        print("📧 Alert email sent.")
    except Exception as e:
        print("❌ Failed to send email:", e)
