"""ALVAN best-effort transaction notifications via SMS, WhatsApp, and email.

Place this file beside upi_fraud_app.py in src/. Put credentials in src/.env.
No provider exception is allowed to change a transaction decision.
"""

import os
import smtplib
from email.mime.text import MIMEText

try:
    from dotenv import load_dotenv
    _ENV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    load_dotenv(_ENV_PATH, override=False)
except ImportError:
    # Environment variables configured in Windows remain usable without dotenv.
    pass

TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID", "").strip()
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN", "").strip()
TWILIO_SMS_NUMBER = os.getenv("TWILIO_SMS_NUMBER", "").strip()
TWILIO_WHATSAPP_NUMBER = os.getenv(
    "TWILIO_WHATSAPP_NUMBER", "whatsapp:+14155238886"
).strip()
GMAIL_ADDRESS = os.getenv("GMAIL_ADDRESS", "").strip()
GMAIL_APP_PASSWORD = os.getenv("GMAIL_APP_PASSWORD", "").strip()


def _build_message(txn: dict) -> str:
    """Build a short plain-text alert for the sender."""
    status = str(txn.get("fraud_flag", ""))
    amount = txn.get("amount", 0)
    recipient = str(txn.get("merchant_id", "unknown"))
    probability = txn.get("fraud_probability", 0)
    txn_status = txn.get("transaction_status", "UNKNOWN")
    txn_id = txn.get("transaction_id", "")
    timestamp = txn.get("timestamp", "")

    try:
        amount_text = f"Rs.{float(amount):,.2f}"
    except (TypeError, ValueError):
        amount_text = f"Rs.{amount}"
    try:
        risk_text = f"{float(probability):.2f}%"
    except (TypeError, ValueError):
        risk_text = str(probability)

    if "FRAUD" in status.upper():
        return (
            f"ALVAN FRAUD ALERT: {amount_text} to {recipient}. "
            f"Risk: {risk_text}. Decision: FRAUD. Status: {txn_status}. "
            f"ID: {txn_id}. Time: {timestamp}. If this wasn't you, open ALVAN "
            "and contact your payment provider. Do not share your OTP or PIN."
        )
    return (
        f"ALVAN SAFE NOTICE: {amount_text} to {recipient}. Risk: {risk_text}. "
        f"Decision: SAFE. Status: {txn_status}. ID: {txn_id}. Time: {timestamp}."
    )


def send_sms_alert(phone: str, message: str) -> bool:
    """Send SMS via Twilio; return False rather than raising on errors."""
    if not phone:
        print("ALVAN SMS skipped: no phone number is configured for this user.")
        return False
    if not (TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN and TWILIO_SMS_NUMBER):
        print("ALVAN SMS skipped: Twilio SMS credentials are not configured in src/.env.")
        return False
    try:
        from twilio.rest import Client
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        client.messages.create(body=message, from_=TWILIO_SMS_NUMBER, to=phone)
        return True
    except Exception as exc:
        print(f"ALVAN SMS send failed: {exc}")
        return False


def send_whatsapp_alert(phone: str, message: str) -> bool:
    """Send WhatsApp via the configured Twilio WhatsApp sender/sandbox."""
    if not phone:
        print("ALVAN WhatsApp skipped: no phone number is configured for this user.")
        return False
    if not (TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN and TWILIO_WHATSAPP_NUMBER):
        print("ALVAN WhatsApp skipped: Twilio credentials are not configured in src/.env.")
        return False
    try:
        from twilio.rest import Client
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        to_whatsapp = phone if phone.startswith("whatsapp:") else f"whatsapp:{phone}"
        from_whatsapp = (
            TWILIO_WHATSAPP_NUMBER
            if TWILIO_WHATSAPP_NUMBER.startswith("whatsapp:")
            else f"whatsapp:{TWILIO_WHATSAPP_NUMBER}"
        )
        client.messages.create(body=message, from_=from_whatsapp, to=to_whatsapp)
        return True
    except Exception as exc:
        print(f"ALVAN WhatsApp send failed: {exc}")
        return False


def _build_email_report(txn: dict) -> str:
    """Build a detailed plain-text ALVAN transaction report for email only.

    SMS/WhatsApp remain short notices. The email report is based only on
    transaction fields already recorded by the app and does not change the
    fraud model's decision or payment status.
    """
    def value(key, fallback="Not available"):
        v = txn.get(key, fallback)
        if v is None or str(v).strip() == "" or str(v).lower() == "nan":
            return fallback
        return v

    def money(key):
        try:
            return f"Rs. {float(txn.get(key, 0) or 0):,.2f}"
        except (TypeError, ValueError):
            return str(value(key, "Not available"))

    try:
        probability = float(txn.get("fraud_probability", 0) or 0)
        risk_text = f"{probability:.2f}%"
    except (TypeError, ValueError):
        risk_text = str(value("fraud_probability"))

    fraud_flag = str(value("fraud_flag", "UNKNOWN"))
    is_fraud = "FRAUD" in fraud_flag.upper()
    verdict = "FRAUD / BLOCKED OR FLAGGED" if is_fraud else "SAFE"

    lines = [
        "ALVAN — UPI TRANSACTION REPORT",
        "=" * 38,
        "",
        f"Decision: {verdict}",
        f"Risk level: {value('risk_level')}",
        f"Fraud probability: {risk_text}",
        "",
        "TRANSACTION DETAILS",
        "-" * 20,
        f"Transaction ID: {value('transaction_id')}",
        f"Date and time: {value('timestamp')}",
        f"Sender / account: {value('user_id')}",
        f"Recipient / merchant: {value('merchant_id')}",
        f"Amount: {money('amount')}",
        f"Transaction type: {value('txn_type')}",
        f"UPI app: {value('upi_app')}",
        f"Transaction status: {value('transaction_status')}",
        f"Payment status: {value('payment_status')}",
        f"Receiver credited: {value('receiver_credited')}",
        "",
        "RISK SIGNAL SUMMARY",
        "-" * 20,
        f"Merchant risk score: {value('merchant_risk_score')}",
        f"User average transaction: {money('user_avg_amount')}",
        f"User transaction count: {value('user_txn_count')}",
        f"Previous fraud count: {value('previous_fraud_count')}",
        f"New merchant indicator: {value('is_new_merchant')}",
        f"Device change indicator: {value('device_change')}",
        f"IP change indicator: {value('ip_change')}",
        f"Transaction velocity: {value('txn_velocity')}",
        "",
        "SYSTEM MESSAGE",
        str(value('api_message', "No additional message provided.")),
        "",
        "This report was generated by ALVAN for the transaction recorded in the app.",
        "If this transaction was not initiated by you, contact your payment provider",
        "through its official channel. Never share your UPI PIN or OTP.",
    ]
    return "\n".join(lines)


def send_email_alert(email: str, subject: str, message: str) -> bool:
    """Send email using Gmail SMTP SSL; return False on errors."""
    if not email:
        print("ALVAN email skipped: no email address is configured for this user.")
        return False
    if not (GMAIL_ADDRESS and GMAIL_APP_PASSWORD):
        print("ALVAN email skipped: Gmail credentials are not configured in src/.env.")
        return False
    try:
        msg = MIMEText(message, _charset="utf-8")
        msg["Subject"] = subject
        msg["From"] = GMAIL_ADDRESS
        msg["To"] = email
        # STARTTLS on port 587; this matches the Gmail SMTP method verified
        # in the user's project environment.
        with smtplib.SMTP("smtp.gmail.com", 587, timeout=20) as server:
            server.ehlo()
            server.starttls()
            server.ehlo()
            server.login(GMAIL_ADDRESS, GMAIL_APP_PASSWORD.replace(" ", "").strip())
            server.sendmail(GMAIL_ADDRESS, [email], msg.as_string())
        return True
    except Exception as exc:
        print(f"ALVAN email send failed: {exc}")
        return False


def notify_user(
    phone: str,
    email: str,
    txn: dict,
    *,
    send_sms: bool = True,
    send_whatsapp: bool = True,
    send_email: bool = True,
) -> dict:
    """Attempt enabled channels independently and return send-accepted status.

    The booleans mean the provider accepted the send request, not guaranteed
    delivery to the handset/mailbox. This function never propagates provider
    errors to the transaction workflow.
    """
    try:
        message = _build_message(txn)
        is_fraud = "FRAUD" in str(txn.get("fraud_flag", "")).upper()
        subject = "ALVAN Fraud Alert" if is_fraud else "ALVAN Transaction Notice"
    except Exception as exc:
        print(f"ALVAN notification message creation failed: {exc}")
        return {"sms": False, "whatsapp": False, "email": False}

    # Each branch is isolated so one channel cannot prevent the next channel.
    results = {"sms": False, "whatsapp": False, "email": False}
    if send_sms:
        try:
            results["sms"] = send_sms_alert(phone, message)
        except Exception as exc:
            print(f"ALVAN SMS unexpected error: {exc}")
    if send_whatsapp:
        try:
            results["whatsapp"] = send_whatsapp_alert(phone, message)
        except Exception as exc:
            print(f"ALVAN WhatsApp unexpected error: {exc}")
    if send_email:
        try:
            report_subject = (
                "ALVAN Fraud Transaction Report"
                if is_fraud
                else "ALVAN Safe Transaction Report"
            )
            report_body = _build_email_report(txn)
            results["email"] = send_email_alert(email, report_subject, report_body)
        except Exception as exc:
            print(f"ALVAN email unexpected error: {exc}")
    return results


if __name__ == "__main__":
    print(
        "Configure src/.env and call notify_user() from a real ALVAN transaction. "
        "Do not test with a real recipient unless you consent to the message."
    )
