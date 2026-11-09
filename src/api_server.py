"""
api_server.py

UPI Fraud Detection API
Receives a transaction, runs the ML fraud model,
makes a transaction decision, and stores the result.

Self-healing note:
If data/live_transactions.csv ever has rows with a different number of fields
than its header (for example rows written by an older script or by Streamlit),
the API now backs the file up, salvages every readable row into a clean file,
and carries on instead of failing with a ParserError.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import os
import numpy as np
import json
import hashlib
import csv
import re
import uuid
import shutil
import threading
from urllib.parse import urlparse, parse_qs
from datetime import datetime

from predict import predict_transaction
from alvan_reasons import (
    MARK,
    build_reasons,
    build_summary,
    enrich_payload_with_history,
    explain_latest_transaction,
    is_why_question,
)


app = Flask(__name__)
CORS(app)


# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "..",
    "models",
    "lightgbm_model.pkl",
)

LIVE_DATA_PATH = os.path.join(
    BASE_DIR,
    "..",
    "data",
    "live_transactions.csv",
)

DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "data"))
QR_REGISTRY_CANDIDATES = [
    os.path.join(DATA_DIR, "qr_merchants.csv"),
    os.path.join(DATA_DIR, "qr_merchants.csv.csv"),
]
RECOVERY_FILE = os.path.join(DATA_DIR, "recovery_requests.csv")
RECOVERY_COLUMNS = [
    "request_id", "user_id", "transaction_id", "amount",
    "recipient_upi", "reason", "status", "timestamp"
]
ANDROID_ALERTS_FILE = os.path.join(DATA_DIR, "android_alert_profiles.csv")
ANDROID_ALERT_COLUMNS = ["user_id", "email", "notify_email", "updated_at"]
USERS_FILE = os.path.join(DATA_DIR, "users.csv")
USER_COLUMNS = ["username", "password_hash", "role", "balance", "created", "phone", "email", "notify_sms", "notify_whatsapp", "notify_email"]

# Column order of the records this API writes to live_transactions.csv.
API_RECORD_COLUMNS = [
    "timestamp", "transaction_id", "user_id", "merchant_id", "txn_type",
    "upi_app", "amount", "fraud_probability", "is_fraud",
    "decision_threshold", "risk_level", "transaction_status",
    "payment_status", "receiver_credited",
]

# Guards reads/repairs/appends of live_transactions.csv across request threads.
_LIVE_FILE_LOCK = threading.RLock()

# Use the existing notification service when it is available. It remains optional
# so an email-provider issue never changes the model decision or breaks a request.
try:
    from notifications import notify_user as _notify_user
except Exception as _notification_import_error:
    _notify_user = None
    print(f"ALVAN API email notifications unavailable: {_notification_import_error}")


# ---------------------------------------------------------
# LOAD MODEL (health check only)
# ---------------------------------------------------------

model = None
encoders = {}

if os.path.exists(MODEL_PATH):
    try:
        data = joblib.load(MODEL_PATH)
        if isinstance(data, dict):
            model = data.get("model")
            encoders = data.get("encoders", {})
        else:
            model = data
        print("Model loaded successfully.")
    except Exception as e:
        print("Model loading error:", e)
else:
    print("Model file not found:", MODEL_PATH)


def normalize_probability(value) -> float:
    """Normalize either 0..1 or 0..100 input to decimal 0..1."""
    try:
        p = float(value)
    except (TypeError, ValueError):
        raise ValueError("Invalid fraud probability returned by model.")

    if not np.isfinite(p):
        raise ValueError("Non-finite fraud probability returned by model.")

    if p > 1.0:
        p /= 100.0

    return float(np.clip(p, 0.0, 1.0))


# ---------------------------------------------------------
# SELF-HEALING LIVE TRANSACTION FILE HELPERS
# ---------------------------------------------------------

def _repair_live_csv(path):
    """
    Salvage a live_transactions.csv whose rows do not match its header.

    - Makes a timestamped backup copy first (the original data is never lost).
    - Keeps rows that match the header width.
    - Maps rows that match the API's own record width onto API_RECORD_COLUMNS.
    - Skips rows that match neither (they remain in the backup copy).
    - Writes a clean file with a header that covers both layouts.
    """
    with _LIVE_FILE_LOCK:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = os.path.join(
            os.path.dirname(path), f"live_transactions_backup_{stamp}.csv"
        )
        shutil.copy2(path, backup_path)

        with open(path, "r", newline="", encoding="utf-8-sig") as handle:
            all_rows = [row for row in csv.reader(handle) if any(cell.strip() for cell in row)]

        if not all_rows:
            print(f"ALVAN API: {path} was empty; nothing to repair.")
            return

        header = [str(h).strip() for h in all_rows[0]]
        columns = list(header) + [c for c in API_RECORD_COLUMNS if c not in header]

        kept = []
        skipped = 0
        for row in all_rows[1:]:
            if len(row) == len(header):
                kept.append(dict(zip(header, row)))
            elif len(row) == len(API_RECORD_COLUMNS):
                kept.append(dict(zip(API_RECORD_COLUMNS, row)))
            else:
                skipped += 1

        temp_path = path + ".repair_tmp"
        with open(temp_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            for record in kept:
                writer.writerow({column: record.get(column, "") for column in columns})
        os.replace(temp_path, path)

        print(
            f"ALVAN API repaired live_transactions.csv: kept {len(kept)} rows, "
            f"skipped {skipped} unreadable rows. Backup: {backup_path}"
        )


def _read_live_transactions():
    """
    Read live_transactions.csv safely. If the file is malformed it is repaired
    once and read again. Returns an empty DataFrame if the file is missing/empty.
    """
    with _LIVE_FILE_LOCK:
        if not os.path.exists(LIVE_DATA_PATH) or os.path.getsize(LIVE_DATA_PATH) == 0:
            return pd.DataFrame()
        try:
            return pd.read_csv(LIVE_DATA_PATH).fillna("")
        except pd.errors.EmptyDataError:
            return pd.DataFrame()
        except pd.errors.ParserError:
            _repair_live_csv(LIVE_DATA_PATH)
            try:
                return pd.read_csv(LIVE_DATA_PATH).fillna("")
            except pd.errors.EmptyDataError:
                return pd.DataFrame()


# ---------------------------------------------------------
# HOME / HEALTH CHECK
# ---------------------------------------------------------

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "UPI Fraud API running",
        "model_loaded": model is not None,
        "mode": "transaction_simulation",
    })


# ---------------------------------------------------------
# TRANSACTION HISTORY (READ-ONLY)
# ---------------------------------------------------------

@app.route("/history", methods=["GET"])
def history():
    """Read recent API records and matching per-user Streamlit history, without writing anything."""
    limit = request.args.get("limit", default=50, type=int)
    limit = max(1, min(limit or 50, 100))
    requested_user = (request.args.get("user_id") or "").strip()
    frames = []

    if os.path.exists(LIVE_DATA_PATH):
        try:
            live = _read_live_transactions()
            if requested_user:
                if "user_id" in live.columns:
                    live = live[live["user_id"].astype(str) == requested_user]
                else:
                    live = live.iloc[0:0]
            frames.append(live)
        except (pd.errors.EmptyDataError, OSError, pd.errors.ParserError):
            pass

    if requested_user and re.fullmatch(r"[A-Za-z0-9_.-]{1,40}", requested_user):
        user_history_path = os.path.join(DATA_DIR, "user_history", f"{requested_user}.csv")
        if os.path.exists(user_history_path):
            try:
                personal = pd.read_csv(user_history_path).fillna("")
                if "user_id" not in personal.columns:
                    personal["user_id"] = requested_user
                personal = personal[personal["user_id"].astype(str) == requested_user]
                frames.append(personal)
            except (pd.errors.EmptyDataError, OSError, pd.errors.ParserError):
                pass

    frames = [frame for frame in frames if frame is not None and not frame.empty]
    if not frames:
        return jsonify({"success": True, "count": 0, "transactions": []})

    try:
        df = pd.concat(frames, ignore_index=True, sort=False).fillna("")
        if "transaction_id" in df.columns:
            df = df.drop_duplicates(subset=["transaction_id"], keep="last")
        if "timestamp" in df.columns:
            df["_alvan_sort_timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            df = df.sort_values("_alvan_sort_timestamp", ascending=True, na_position="first")
            df = df.drop(columns=["_alvan_sort_timestamp"])
        rows = json.loads(df.tail(limit).iloc[::-1].to_json(orient="records", date_format="iso"))
        return jsonify({"success": True, "count": len(rows), "transactions": rows})
    except Exception:
        app.logger.exception("Could not read transaction history")
        return jsonify({"success": False, "error": "Transaction history could not be loaded."}), 500


# ---------------------------------------------------------
# ANDROID APP SUPPORT: SMALL FILE HELPERS
# ---------------------------------------------------------

def _as_bool(value):
    return str(value or "").strip().lower() in {"true", "1", "yes", "on", "enabled"}


def _load_alert_profiles():
    if not os.path.exists(ANDROID_ALERTS_FILE):
        return pd.DataFrame(columns=ANDROID_ALERT_COLUMNS)
    try:
        df = pd.read_csv(ANDROID_ALERTS_FILE).fillna("")
    except (pd.errors.EmptyDataError, OSError, pd.errors.ParserError):
        df = pd.DataFrame(columns=ANDROID_ALERT_COLUMNS)
    for column in ANDROID_ALERT_COLUMNS:
        if column not in df.columns:
            df[column] = ""
    return df


def _get_alert_profile(user_id):
    profiles = _load_alert_profiles()
    if profiles.empty:
        return {"email": "", "notify_email": False}
    rows = profiles[profiles["user_id"].astype(str) == str(user_id)]
    if rows.empty:
        return {"email": "", "notify_email": False}
    row = rows.iloc[-1]
    return {"email": str(row.get("email", "") or "").strip(),
            "notify_email": _as_bool(row.get("notify_email"))}


def _save_alert_profile(user_id, email, notify_email):
    profiles = _load_alert_profiles()
    profiles = profiles[profiles["user_id"].astype(str) != str(user_id)]
    row = {
        "user_id": str(user_id),
        "email": str(email or "").strip(),
        "notify_email": "Yes" if notify_email else "No",
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    profiles = pd.concat([profiles, pd.DataFrame([row])], ignore_index=True)
    os.makedirs(DATA_DIR, exist_ok=True)
    profiles.reindex(columns=ANDROID_ALERT_COLUMNS).to_csv(ANDROID_ALERTS_FILE, index=False)


def _parse_qr_payload(raw):
    raw = str(raw or "").strip()
    if not raw:
        return {"valid": False, "raw": "", "vpa": "", "name": "", "amount": None, "qr_type": "empty"}
    if raw.lower().startswith("upi://"):
        parsed = urlparse(raw)
        query = parse_qs(parsed.query)
        vpa = (query.get("pa") or [""])[0].strip()
        name = (query.get("pn") or [""])[0].strip()
        raw_amount = (query.get("am") or [""])[0].strip()
        return {"valid": bool(vpa and "@" in vpa), "raw": raw, "vpa": vpa,
                "name": name, "amount": raw_amount or None, "qr_type": "upi"}
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}@[A-Za-z0-9][A-Za-z0-9.-]{0,62}", raw):
        return {"valid": True, "raw": raw, "vpa": raw, "name": "", "amount": None, "qr_type": "vpa"}
    return {"valid": False, "raw": raw, "vpa": "", "name": "", "amount": None, "qr_type": "unknown"}


def _resolve_qr_merchant(parsed):
    vpa = str(parsed.get("vpa") or "").strip().lower()
    name = str(parsed.get("name") or "").strip().lower()
    registry = None
    registry_path = next((path for path in QR_REGISTRY_CANDIDATES if os.path.exists(path)), None)
    if registry_path:
        try:
            registry = pd.read_csv(registry_path).fillna("")
            registry.columns = [str(c).strip().lower() for c in registry.columns]
        except Exception:
            registry = None
    if registry is not None and not registry.empty:
        for column in ("merchant_id", "merchant_name", "upi_vpa"):
            if column not in registry.columns:
                registry[column] = ""
        matches = registry[registry["upi_vpa"].astype(str).str.strip().str.lower() == vpa] if vpa else registry.iloc[0:0]
        if matches.empty and name:
            matches = registry[registry["merchant_name"].astype(str).str.strip().str.lower() == name]
        if not matches.empty:
            row = matches.iloc[0]
            score_raw = row.get("merchant_risk_score", 0.10)
            try:
                score = float(score_raw)
            except (TypeError, ValueError):
                score = 0.10
            return {
                "recognized": True,
                "merchant_id": str(row.get("merchant_id", "") or "").strip(),
                "merchant_name": str(row.get("merchant_name", "") or "").strip(),
                "upi_vpa": str(row.get("upi_vpa", "") or "").strip(),
                "merchant_risk_score": float(np.clip(score, 0.0, 1.0)),
                "historical_txn_count": _safe_int(row.get("historical_txn_count", 0)),
                "historical_fraud_count": _safe_int(row.get("historical_fraud_count", 0)),
                "registry_available": True,
            }
    return {
        "recognized": False,
        "merchant_id": vpa or parsed.get("raw", ""),
        "merchant_name": str(parsed.get("name") or ""),
        "upi_vpa": vpa,
        "merchant_risk_score": None,
        "historical_txn_count": 0,
        "historical_fraud_count": 0,
        "registry_available": bool(registry_path),
    }


def _safe_int(value):
    try:
        return max(0, int(float(value)))
    except (TypeError, ValueError):
        return 0


def _detect_language(text):
    sample = str(text or "")
    checks = [
        ("hi", r"[\u0900-\u097F]"), ("bn", r"[\u0980-\u09FF]"),
        ("kn", r"[\u0C80-\u0CFF]"), ("ta", r"[\u0B80-\u0BFF]"),
        ("te", r"[\u0C00-\u0C7F]"), ("ml", r"[\u0D00-\u0D7F]"),
        ("ur", r"[\u0600-\u06FF]"),
    ]
    for code, pattern in checks:
        if re.search(pattern, sample):
            return code
    try:
        from langdetect import detect
        detected = detect(sample).lower()
        if detected in {"en", "hi", "kn", "ta", "te", "ml", "mr", "bn", "ur", "es", "fr", "de"}:
            return detected
    except Exception:
        pass
    return "en"


def _translate_text(text, source, target):
    if not text or source == target:
        return text
    try:
        from deep_translator import GoogleTranslator
        output = GoogleTranslator(source=source or "auto", target=target).translate(str(text))
        return str(output).strip() if output else None
    except Exception as exc:
        app.logger.info("Optional chatbot translation unavailable: %s", exc)
        return None


def _assistant_answer_english(question, current_page, user_id):
    q = str(question or "").strip().lower()
    page = str(current_page or "Home").strip().lower()
    if not q:
        return "Please enter a valid question about ALVAN so I can help you."
    if is_why_question(q):
        try:
            return explain_latest_transaction(user_id, _read_live_transactions())
        except Exception:
            app.logger.exception("Could not explain latest transaction")
    page_guides = {
        "home": "Home shows ALVAN's demo overview, API connection status, quick actions, and a clearly labelled demo balance. It is not a real bank balance.",
        "pay": "Pay checks the recipient and amount with ALVAN's fraud model. If the payment is approved, your UPI app opens so you can complete it, and ALVAN shows the result that app reports.",
        "history": "History lists your fraud checks with the decision, the reasons, and the payment result reported by your UPI app. Use Refresh to reload the latest rows.",
        "assistant": "The Assistant answers app-related questions. Choose the language, type a question or use voice input, then send it.",
        "qr": "QR verification decodes a UPI QR or VPA and compares it with the local merchant registry. An unknown merchant is not automatically safe.",
        "recovery": "Recovery records a request against an eligible simulated transaction. A pending record is not a refund and does not reverse a bank transfer.",
        "alerts": "Alerts saves the email notification preference for the Android demo user. Delivery needs the existing notifications.py configuration to work.",
        "tools": "Tools contains QR verification, recovery requests, alert preferences, and API connection settings.",
    }
    if any(x in q for x in ("this page", "current page", "what can i do here", "how do i use this", "what is this screen")):
        return page_guides.get(page, "Ask about fraud predictions, transaction history, QR verification, recovery requests, alerts, or the demo balance.")
    if any(x in q for x in ("real bank", "bank balance", "real balance", "actual balance", "my balance", "how much money")):
        return "The ₹50,000 shown on Home is an ALVAN demo balance only. This prototype does not connect to a bank and cannot retrieve real bank funds. Check your bank or UPI app for the actual balance."
    if any(x in q for x in ("real payment", "actual payment", "send money", "phonepe", "google pay", "upi pin", "bank transfer", "payment completed")):
        return "ALVAN checks every payment with its fraud model first. If it approves, your UPI app (PhonePe, Google Pay, BHIM, etc.) opens and you complete the payment there with your own UPI PIN. ALVAN then shows the status your UPI app reports, which is not a bank statement. Never share your UPI PIN or OTP with ALVAN."
    if any(x in q for x in ("qr", "scan code", "vpa", "merchant verify")):
        return "Open Tools > QR verification and scan a QR or paste a UPI URI/VPA. ALVAN compares it with its local merchant registry. A decoded QR is not proof that a merchant is trustworthy."
    if any(x in q for x in ("recover", "refund", "recovery request")):
        return "Open Tools > Recovery, enter the transaction ID of an eligible simulated transaction, describe the reason, and submit. The request is saved as Pending; it does not return real money."
    if any(x in q for x in ("history", "previous transaction", "last transaction", "transaction status", "latest transaction")):
        if os.path.exists(LIVE_DATA_PATH):
            try:
                df = _read_live_transactions()
                if "user_id" in df.columns:
                    df = df[df["user_id"].astype(str) == str(user_id)]
                if not df.empty:
                    row = df.iloc[-1]
                    amount = row.get("amount", "unknown")
                    risk = row.get("risk_level", "unknown")
                    status = row.get("transaction_status", "unknown")
                    txid = row.get("transaction_id", "unknown")
                    return f"Your latest saved transaction is {txid}, amount ₹{amount}, risk {risk}, status {status}. Check History for all recorded results."
            except Exception:
                pass
        return "No saved transaction was found for this app user yet. Open Pay and run one fraud check, then refresh History."
    if any(x in q for x in ("shap", "lime", "explain", "feature importance", "why did")):
        return "SHAP and LIME explain which input features influenced a model prediction. The current Android API response shows the fraud probability, risk level, decision and plain-language reasons; full SHAP/LIME plots remain in the existing Streamlit report unless separately exposed by an API. A score is not proof of fraud."
    if any(x in q for x in ("fraud", "lightgbm", "machine learning", "model", "risk score", "probability", "threshold")):
        return "ALVAN sends transaction details to predict.py, which runs the trained fraud model and returns a probability, risk level, threshold-based decision, and a message. Low risk is not a guarantee; confirm transaction details before any real payment."
    if any(x in q for x in ("email", "gmail", "notification", "alert")):
        return "Open Tools > Alerts to save an email address and enable email reports. Delivery depends on the existing notifications.py module and its SMTP/provider configuration; a failed email must not change the fraud decision."
    if any(x in q for x in ("help", "what can you answer", "valid question", "features", "what does alvan do", "about alvan")):
        return "I can explain Home, fraud checks, why a payment was safe or blocked, transaction history, QR merchant verification, recovery requests, email alert settings, multilingual voice questions, and the difference between ALVAN demo data and real bank activity."
    return f"I can help with ALVAN's Home, fraud checks, History, QR verification, recovery requests, and Alerts. You're asking from the {current_page or 'app'} screen. Try asking why your last transaction was safe or blocked, how a fraud score is calculated, or what a QR verification result means."


# ---------------------------------------------------------
# MULTI-FEATURE ENDPOINTS USED BY THE NATIVE ANDROID APP
# ---------------------------------------------------------

def _read_auth_users():
    if not os.path.exists(USERS_FILE):
        return []
    try:
        with open(USERS_FILE, "r", newline="", encoding="utf-8-sig") as handle:
            return [dict(row) for row in csv.DictReader(handle) if row.get("username")]
    except Exception:
        app.logger.exception("Could not read users.csv")
        return []


def _write_auth_users(rows):
    os.makedirs(DATA_DIR, exist_ok=True)
    normalized_rows = []
    for original in rows:
        row = dict(original)
        if not row.get("password_hash") and row.get("password"):
            row["password_hash"] = _password_hash(row.get("password"))
        row.setdefault("balance", "50000.00")
        row.setdefault("role", "user")
        for field in ("phone", "email", "notify_sms", "notify_whatsapp", "notify_email"):
            if not row.get(field):
                row[field] = "No" if field.startswith("notify_") else ""
        normalized_rows.append(row)
    with open(USERS_FILE, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=USER_COLUMNS)
        writer.writeheader()
        for row in normalized_rows:
            writer.writerow({column: row.get(column, "") for column in USER_COLUMNS})


def _password_hash(value):
    return hashlib.sha256(str(value or "").encode("utf-8")).hexdigest()


@app.route("/auth/login", methods=["POST"])
def auth_login():
    payload = request.get_json(silent=True) or {}
    username = str(payload.get("username") or "").strip()
    password = str(payload.get("password") or "")
    if not username or not password:
        return jsonify({"success": False, "error": "Enter both username and password."}), 400
    rows = _read_auth_users()
    for row in rows:
        if str(row.get("username") or "").strip().lower() != username.lower():
            continue
        stored_hash = str(row.get("password_hash") or "")
        legacy_password = str(row.get("password") or "")
        if stored_hash == _password_hash(password) or (not stored_hash and legacy_password == password):
            if not stored_hash:
                row["password_hash"] = _password_hash(password)
                row.pop("password", None)
                _write_auth_users(rows)
            return jsonify({"success": True, "username": row.get("username", username), "role": row.get("role", "user"), "message": "Signed in locally."})
        return jsonify({"success": False, "error": "Username or password is incorrect."}), 401
    return jsonify({"success": False, "error": "Account not found. Choose Register to create a user account."}), 404


@app.route("/auth/register", methods=["POST"])
def auth_register():
    payload = request.get_json(silent=True) or {}
    username = str(payload.get("username") or "").strip()
    password = str(payload.get("password") or "")
    if not re.fullmatch(r"[A-Za-z0-9_.-]{3,40}", username):
        return jsonify({"success": False, "error": "Username must be 3–40 characters using letters, numbers, dot, underscore or hyphen."}), 400
    if len(password) < 8:
        return jsonify({"success": False, "error": "Password must contain at least 8 characters."}), 400
    rows = _read_auth_users()
    if any(str(row.get("username") or "").strip().lower() == username.lower() for row in rows):
        return jsonify({"success": False, "error": "That username already exists. Sign in instead."}), 409
    rows.append({
        "username": username, "password_hash": _password_hash(password), "role": "user",
        "balance": "50000.00", "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "phone": "", "email": "", "notify_sms": "No", "notify_whatsapp": "No", "notify_email": "No",
    })
    try:
        _write_auth_users(rows)
    except Exception:
        app.logger.exception("Could not register the account")
        return jsonify({"success": False, "error": "Account registration could not be saved."}), 500
    return jsonify({"success": True, "username": username, "role": "user", "message": "Account created. Balance is ALVAN demo data only."}), 201


@app.route("/assistant", methods=["POST"])
def assistant():
    payload = request.get_json(silent=True) or {}
    question = str(payload.get("question") or "").strip()
    user_id = str(payload.get("user_id") or "android_demo_user").strip()
    current_page = str(payload.get("current_page") or "Home")
    requested_language = str(payload.get("response_language") or payload.get("language") or "auto").lower()
    if not question:
        return jsonify({"success": False, "error": "Please enter a valid question about ALVAN."}), 400

    detected = _detect_language(question)
    language = detected if requested_language in ("", "auto") else requested_language
    supported = {"en", "hi", "kn", "ta", "te", "ml", "mr", "bn", "ur", "es", "fr", "de"}
    if language not in supported:
        language = "en"

    english_question = question
    if detected != "en":
        translated_question = _translate_text(question, "auto", "en")
        if translated_question:
            english_question = translated_question

    answer = _assistant_answer_english(english_question, current_page, user_id)
    translation_available = True
    if language != "en":
        translated_answer = _translate_text(answer, "en", language)
        if translated_answer:
            answer = translated_answer
        else:
            translation_available = False
            answer += "\n\n(Translation service unavailable; this answer is in English.)"

    return jsonify({
        "success": True,
        "answer": answer,
        "language": language,
        "translation_available": translation_available,
        "current_page": current_page,
    })


@app.route("/qr/verify", methods=["POST"])
def verify_qr():
    payload = request.get_json(silent=True) or {}
    parsed = _parse_qr_payload(payload.get("raw") or payload.get("qr_text") or payload.get("vpa") or "")
    if not parsed["valid"]:
        return jsonify({
            "success": False,
            "valid_upi": False,
            "recognized": False,
            "error": "This is not a supported UPI QR payload or VPA. Scan a UPI payment QR or paste its UPI URI/VPA.",
            "qr_type": parsed["qr_type"],
        }), 400

    match = _resolve_qr_merchant(parsed)
    known = bool(match.get("recognized"))
    return jsonify({
        "success": True,
        "valid_upi": True,
        "recognized": known,
        "status": "KNOWN_MERCHANT" if known else "UNKNOWN_MERCHANT",
        "qr_type": parsed["qr_type"],
        "merchant_id": match.get("merchant_id", ""),
        "merchant_name": match.get("merchant_name", "") or parsed.get("name", ""),
        "upi_vpa": match.get("upi_vpa", "") or parsed.get("vpa", ""),
        "amount": parsed.get("amount"),
        "merchant_risk_score": match.get("merchant_risk_score"),
        "historical_txn_count": match.get("historical_txn_count", 0),
        "historical_fraud_count": match.get("historical_fraud_count", 0),
        "registry_available": match.get("registry_available", False),
        "message": (
            "Merchant matched the local ALVAN registry. Verify the payee and amount in your UPI app before proceeding."
            if known else
            "The UPI payload is readable but the merchant was not found in the local registry. Treat it as unverified."
        ),
        "payment_initiated": False,
    })


@app.route("/recovery", methods=["GET", "POST"])
def recovery():
    if request.method == "GET":
        user_id = str(request.args.get("user_id") or "android_demo_user").strip()
        if not os.path.exists(RECOVERY_FILE):
            return jsonify({"success": True, "count": 0, "requests": []})
        try:
            try:
                df = pd.read_csv(RECOVERY_FILE).fillna("")
            except pd.errors.EmptyDataError:
                df = pd.DataFrame(columns=RECOVERY_COLUMNS)
            for col in RECOVERY_COLUMNS:
                if col not in df.columns:
                    df[col] = ""
            df = df[df["user_id"].astype(str) == user_id]
            rows = json.loads(df.tail(100).iloc[::-1].to_json(orient="records")) if not df.empty else []
            return jsonify({"success": True, "count": len(rows), "requests": rows})
        except Exception:
            app.logger.exception("Recovery request history could not be read")
            return jsonify({"success": False, "error": "Recovery request history could not be loaded."}), 500

    payload = request.get_json(silent=True) or {}
    user_id = str(payload.get("user_id") or "android_demo_user").strip()
    transaction_id = str(payload.get("transaction_id") or "").strip()
    reason = str(payload.get("reason") or "").strip()
    if not transaction_id or not reason:
        return jsonify({"success": False, "error": "Transaction ID and reason are required."}), 400
    if not os.path.exists(LIVE_DATA_PATH):
        return jsonify({"success": False, "error": "No saved simulated transactions were found."}), 404

    try:
        txns = _read_live_transactions()
        required = {"transaction_id", "user_id"}
        if not required.issubset(set(txns.columns)):
            return jsonify({"success": False, "error": "Transaction data is missing required columns."}), 500
        matches = txns[(txns["transaction_id"].astype(str) == transaction_id) & (txns["user_id"].astype(str) == user_id)]
        if matches.empty:
            return jsonify({"success": False, "error": "That transaction was not found for this ALVAN app user."}), 404
        tx = matches.iloc[-1]
        payment_status = str(tx.get("payment_status", "")).strip().upper()
        credited = _as_bool(tx.get("receiver_credited"))
        if payment_status != "SIMULATED_SUCCESS" or not credited:
            return jsonify({"success": False, "error": "Only completed simulated transactions can have a recovery request."}), 400

        if os.path.exists(RECOVERY_FILE):
            try:
                existing = pd.read_csv(RECOVERY_FILE).fillna("")
            except pd.errors.EmptyDataError:
                existing = pd.DataFrame(columns=RECOVERY_COLUMNS)
        else:
            existing = pd.DataFrame(columns=RECOVERY_COLUMNS)
        for col in RECOVERY_COLUMNS:
            if col not in existing.columns:
                existing[col] = ""
        duplicate = existing[(existing["user_id"].astype(str) == user_id) & (existing["transaction_id"].astype(str) == transaction_id)]
        if not duplicate.empty:
            return jsonify({
                "success": False,
                "error": "A recovery request already exists for this transaction.",
                "request_id": str(duplicate.iloc[-1].get("request_id", "")),
                "status": str(duplicate.iloc[-1].get("status", "Pending")),
            }), 409

        row = {
            "request_id": f"REC_{uuid.uuid4().hex[:8]}",
            "user_id": user_id,
            "transaction_id": transaction_id,
            "amount": tx.get("amount", ""),
            "recipient_upi": tx.get("merchant_id", ""),
            "reason": reason,
            "status": "Pending",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        columns = list(existing.columns)
        for col in RECOVERY_COLUMNS:
            if col not in columns:
                columns.append(col)
        existing = pd.concat([existing, pd.DataFrame([{col: row.get(col, "") for col in columns}])], ignore_index=True)
        existing.reindex(columns=columns).to_csv(RECOVERY_FILE, index=False)
        return jsonify({"success": True, "request_id": row["request_id"], "status": row["status"], "message": "Recovery request saved as Pending. This is not a bank refund."})
    except Exception:
        app.logger.exception("Recovery request submission failed")
        return jsonify({"success": False, "error": "Could not save the recovery request."}), 500


@app.route("/alerts", methods=["GET", "POST"])
def alerts():
    if request.method == "GET":
        user_id = str(request.args.get("user_id") or "android_demo_user").strip()
        profile = _get_alert_profile(user_id)
        return jsonify({"success": True, "user_id": user_id, "email": profile["email"], "notify_email": profile["notify_email"], "notification_module_available": _notify_user is not None})

    payload = request.get_json(silent=True) or {}
    user_id = str(payload.get("user_id") or "android_demo_user").strip()
    email = str(payload.get("email") or "").strip()
    notify_email = _as_bool(payload.get("notify_email"))
    if len(user_id) < 1:
        return jsonify({"success": False, "error": "User ID is required."}), 400
    if email and not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", email):
        return jsonify({"success": False, "error": "Enter a valid email address."}), 400
    if notify_email and not email:
        return jsonify({"success": False, "error": "Enter an email address before enabling email alerts."}), 400
    try:
        _save_alert_profile(user_id, email, notify_email)
        return jsonify({"success": True, "email": email, "notify_email": notify_email, "notification_module_available": _notify_user is not None, "message": "Alert preferences saved. Email delivery depends on the existing notification provider configuration."})
    except Exception:
        app.logger.exception("Could not save Android alert preferences")
        return jsonify({"success": False, "error": "Could not save alert preferences."}), 500


# ---------------------------------------------------------
# TRANSACTION
# ---------------------------------------------------------

def _append_csv_record_preserving_columns(path, record):
    """
    Append a row while preserving any pre-existing CSV columns and rows.
    If the existing file is malformed (rows with a different field count than
    the header), it is backed up and repaired automatically, then the append
    is retried.
    """
    with _LIVE_FILE_LOCK:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            pd.DataFrame([record]).to_csv(path, index=False)
            return
        try:
            existing = pd.read_csv(path).fillna("")
        except pd.errors.EmptyDataError:
            pd.DataFrame([record]).to_csv(path, index=False)
            return
        except pd.errors.ParserError:
            _repair_live_csv(path)
            try:
                existing = pd.read_csv(path).fillna("")
            except pd.errors.EmptyDataError:
                pd.DataFrame([record]).to_csv(path, index=False)
                return
        columns = list(existing.columns)
        for key in record.keys():
            if key not in columns:
                columns.append(key)
        new_row = {column: record.get(column, "") for column in columns}
        combined = pd.concat([existing.reindex(columns=columns), pd.DataFrame([new_row])], ignore_index=True, sort=False)
        combined.to_csv(path, index=False)


@app.route("/transaction", methods=["POST"])
def transaction():

    payload = request.get_json(silent=True)

    if not isinstance(payload, dict):
        return jsonify({
            "success": False,
            "error": "No valid JSON received",
        }), 400

    required_fields = ["user_id", "amount"]
    missing_fields = [
        field for field in required_fields
        if field not in payload or payload.get(field) in (None, "")
    ]

    if missing_fields:
        return jsonify({
            "success": False,
            "error": "Missing required fields",
            "missing_fields": missing_fields,
        }), 400

    # Preserve IDs/timestamps supplied by Streamlit whenever possible.
    if not payload.get("transaction_id"):
        payload["transaction_id"] = f"TXN_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"

    if not payload.get("timestamp"):
        payload["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")






        # ---------------------------------------------------------
    # QR-SCAN CONTEXT: resolve merchant on the SERVER
    # ---------------------------------------------------------
    qr_merchant = None
    is_qr_scan = bool(payload.get("qr_scan", False))

    if is_qr_scan:
        qr_raw = payload.get("qr_raw") or payload.get("merchant_id", "")
        parsed_qr = _parse_qr_payload(qr_raw)

        if not parsed_qr.get("valid"):
            return jsonify({
                "success": False,
                "error": "Invalid UPI QR data. Please scan again."
            }), 400

        qr_merchant = _resolve_qr_merchant(parsed_qr)

        # Never trust the merchant identity sent by Android.
        payload["merchant_id"] = parsed_qr["vpa"]
        payload["upi_vpa"] = parsed_qr["vpa"]
        payload["is_unknown_recipient"] = int(
            not qr_merchant.get("recognized", False)
        )
        payload["is_new_merchant"] = int(
            not qr_merchant.get("recognized", False)
        )

        # Use the registry's actual score for recognized merchants.
        # Unknown merchants retain the model's neutral default score.
        if qr_merchant.get("recognized"):
            payload["merchant_risk_score"] = qr_merchant.get(
                "merchant_risk_score", 0.10
            )

        # If the QR contains a fixed amount, do not allow Android
        # to silently substitute a different amount.
        qr_amount = parsed_qr.get("amount")
        if qr_amount not in (None, ""):
            try:
                fixed_amount = float(qr_amount)
                submitted_amount = float(payload.get("amount", 0))
                if (
                    not np.isfinite(fixed_amount)
                    or fixed_amount <= 0
                    or abs(fixed_amount - submitted_amount) > 0.01
                ):
                    return jsonify({
                        "success": False,
                        "error": "The amount does not match the amount encoded in this QR."
                    }), 400
                payload["amount"] = fixed_amount
            except (TypeError, ValueError):
                return jsonify({
                    "success": False,
                    "error": "Invalid amount encoded in the QR."
                }), 400

    # Optional: compute real behaviour signals from the user's saved history.
    # Turned on by setting the environment variable ALVAN_LIVE_CONTEXT=on.
    if os.environ.get("ALVAN_LIVE_CONTEXT", "off").strip().lower() == "on" and payload.get("txn_type") != "test_scenario":
        try:
            _merchant_for_context = None
            _parsed_ctx = _parse_qr_payload(payload.get("merchant_id", ""))
            if _parsed_ctx.get("valid"):
                _merchant_for_context = _resolve_qr_merchant(_parsed_ctx)
            payload = enrich_payload_with_history(payload, _read_live_transactions(), _merchant_for_context)
        except Exception:
            app.logger.exception("Live context enrichment skipped")

    try:
        result = predict_transaction(payload)
        prob = normalize_probability(
            result.get("fraud_probability", result.get("probability_fraud"))
        )
        is_fraud = int(result.get("is_fraud", int(prob >= 0.50)))
        decision_threshold = normalize_probability(
            result.get("decision_threshold", 0.50)
        )
    except Exception as e:
        print("Prediction error:", e)
        return jsonify({
            "success": False,
            "error": "Fraud prediction could not be completed",
            "transaction_status": "NOT_APPROVED",
            "payment_status": "NOT_EXECUTED",
            "receiver_credited": False,
        }), 503

    # The model's own threshold determines the block decision.
    is_fraud = int(is_fraud == 1 or prob >= decision_threshold)

    if is_fraud:
        risk_level = "HIGH"
        transaction_status = "BLOCKED"
        payment_status = "NOT_EXECUTED"
        receiver_credited = False
        message = "Transaction blocked because high fraud risk was detected."
    elif prob >= 0.30:
        risk_level = "MEDIUM"
        transaction_status = "APPROVED"    
        payment_status = (
       "READY_FOR_UPI" if is_qr_scan else "SIMULATED_SUCCESS"
        )
        receiver_credited = False if is_qr_scan else True
        
        message = "Transaction approved; risk score is being monitored."
    else:
        risk_level = "LOW"
        transaction_status = "APPROVED"
        payment_status = (
       "READY_FOR_UPI" if is_qr_scan else "SIMULATED_SUCCESS"
        )
        receiver_credited = False if is_qr_scan else True
        message = (
      "ALVAN fraud check approved. Review and authorize the payment "
      "inside your UPI app."
      if is_qr_scan else
      "Transaction approved by ALVAN's fraud check. "
      "Complete the payment inside your UPI app."
       )

    # ALVAN explanation: human-readable reasons for this decision
    try:
        _hist = _read_live_transactions()
    except Exception:
        _hist = pd.DataFrame()
    try:
        _parsed = _parse_qr_payload(payload.get("merchant_id", ""))
        _merchant = _resolve_qr_merchant(_parsed) if _parsed.get("valid") else None
    except Exception:
        _merchant = None
    reasons = build_reasons(payload, prob, decision_threshold, is_fraud, _hist, _merchant)
    explanation = build_summary(reasons, is_fraud, risk_level, prob, decision_threshold)

    os.makedirs(os.path.dirname(LIVE_DATA_PATH), exist_ok=True)

    transaction_record = {
        "timestamp": payload["timestamp"],
        "transaction_id": payload.get("transaction_id"),
        "user_id": payload.get("user_id", "user_anon"),
        "merchant_id": payload.get("merchant_id", "unknown"),
        "txn_type": payload.get("txn_type", "send"),
        "upi_app": payload.get("upi_app", payload.get("UPI_app", "unknown")),
        "amount": payload.get("amount", 0.0),
        "fraud_probability": prob,
        "is_fraud": is_fraud,
        "decision_threshold": decision_threshold,
        "risk_level": risk_level,
        "transaction_status": transaction_status,
        "payment_status": payment_status,
        "receiver_credited": receiver_credited,
        "reasons": " | ".join(f"{MARK[r['impact']]} {r['text']}" for r in reasons),
    }

    try:
        _append_csv_record_preserving_columns(LIVE_DATA_PATH, transaction_record)
    except Exception:
        app.logger.exception("Could not append simulated transaction record")
        return jsonify({
            "success": False,
            "error": "The prediction completed, but the transaction record could not be saved.",
            "transaction_status": "NOT_RECORDED",
            "payment_status": "NOT_EXECUTED",
            "receiver_credited": False,
        }), 500

    # Optional Android email alert. The notification module remains untouched and
    # any email failure must never alter the transaction decision.
    notify_email_sent = False
    alert_profile = _get_alert_profile(payload.get("user_id", "android_demo_user"))
    if alert_profile.get("notify_email") and alert_profile.get("email") and _notify_user is not None:
        try:
            email_txn = dict(payload)
            email_txn.update(transaction_record)
            email_txn["fraud_probability"] = round(prob * 100.0, 2)
            email_txn["fraud_flag"] = "⚠️ FRAUD" if is_fraud else "✅ SAFE"
            email_txn["is_fraud"] = is_fraud
            email_txn["api_message"] = message
            notify_result = _notify_user(
                phone="",
                email=alert_profile["email"],
                txn=email_txn,
                send_sms=False,
                send_whatsapp=False,
                send_email=True,
            )
            notify_email_sent = bool((notify_result or {}).get("email"))
        except Exception as notification_error:
            app.logger.warning("Android email alert failed without changing prediction: %s", notification_error)

    return jsonify({
        "success": True,
        "transaction_id": payload.get("transaction_id"),
        "fraud_probability": prob,
        "probability_fraud": prob,
        "is_fraud": is_fraud,
        "decision_threshold": decision_threshold,
        "risk_level": risk_level,
        "transaction_status": transaction_status,
        "payment_status": payment_status,
        "receiver_credited": receiver_credited,
        "message": message,
        "simulation": True,
        "notify_email_sent": notify_email_sent,
        "reasons": reasons,
        "explanation": explanation,
    })


@app.route("/transaction/payment-status", methods=["POST"])
def transaction_payment_status():
    """Record the result a UPI app reported for an approved transaction."""
    payload = request.get_json(silent=True) or {}
    user_id = str(payload.get("user_id") or "").strip()
    transaction_id = str(payload.get("transaction_id") or "").strip()
    status = str(payload.get("upi_status") or "").strip().upper()
    allowed = {"SUCCESS", "FAILURE", "SUBMITTED", "CANCELLED", "UNKNOWN", "USER_CONFIRMED"}
    if not user_id or not transaction_id or status not in allowed:
        return jsonify({"success": False, "error": "user_id, transaction_id and a valid upi_status are required."}), 400
    reference = re.sub(r"[^A-Za-z0-9_.-]", "", str(payload.get("upi_txn_ref") or ""))[:60]

    with _LIVE_FILE_LOCK:
        df = _read_live_transactions()
        if df.empty or not {"transaction_id", "user_id"}.issubset(df.columns):
            return jsonify({"success": False, "error": "Transaction not found."}), 404
        mask = (df["transaction_id"].astype(str) == transaction_id) & (df["user_id"].astype(str) == user_id)
        if not mask.any():
            return jsonify({"success": False, "error": "Transaction not found for this user."}), 404
        if "transaction_status" in df.columns and not (
            df.loc[mask, "transaction_status"].astype(str).str.upper() == "APPROVED"
        ).all():
            return jsonify({"success": False, "error": "Only approved transactions can have a payment status."}), 400
        for column in ("upi_status", "upi_txn_ref", "upi_status_time"):
            if column not in df.columns:
                df[column] = ""
        df.loc[mask, "upi_status"] = status
        if reference:
            df.loc[mask, "upi_txn_ref"] = reference
        df.loc[mask, "upi_status_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        df.to_csv(LIVE_DATA_PATH, index=False)

    return jsonify({"success": True, "transaction_id": transaction_id, "upi_status": status})
    


# ---------------------------------------------------------
# RUN SERVER
# ---------------------------------------------------------

if __name__ == "__main__":
    # Stable local-development server. Disabling the debug reloader avoids an
    # unexpected restart while the Android emulator is sending API requests.
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False,
        use_reloader=False,
    )