"""
recovery_service.py
ALVAN 24-hour recovery workflow (additive; does not touch existing routes).
"""

import csv
import os
import re
import threading
import uuid
from datetime import datetime, timedelta

from flask import Blueprint, jsonify, request

WINDOW_HOURS = 24
TS_FORMAT = "%Y-%m-%d %H:%M:%S"
OPEN_STATES = {"PENDING", "INFO_REQUESTED"}
VPA_REGEX = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}@[A-Za-z0-9][A-Za-z0-9.-]{0,62}")

REQ_COLUMNS = [
    "request_id", "transaction_id", "sender_id", "recipient_upi", "amount",
    "reason", "status", "created_at", "deadline_at", "responded_at",
    "sender_confirmed_at", "updated_at",
]
MSG_COLUMNS = ["request_id", "timestamp", "from_role", "text"]
DIR_COLUMNS = ["user_id", "upi_id", "updated_at"]

bp = Blueprint("recovery_v2", __name__)
_LOCK = threading.RLock()
_CFG = {"data_dir": "", "read_live": None}


# ----------------------------- helpers -----------------------------

def _path(name):
    return os.path.join(_CFG["data_dir"], name)


def _now():
    return datetime.now()


def _stamp(dt=None):
    return (dt or _now()).strftime(TS_FORMAT)


def _parse(value):
    try:
        return datetime.strptime(str(value), TS_FORMAT)
    except (TypeError, ValueError):
        return None


def _read(name):
    path = _path(name)
    if not os.path.exists(path):
        return []
    with open(path, "r", newline="", encoding="utf-8-sig") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _write(name, columns, rows):
    os.makedirs(_CFG["data_dir"], exist_ok=True)
    path = _path(name)
    temp = path + ".tmp"
    with open(temp, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, "") for c in columns})
    os.replace(temp, path)


def _norm(value):
    return str(value or "").strip().lower()


def _effective_status(row):
    status = row.get("status", "")
    deadline = _parse(row.get("deadline_at"))
    if status in OPEN_STATES and deadline and _now() > deadline:
        return "EXPIRED"
    return status


def _user_upi(user_id):
    for row in _read("upi_directory.csv"):
        if _norm(row.get("user_id")) == _norm(user_id):
            return _norm(row.get("upi_id"))
    return ""


def _owner_of_upi(upi_id):
    for row in _read("upi_directory.csv"):
        if _norm(row.get("upi_id")) == _norm(upi_id):
            return row.get("user_id", "")
    return ""


def _view(row, viewer_id):
    status = _effective_status(row)
    deadline = _parse(row.get("deadline_at"))
    remaining = 0
    if status in OPEN_STATES and deadline:
        remaining = max(0, int((deadline - _now()).total_seconds()))
    is_sender = _norm(row.get("sender_id")) == _norm(viewer_id)
    return {
        "request_id": row.get("request_id", ""),
        "transaction_id": row.get("transaction_id", ""),
        "sender_id": row.get("sender_id", ""),
        "recipient_upi": row.get("recipient_upi", ""),
        "recipient_on_alvan": bool(_owner_of_upi(row.get("recipient_upi"))),
        "amount": row.get("amount", ""),
        "reason": row.get("reason", ""),
        "status": status,
        "created_at": row.get("created_at", ""),
        "deadline_at": row.get("deadline_at", ""),
        "responded_at": row.get("responded_at", ""),
        "sender_confirmed_at": row.get("sender_confirmed_at", ""),
        "remaining_seconds": remaining,
        "role": "sender" if is_sender else "recipient",
    }


def _add_message(request_id, role, text):
    text = str(text or "").strip()[:500]
    if not text:
        return
    rows = _read("recovery_messages.csv")
    rows.append({"request_id": request_id, "timestamp": _stamp(), "from_role": role, "text": text})
    _write("recovery_messages.csv", MSG_COLUMNS, rows)


def _find(rows, request_id):
    for row in rows:
        if row.get("request_id") == request_id:
            return row
    return None


def _can_see(row, user_id):
    if _norm(row.get("sender_id")) == _norm(user_id):
        return True
    return _norm(row.get("recipient_upi")) == _user_upi(user_id) and bool(_user_upi(user_id))


def _is_recipient(row, user_id):
    mine = _user_upi(user_id)
    return bool(mine) and _norm(row.get("recipient_upi")) == mine


# ----------------------------- routes -----------------------------

@bp.route("/recovery/upi-id", methods=["GET", "POST"])
def upi_id():
    if request.method == "GET":
        user_id = str(request.args.get("user_id") or "").strip()
        return jsonify({"success": True, "user_id": user_id, "upi_id": _user_upi(user_id)})

    payload = request.get_json(silent=True) or {}
    user_id = str(payload.get("user_id") or "").strip()
    vpa = str(payload.get("upi_id") or "").strip().lower()
    if not user_id or not VPA_REGEX.fullmatch(vpa):
        return jsonify({"success": False, "error": "Enter a valid UPI ID such as name@bank."}), 400
    with _LOCK:
        rows = _read("upi_directory.csv")
        owner = next((r for r in rows if _norm(r.get("upi_id")) == vpa), None)
        if owner and _norm(owner.get("user_id")) != _norm(user_id):
            return jsonify({"success": False, "error": "That UPI ID is already linked to another ALVAN account."}), 409
        rows = [r for r in rows if _norm(r.get("user_id")) != _norm(user_id)]
        rows.append({"user_id": user_id, "upi_id": vpa, "updated_at": _stamp()})
        _write("upi_directory.csv", DIR_COLUMNS, rows)
    return jsonify({"success": True, "upi_id": vpa, "message": "UPI ID linked. Recovery requests sent to it will appear in your inbox."})


@bp.route("/recovery/create", methods=["POST"])
def create():
    payload = request.get_json(silent=True) or {}
    user_id = str(payload.get("user_id") or "").strip()
    txn_id = str(payload.get("transaction_id") or "").strip()
    reason = str(payload.get("reason") or "").strip()[:500]
    if not user_id or not txn_id or not reason:
        return jsonify({"success": False, "error": "Transaction ID and reason are required."}), 400

    live = _CFG["read_live"]()
    if live.empty or not {"transaction_id", "user_id"}.issubset(live.columns):
        return jsonify({"success": False, "error": "No saved transactions were found."}), 404
    match = live[(live["transaction_id"].astype(str) == txn_id) & (live["user_id"].astype(str) == user_id)]
    if match.empty:
        return jsonify({"success": False, "error": "That transaction was not found for your account."}), 404
    tx = match.iloc[-1]

    approved = str(tx.get("transaction_status", "")).upper() == "APPROVED"
    upi_status = str(tx.get("upi_status", "")).upper()
    simulated_ok = str(tx.get("payment_status", "")).upper() == "SIMULATED_SUCCESS"
    paid = upi_status in {"SUCCESS", "USER_CONFIRMED"} or simulated_ok
    if str(tx.get("txn_type", "")) == "test_scenario" or not approved or not paid:
        return jsonify({"success": False, "error": "Only completed payments (reported paid by your UPI app, or confirmed by you) can be recovered."}), 400

    recipient_upi = _norm(tx.get("merchant_id"))
    if not VPA_REGEX.fullmatch(recipient_upi):
        return jsonify({"success": False, "error": "This transaction has no valid recipient UPI ID."}), 400
    if recipient_upi == _user_upi(user_id):
        return jsonify({"success": False, "error": "You cannot send a recovery request to your own UPI ID."}), 400

    with _LOCK:
        rows = _read("recovery_v2_requests.csv")
        for row in rows:
            if row.get("transaction_id") == txn_id and _norm(row.get("sender_id")) == _norm(user_id):
                return jsonify({"success": False, "error": "A recovery request already exists for this transaction.",
                                "request": _view(row, user_id)}), 409
        created = _now()
        row = {
            "request_id": f"REC_{uuid.uuid4().hex[:8].upper()}",
            "transaction_id": txn_id,
            "sender_id": user_id,
            "recipient_upi": recipient_upi,
            "amount": str(tx.get("amount", "")),
            "reason": reason,
            "status": "PENDING",
            "created_at": _stamp(created),
            "deadline_at": _stamp(created + timedelta(hours=WINDOW_HOURS)),
            "responded_at": "",
            "sender_confirmed_at": "",
            "updated_at": _stamp(created),
        }
        rows.append(row)
        _write("recovery_v2_requests.csv", REQ_COLUMNS, rows)
    return jsonify({"success": True, "request": _view(row, user_id),
                    "message": "Recovery request sent. The recipient has 24 hours to respond."}), 201


@bp.route("/recovery/sent", methods=["GET"])
def sent():
    user_id = str(request.args.get("user_id") or "").strip()
    rows = [r for r in _read("recovery_v2_requests.csv") if _norm(r.get("sender_id")) == _norm(user_id)]
    rows.sort(key=lambda r: r.get("created_at", ""), reverse=True)
    return jsonify({"success": True, "requests": [_view(r, user_id) for r in rows[:100]]})


@bp.route("/recovery/inbox", methods=["GET"])
def inbox():
    user_id = str(request.args.get("user_id") or "").strip()
    mine = _user_upi(user_id)
    if not mine:
        return jsonify({"success": True, "linked": False, "requests": []})
    rows = [r for r in _read("recovery_v2_requests.csv") if _norm(r.get("recipient_upi")) == mine]
    rows.sort(key=lambda r: r.get("created_at", ""), reverse=True)
    return jsonify({"success": True, "linked": True, "requests": [_view(r, user_id) for r in rows[:100]]})


@bp.route("/recovery/detail", methods=["GET"])
def detail():
    user_id = str(request.args.get("user_id") or "").strip()
    request_id = str(request.args.get("request_id") or "").strip()
    row = _find(_read("recovery_v2_requests.csv"), request_id)
    if row is None or not _can_see(row, user_id):
        return jsonify({"success": False, "error": "Recovery request not found."}), 404
    messages = [m for m in _read("recovery_messages.csv") if m.get("request_id") == request_id]
    return jsonify({"success": True, "request": _view(row, user_id), "messages": messages})


@bp.route("/recovery/respond", methods=["POST"])
def respond():
    payload = request.get_json(silent=True) or {}
    user_id = str(payload.get("user_id") or "").strip()
    request_id = str(payload.get("request_id") or "").strip()
    action = str(payload.get("action") or "").strip().upper()
    note = str(payload.get("note") or "").strip()
    new_status = {"AGREE": "AGREED", "DECLINE": "DECLINED", "MORE_INFO": "INFO_REQUESTED"}.get(action)
    if not new_status:
        return jsonify({"success": False, "error": "Action must be AGREE, DECLINE or MORE_INFO."}), 400
    if action == "MORE_INFO" and not note:
        return jsonify({"success": False, "error": "Tell the sender what information you need."}), 400

    with _LOCK:
        rows = _read("recovery_v2_requests.csv")
        row = _find(rows, request_id)
        if row is None or not _is_recipient(row, user_id):
            return jsonify({"success": False, "error": "Recovery request not found for your account."}), 404
        status = _effective_status(row)
        if status not in OPEN_STATES:
            return jsonify({"success": False, "error": f"This request is already {status}."}), 409
        row["status"] = new_status
        row["responded_at"] = _stamp()
        row["updated_at"] = _stamp()
        _write("recovery_v2_requests.csv", REQ_COLUMNS, rows)
        _add_message(request_id, "recipient", note)
    msg = ("You agreed to return the payment. Please send it from your UPI app; "
           "ALVAN cannot verify the refund itself." if new_status == "AGREED" else "Response recorded.")
    return jsonify({"success": True, "request": _view(row, user_id), "message": msg})


@bp.route("/recovery/reply", methods=["POST"])
def reply():
    """Sender answers a 'more information' request. The 24h deadline does NOT extend."""
    payload = request.get_json(silent=True) or {}
    user_id = str(payload.get("user_id") or "").strip()
    request_id = str(payload.get("request_id") or "").strip()
    text = str(payload.get("text") or "").strip()
    if not text:
        return jsonify({"success": False, "error": "Enter a reply."}), 400
    with _LOCK:
        rows = _read("recovery_v2_requests.csv")
        row = _find(rows, request_id)
        if row is None or _norm(row.get("sender_id")) != _norm(user_id):
            return jsonify({"success": False, "error": "Recovery request not found for your account."}), 404
        if _effective_status(row) != "INFO_REQUESTED":
            return jsonify({"success": False, "error": "This request is not waiting for more information."}), 409
        row["status"] = "PENDING"
        row["updated_at"] = _stamp()
        _write("recovery_v2_requests.csv", REQ_COLUMNS, rows)
        _add_message(request_id, "sender", text)
    return jsonify({"success": True, "request": _view(row, user_id)})


@bp.route("/recovery/confirm-return", methods=["POST"])
def confirm_return():
    """Sender separately confirms whether the money actually came back."""
    payload = request.get_json(silent=True) or {}
    user_id = str(payload.get("user_id") or "").strip()
    request_id = str(payload.get("request_id") or "").strip()
    received = bool(payload.get("received"))
    with _LOCK:
        rows = _read("recovery_v2_requests.csv")
        row = _find(rows, request_id)
        if row is None or _norm(row.get("sender_id")) != _norm(user_id):
            return jsonify({"success": False, "error": "Recovery request not found for your account."}), 404
        if row.get("status") != "AGREED":
            return jsonify({"success": False, "error": "You can confirm a return only after the recipient agreed."}), 409
        row["status"] = "RETURN_CONFIRMED" if received else "RETURN_DISPUTED"
        row["sender_confirmed_at"] = _stamp()
        row["updated_at"] = _stamp()
        _write("recovery_v2_requests.csv", REQ_COLUMNS, rows)
    return jsonify({"success": True, "request": _view(row, user_id)})
@bp.errorhandler(Exception)
def _recovery_error(exc):
    from werkzeug.exceptions import HTTPException
    if isinstance(exc, HTTPException):
        return exc
    import traceback
    traceback.print_exc()
    return jsonify({"success": False, "error": f"Server error: {type(exc).__name__}: {exc}"}), 500

def register_recovery(app, data_dir, read_live_fn):
    _CFG["data_dir"] = data_dir
    _CFG["read_live"] = read_live_fn
    app.register_blueprint(bp)