"""
alvan_reasons.py

Human-readable explanations for ALVAN fraud decisions.

Put this file in the same folder as api_server.py (the src folder).

The reasons combine the model's fraud score with context ALVAN already has:
the user's own saved history, the recipient's history and the merchant
registry. They describe WHY a decision looks safe or risky; they are not the
internal feature weights of the LightGBM model.
"""

import numpy as np
import pandas as pd

MARK = {"safe": "✔", "warn": "⚠", "risk": "✖"}

NO_TXN_TEXT = (
    "No saved transaction was found for this user yet. Open Pay, run one fraud "
    "check, and then ask me again why it was safe or blocked."
)


def _to_float(value):
    try:
        number = float(value)
        return number if np.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def _money(value):
    return f"₹{value:,.2f}"


def _lc(text):
    return text[:1].lower() + text[1:] if text else text


def _add(reasons, impact, text):
    reasons.append({"impact": impact, "text": text})


def build_reasons(payload, prob, threshold, is_fraud, history=None, merchant=None):
    """Return a list of {"impact": "safe|warn|risk", "text": "..."} items."""
    reasons = []
    pct = prob * 100.0
    thr = threshold * 100.0

    # 1) The model score itself (always present)
    if prob >= threshold:
        _add(reasons, "risk",
             f"The fraud model scored this transaction {pct:.2f}%, which reaches the {thr:.0f}% block threshold.")
    elif prob >= 0.30:
        _add(reasons, "warn",
             f"The fraud model scored this transaction {pct:.2f}%. That is below the {thr:.0f}% block threshold but inside the watch zone (30% and above).")
    else:
        _add(reasons, "safe",
             f"The fraud model scored this transaction only {pct:.2f}%, far below the {thr:.0f}% block threshold.")

    # 2) Context signals. A problem in any one of them must never break the API.
    try:
        user = str(payload.get("user_id", ""))
        vendor = str(payload.get("merchant_id", "")).strip().lower()
        amount = _to_float(payload.get("amount"))

        past = pd.DataFrame()
        if history is not None and not history.empty and "user_id" in history.columns:
            past = history[history["user_id"].astype(str) == user].copy()
            if "txn_type" in past.columns:
                past = past[past["txn_type"].astype(str) != "test_scenario"]

        # Amount compared with the user's normal spending
        if amount is not None:
            amounts = pd.Series(dtype=float)
            if "amount" in past.columns:
                amounts = pd.to_numeric(past["amount"], errors="coerce").dropna()
            avg = None
            if len(amounts) >= 3:
                avg = float(amounts.mean())
            elif "user_avg_amount" in payload:
                supplied = _to_float(payload.get("user_avg_amount"))
                if supplied and supplied > 0:
                    avg = supplied
            if avg is not None:
                if avg > 0 and amount > 3 * avg:
                    _add(reasons, "risk",
                         f"The amount {_money(amount)} is {amount / avg:.1f} times your usual average of {_money(avg)}.")
                elif avg > 0 and amount > 1.5 * avg:
                    _add(reasons, "warn",
                         f"The amount {_money(amount)} is higher than your usual average of {_money(avg)}.")
                else:
                    _add(reasons, "safe",
                         f"The amount {_money(amount)} is in line with your usual spending (average {_money(avg)}).")
            else:
                _add(reasons, "warn",
                     "There are too few earlier transactions to compare this amount with your normal spending.")
            if amount >= 50000:
                _add(reasons, "warn", f"{_money(amount)} is a large amount, so extra care is needed.")

        # Recipient history for this user
        if vendor:
            prior = past
            if "merchant_id" in past.columns:
                prior = past[past["merchant_id"].astype(str).str.strip().str.lower() == vendor]
            else:
                prior = past.iloc[0:0]
            if not prior.empty:
                flagged = 0
                if "is_fraud" in prior.columns:
                    flagged = int(pd.to_numeric(prior["is_fraud"], errors="coerce").fillna(0).sum())
                if flagged > 0:
                    _add(reasons, "risk",
                         f"{flagged} earlier transaction(s) to this recipient were flagged as fraud.")
                else:
                    _add(reasons, "safe",
                         f"You have paid this recipient {len(prior)} time(s) before with no fraud flags.")
            else:
                _add(reasons, "warn", "This is your first payment to this recipient.")

        # Merchant registry
        if merchant:
            if merchant.get("recognized"):
                fraud_n = int(merchant.get("historical_fraud_count") or 0)
                txn_n = int(merchant.get("historical_txn_count") or 0)
                if fraud_n > 0:
                    _add(reasons, "risk",
                         f"The recipient is in the ALVAN merchant registry but has {fraud_n} past fraud report(s).")
                else:
                    _add(reasons, "safe",
                         f"The recipient is a known merchant in the ALVAN registry ({txn_n} past transactions, no fraud reports).")
            elif merchant.get("registry_available"):
                _add(reasons, "warn",
                     "The recipient is not in the ALVAN merchant registry, so it is unverified.")

        # Behaviour signals supplied with the transaction (for example by the fraud test lab).
        # Real Android payments do not send these, so they only appear for scenarios.
        def _val(key):
            return (_to_float(payload.get(key)) or 0.0) if key in payload else 0.0

        if _val("device_change") == 1:
            _add(reasons, "risk", "The payment came from a new or different device.")
        if _val("ip_change") == 1:
            _add(reasons, "warn", "The network (IP address) changed since the user's last activity.")
        if _val("txn_velocity") > 5:
            _add(reasons, "risk", f"{int(_val('txn_velocity'))} transactions were made in a very short time.")
        if _val("geo_distance_km") > 200:
            _add(reasons, "warn", f"The payment originates about {_val('geo_distance_km'):,.0f} km from the usual location.")
        if _val("previous_fraud_count") > 0:
            _add(reasons, "risk", f"The account has {int(_val('previous_fraud_count'))} earlier fraud flag(s).")
        if _val("merchant_risk_score") > 0.7:
            _add(reasons, "risk", "The recipient has a high merchant risk score.")
        if "user_age_days" in payload and 0 < _val("user_age_days") < 14:
            _add(reasons, "warn", f"The account is only {int(_val('user_age_days'))} days old.")

        # Time of day and burst of payments
        ts = pd.to_datetime(payload.get("timestamp"), errors="coerce")
        if pd.notna(ts):
            if 0 <= ts.hour < 5:
                _add(reasons, "warn",
                     f"The payment is being made at an unusual hour ({ts.strftime('%I:%M %p')}).")
            if not past.empty and "timestamp" in past.columns and "txn_velocity" not in payload:
                times = pd.to_datetime(past["timestamp"], errors="coerce")
                recent = int(((ts - times).dt.total_seconds().between(0, 600)).sum())
                if recent >= 3:
                    _add(reasons, "warn",
                         f"{recent} other transactions were made in the last 10 minutes.")
    except Exception as exc:  # pragma: no cover - defensive
        print("ALVAN reasons: context signals skipped:", exc)

    order = ["risk", "warn", "safe"] if is_fraud else ["safe", "warn", "risk"]
    reasons.sort(key=lambda item: order.index(item["impact"]))
    return reasons


def enrich_payload_with_history(payload, history, merchant=None):
    """
    Return a copy of the payload with REAL behaviour signals computed from the
    user's saved history. Only fields the caller did not send are filled, and the
    fraud-lab test scenarios are never used as history.

    Signals: user_avg_amount (needs 3+ earlier payments), txn_velocity (payments in
    the last 10 minutes), is_new_merchant (first payment to this recipient) and
    merchant_risk_score (when the recipient is in the merchant registry).
    """
    enriched = dict(payload)
    if history is None or history.empty or "user_id" not in history.columns:
        return enriched

    user = str(payload.get("user_id", ""))
    past = history[history["user_id"].astype(str) == user].copy()
    if "txn_type" in past.columns:
        past = past[past["txn_type"].astype(str) != "test_scenario"]
    if past.empty:
        return enriched

    def _set(key, value):
        if key not in enriched or enriched[key] in (None, ""):
            enriched[key] = value

    if "amount" in past.columns:
        amounts = pd.to_numeric(past["amount"], errors="coerce").dropna()
        if len(amounts) >= 3:
            _set("user_avg_amount", float(amounts.mean()))

    ts = pd.to_datetime(payload.get("timestamp"), errors="coerce")
    if pd.notna(ts) and "timestamp" in past.columns:
        times = pd.to_datetime(past["timestamp"], errors="coerce")
        recent = int(((ts - times).dt.total_seconds().between(0, 600)).sum())
        _set("txn_velocity", recent + 1)

    vendor = str(payload.get("merchant_id", "")).strip().lower()
    if vendor and "merchant_id" in past.columns:
        seen = bool((past["merchant_id"].astype(str).str.strip().str.lower() == vendor).any())
        _set("is_new_merchant", 0 if seen else 1)

    if merchant and merchant.get("recognized") and merchant.get("merchant_risk_score") is not None:
        _set("merchant_risk_score", float(merchant["merchant_risk_score"]))

    return enriched


def build_summary(reasons, is_fraud, risk_level, prob, threshold):
    """One plain sentence that sits above the list of reasons."""
    risks = [r["text"] for r in reasons if r["impact"] == "risk"]
    warns = [r["text"] for r in reasons if r["impact"] == "warn"]
    safes = [r["text"] for r in reasons if r["impact"] == "safe"]
    pct = prob * 100.0
    thr = threshold * 100.0

    if is_fraud:
        lead = "; ".join(_lc(t).rstrip(".") for t in risks[:2])
        if not lead:
            lead = f"the fraud score of {pct:.2f}% reached the {thr:.0f}% block threshold"
        return f"Blocked because {lead}."
    if risk_level == "MEDIUM":
        extra = f" Also, {_lc(warns[0]).rstrip('.')}." if warns else ""
        return f"Approved with caution: the fraud score of {pct:.2f}% is in the watch zone.{extra}"
    lead = "; ".join(_lc(t).rstrip(".") for t in safes[:2])
    if not lead:
        lead = f"the fraud score of {pct:.2f}% is low"
    return f"Approved because {lead}."


def is_why_question(q):
    """True when the user asks why a transaction was safe, approved, blocked or fraud."""
    q = str(q or "").lower()
    asks_why = any(w in q for w in ("why", "reason", "explain", "how come", "what made"))
    about_txn = any(w in q for w in (
        "transaction", "payment", "blocked", "approved", "safe", "fraud",
        "risky", "flagged", "declined", "last", "latest", "recent",
    ))
    generic = any(w in q for w in (
        "how is", "how does", "how do", "calculated", "what is shap",
        "what is lime", "feature importance", "model work",
    ))
    return asks_why and about_txn and not generic


def explain_latest_transaction(user_id, df):
    """Explain the user's most recent saved transaction."""
    if df is None or df.empty or "user_id" not in df.columns:
        return NO_TXN_TEXT
    rows = df[df["user_id"].astype(str) == str(user_id)]
    if rows.empty:
        return NO_TXN_TEXT

    row = rows.iloc[-1]
    status = str(row.get("transaction_status", "") or "UNKNOWN").upper()
    risk = str(row.get("risk_level", "") or "UNKNOWN").upper()
    txid = str(row.get("transaction_id", "") or "unknown")
    recipient = str(row.get("merchant_id", "") or "the recipient")
    amount = _to_float(row.get("amount"))
    prob = _to_float(row.get("fraud_probability"))
    threshold = _to_float(row.get("decision_threshold")) or 0.50
    saved_reasons = str(row.get("reasons", "") or "").strip()

    amount_text = _money(amount) if amount is not None else "an unknown amount"
    head = f"Your latest transaction {txid} ({amount_text} to {recipient}) was {status}."
    if prob is not None:
        head += (f" The fraud probability was {prob * 100.0:.2f}% against a "
                 f"{threshold * 100.0:.0f}% block threshold (risk level {risk}).")

    title = "Why it was blocked:" if status == "BLOCKED" else "Why it was approved:"
    if saved_reasons:
        body = "\n".join(part.strip() for part in saved_reasons.split(" | ") if part.strip())
    elif prob is not None and prob >= threshold:
        body = "✖ The fraud score reached the block threshold."
    elif prob is not None:
        body = "✔ The fraud score stayed below the block threshold."
    else:
        body = "No detailed reasons were saved for this older transaction."

    note = ("These reasons compare the payment with your own history and the merchant "
            "registry alongside the model score. A low score is not a guarantee, so "
            "always check the recipient and amount inside your UPI app.")
    return f"{head}\n\n{title}\n{body}\n\n{note}"