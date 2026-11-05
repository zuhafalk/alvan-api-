def explain_transaction(txn, fraud_probability):
    reasons = []

    if txn.get("is_unknown_recipient", 0) == 1:
        reasons.append("Recipient is not a verified/registered UPI ID")

    if txn.get("amount", 0) > 10000:
        reasons.append("Large transaction amount")

    if txn.get("merchant_risk_score", 0) > 0.7:
        reasons.append("High-risk merchant")

    if txn.get("ip_change", 0) == 1:
        reasons.append("IP address changed")

    if txn.get("device_change", 0) == 1:
        reasons.append("Device changed")

    if txn.get("txn_velocity", 0) > 5:
        reasons.append("Too many transactions in short time")

    if txn.get("geo_distance_km", 0) > 500:
        reasons.append("Transaction from distant location")

    if txn.get("previous_fraud_count", 0) > 0:
        reasons.append("Previous fraud history")

    if not reasons:
        reasons.append("No suspicious behaviour detected")

    return {
        "probability": fraud_probability,
        "reasons": reasons
    }