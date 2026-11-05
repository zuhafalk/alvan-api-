"""
predict.py
Helper functions for loading the saved model and making predictions on new transactions.
"""

import os
import joblib
import pandas as pd
import numpy as np

MODEL_IN = os.path.join(
    os.path.dirname(__file__),
    "..",
    "models",
    "lightgbm_model.pkl",
)


# ---------------------------------------------------------
# DEFAULTS FOR MISSING OPTIONAL FIELDS
# ---------------------------------------------------------
# These defaults are intentionally neutral. They are used only when the
# caller does not have real behavioural context. The Streamlit application
# supplies real stored history whenever it has it.
DEFAULT_CONTEXT = {
    "user_txn_count": 10,
    "user_age_days": 180,
    "user_device_count": 1,
    "merchant_risk_score": 0.10,
    "txn_velocity": 1,
    "previous_fraud_count": 0,
    "geo_distance_km": 0,
    "ip_change": 0,
    "device_change": 0,
    "is_new_merchant": 0,
    "is_unknown_recipient": 0,
}


def _to_float(value, default=0.0):
    try:
        value = float(value)
        if np.isfinite(value):
            return value
    except (TypeError, ValueError):
        pass
    return float(default)


def apply_defaults(transaction: dict) -> dict:
    """Fill missing optional transaction context with neutral values."""
    txn = dict(transaction or {})

    # Accept either spelling used by older versions/datasets.
    if "upi_app" not in txn and "UPI_app" in txn:
        txn["upi_app"] = txn["UPI_app"]
    if "UPI_app" not in txn and "upi_app" in txn:
        txn["UPI_app"] = txn["upi_app"]

    amount = _to_float(txn.get("amount"), 0.0)
    txn["amount"] = amount

    for key, default in DEFAULT_CONTEXT.items():
        if key not in txn or txn[key] is None or (isinstance(txn[key], float) and np.isnan(txn[key])):
            txn[key] = default

    # Missing user average is neutral: treat this transaction as the current
    # baseline rather than manufacturing a large deviation from zero.
    if "user_avg_amount" not in txn or txn["user_avg_amount"] is None:
        txn["user_avg_amount"] = amount

    txn["user_avg_amount"] = _to_float(txn["user_avg_amount"], amount)
    return txn


def load_model():
    """Load model, encoders, training feature order and optional threshold."""
    data = joblib.load(MODEL_IN)

    if isinstance(data, dict):
        model = data.get("model")
        encoders = data.get("encoders", {})
        feature_names = data.get("feature_names", []) or []
        threshold = data.get(
            "decision_threshold",
            data.get("classification_threshold", data.get("threshold", 0.50)),
        )
    else:
        model = data
        encoders = {}
        feature_names = []
        threshold = 0.50

    threshold = float(np.clip(_to_float(threshold, 0.50), 0.0, 1.0))
    return model, encoders, feature_names, threshold


def prepare_single(transaction: dict):
    """Prepare one transaction using exactly the project's preprocess pipeline."""
    df = pd.DataFrame([transaction])

    from preprocess import basic_clean, feature_engineer

    df = basic_clean(df)
    df, _ = feature_engineer(df)
    return df


def _apply_saved_encoders(df: pd.DataFrame, encoders: dict) -> pd.DataFrame:
    for col, le in (encoders or {}).items():
        if col not in df.columns:
            continue

        df[col] = df[col].astype(str)
        known = set(str(x) for x in le.classes_)
        df[col] = df[col].apply(
            lambda x: le.transform([x])[0] if x in known else -1
        )

    return df


def _fraud_probability(model, X) -> float:
    """Get probability for the fraud class while respecting model classes."""
    if hasattr(model, "predict_proba"):
        probabilities = np.asarray(model.predict_proba(X), dtype=float)[0]

        classes = getattr(model, "classes_", None)
        if classes is not None:
            classes = list(classes)
            if 1 in classes:
                index = classes.index(1)
            elif "1" in classes:
                index = classes.index("1")
            else:
                index = 1 if len(probabilities) > 1 else 0
        else:
            index = 1 if len(probabilities) > 1 else 0

        return float(np.clip(probabilities[index], 0.0, 1.0))

    prediction = float(model.predict(X)[0])
    return float(np.clip(prediction, 0.0, 1.0))


def predict_transaction(transaction: dict):
    """
    Predict one transaction.

    Returned probability is ALWAYS a decimal in [0, 1]. The API converts
    nothing implicitly, so the UI can display probability * 100 exactly once.
    """
    transaction = apply_defaults(transaction)

    model, encoders, feature_names, threshold = load_model()
    if model is None:
        raise RuntimeError("Saved LightGBM model could not be loaded.")

    df = prepare_single(transaction)
    df = _apply_saved_encoders(df, encoders)

    if feature_names:
        for col in feature_names:
            if col not in df.columns:
                df[col] = 0
        df = df[feature_names]

    X = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    proba = _fraud_probability(model, X)
    label = int(proba >= threshold)

    return {
        "probability_fraud": proba,
        "fraud_probability": proba,
        "is_fraud": label,
        "decision_threshold": threshold,
    }


if __name__ == "__main__":
    sample_txn_full = {
        "user_id": "user_999",
        "merchant_id": "unknown",
        "txn_type": "send",
        "upi_app": "gpay",
        "amount": 95000,
        "user_avg_amount": 5000,
        "user_txn_count": 25,
        "user_age_days": 5,
        "user_device_count": 3,
        "merchant_risk_score": 0.95,
        "ip_change": 1,
        "device_change": 1,
        "txn_velocity": 20,
        "previous_fraud_count": 5,
        "is_new_merchant": 1,
        "is_unknown_recipient": 1,
        "geo_distance_km": 250,
    }

    print("Full payload result:", predict_transaction(sample_txn_full))

    sample_txn_minimal = {
        "user_id": "user_123",
        "amount": 500,
    }

    print("Minimal payload result:", predict_transaction(sample_txn_minimal))
