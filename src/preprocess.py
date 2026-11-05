"""
preprocess.py
Data preprocessing and feature engineering for UPI Fraud Detection
Compatible with the existing 20-feature training/inference pipeline.

Key fixes:
- Keeps live and training feature generation consistent.
- Uses an existing `user_txn_count` for `user_total_txn_count` when available,
  instead of counting the single live prediction row.
- Falls back to grouping by user only when a real transaction dataset is being
  processed and `user_txn_count` is unavailable.
- Avoids duplicate-column issues.
- Handles missing numeric values safely.
"""

import pandas as pd
import numpy as np


# -------------------------------------------------------
# BASIC CLEANING
# -------------------------------------------------------
def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and normalize a UPI transaction DataFrame."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("basic_clean() expects a pandas DataFrame.")

    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    num_cols = [
        "amount",
        "user_avg_amount",
        "user_txn_count",
        "user_age_days",
        "user_device_count",
        "merchant_risk_score",
        "hour",
        "day_of_week",
        "txn_velocity",
        "previous_fraud_count",
        "geo_distance_km",
        "is_new_merchant",
        "is_unknown_recipient",
        "ip_change",
        "device_change",
    ]

    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        medians = df[numeric_cols].median()
        df[numeric_cols] = df[numeric_cols].fillna(medians)
        df[numeric_cols] = df[numeric_cols].fillna(0)

    return df


# -------------------------------------------------------
# FEATURE ENGINEERING
# -------------------------------------------------------
def feature_engineer(df: pd.DataFrame):
    """
    Add the derived features used by train.py and predict.py.

    Important:
    When `user_txn_count` is already supplied, it is used as the user's
    historical transaction count. This prevents a one-row live prediction
    from being interpreted as the user's complete transaction count.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("feature_engineer() expects a pandas DataFrame.")

    df = df.copy()
    df = df.loc[:, ~df.columns.duplicated()].copy()

    # ---------------------------------------------------
    # 1. USER TRANSACTION COUNT
    # ---------------------------------------------------
    if "user_txn_count" in df.columns:
        df["user_total_txn_count"] = pd.to_numeric(
            df["user_txn_count"], errors="coerce"
        ).fillna(0)
    elif "user_id" in df.columns and "transaction_id" in df.columns:
        user_counts = (
            df.groupby("user_id")["transaction_id"]
            .count()
            .rename("user_total_txn_count")
        )
        df["user_total_txn_count"] = df["user_id"].map(user_counts).fillna(0)
    else:
        df["user_total_txn_count"] = 0

    # ---------------------------------------------------
    # 2. AMOUNT DEVIATION FROM USER AVERAGE
    # ---------------------------------------------------
    if "amount" in df.columns and "user_avg_amount" in df.columns:
        amount = pd.to_numeric(df["amount"], errors="coerce")
        average = pd.to_numeric(df["user_avg_amount"], errors="coerce")
        df["amount_dev_from_avg"] = (amount - average).fillna(0)

    # ---------------------------------------------------
    # 3. LOG OF AMOUNT
    # ---------------------------------------------------
    if "amount" in df.columns:
        amount = pd.to_numeric(df["amount"], errors="coerce").fillna(0)
        df["log_amount"] = np.log1p(amount.clip(lower=0))

    # ---------------------------------------------------
    # 4. TIME-BASED FEATURES
    # ---------------------------------------------------
    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"], errors="coerce")

        calculated_hour = ts.dt.hour
        calculated_day = ts.dt.weekday

        if calculated_hour.notna().any():
            df["hour"] = calculated_hour
        elif "hour" not in df.columns:
            df["hour"] = 0

        if calculated_day.notna().any():
            df["day_of_week"] = calculated_day
        elif "day_of_week" not in df.columns:
            df["day_of_week"] = 0

    if "hour" in df.columns:
        df["hour"] = pd.to_numeric(df["hour"], errors="coerce").fillna(0)

    if "day_of_week" in df.columns:
        df["day_of_week"] = pd.to_numeric(
            df["day_of_week"], errors="coerce"
        ).fillna(0)

    # ---------------------------------------------------
    # 5. HIGH-RISK TIME WINDOW
    # ---------------------------------------------------
    if "hour" in df.columns:
        hour = pd.to_numeric(df["hour"], errors="coerce").fillna(0)
        df["is_odd_hour"] = ((hour < 5) | (hour > 22)).astype(int)

    # ---------------------------------------------------
    # 6. DEVICE/IP INTERACTION
    # ---------------------------------------------------
    if "ip_change" in df.columns and "device_change" in df.columns:
        ip = pd.to_numeric(df["ip_change"], errors="coerce").fillna(0).astype(int)
        device = pd.to_numeric(
            df["device_change"], errors="coerce"
        ).fillna(0).astype(int)
        df["device_ip_combo"] = ip.astype(str) + "_" + device.astype(str)

    # ---------------------------------------------------
    # 7. USER BEHAVIOR RISK
    # ---------------------------------------------------
    risk_factors = []

    if "user_age_days" in df.columns:
        age = pd.to_numeric(df["user_age_days"], errors="coerce").fillna(180)
        effective_age = age.clip(lower=14)
        risk_factors.append((1.0 / (effective_age + 1.0)) * 1000.0)

    if "user_device_count" in df.columns:
        devices = pd.to_numeric(
            df["user_device_count"], errors="coerce"
        ).fillna(1)
        risk_factors.append(devices * 5.0)

    if "previous_fraud_count" in df.columns:
        previous_fraud = pd.to_numeric(
            df["previous_fraud_count"], errors="coerce"
        ).fillna(0)
        risk_factors.append(previous_fraud * 10.0)

    if risk_factors:
        df["user_behavior_risk"] = np.sum(
            np.vstack([np.asarray(x) for x in risk_factors]),
            axis=0,
        )

    # ---------------------------------------------------
    # 8. FINAL CLEANUP
    # ---------------------------------------------------
    df = df.loc[:, ~df.columns.duplicated()].copy()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        df[numeric_cols] = df[numeric_cols].fillna(0)

    return df, None


if __name__ == "__main__":
    sample = pd.DataFrame(
        [{
            "transaction_id": "TXN_demo",
            "user_id": "tasbhi",
            "amount": 1500.0,
            "user_avg_amount": 15000.0,
            "user_txn_count": 1,
            "user_age_days": 180,
            "user_device_count": 1,
            "merchant_risk_score": 0.10,
            "ip_change": 0,
            "device_change": 0,
            "txn_velocity": 1,
            "previous_fraud_count": 0,
            "is_new_merchant": 1,
            "is_unknown_recipient": 1,
            "geo_distance_km": 0,
            "timestamp": "2026-09-24 19:40:10",
        }]
    )

    cleaned = basic_clean(sample)
    engineered, _ = feature_engineer(cleaned)

    print("preprocess.py self-test")
    print("user_txn_count:", engineered.loc[0, "user_txn_count"])
    print("user_total_txn_count:", engineered.loc[0, "user_total_txn_count"])
    print("amount_dev_from_avg:", engineered.loc[0, "amount_dev_from_avg"])
    print("log_amount:", engineered.loc[0, "log_amount"])
