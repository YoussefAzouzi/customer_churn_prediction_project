"""
Data preprocessing and feature engineering module.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import os
from config import (
    RAW_DATA_FILE, PROCESSED_DATA_FILE, SCALER_FILE,
    CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET, ID_COL, DATA_PROCESSED, MODELS_DIR
)


def load_data(filepath=RAW_DATA_FILE):
    """Load raw data from CSV."""
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} records from {filepath}")
    return df


def clean_data(df):
    """Clean and handle missing values."""
    df = df.copy()

    # Handle missing values
    for col in NUMERIC_FEATURES:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0])

    # Remove duplicates
    df = df.drop_duplicates(subset=[ID_COL], keep="first")

    print(f"Data cleaned. Shape: {df.shape}")
    return df


def engineer_features(df):
    """Create new features from existing data."""
    df = df.copy()

    # Spend per month
    df["spend_per_month"] = df["total_spend"] / df["tenure_months"].replace(0, 1)

    # Purchase frequency (purchases per month)
    df["purchase_frequency"] = df["total_purchases"] / df["tenure_months"].replace(0, 1)

    # Engagement score (combination of email opens and website visits)
    df["engagement_score"] = (df["email_open_rate"] * 50) + (df["website_visits_last_month"] * 2)

    # Recency score (inverse of days since last purchase)
    df["recency_score"] = 100 / (df["days_since_last_purchase"] + 1)

    # Support intensity (tickets per month of tenure)
    df["support_intensity"] = df["num_support_tickets"] / df["tenure_months"].replace(0, 1)

    # Is high value customer
    df["is_high_value"] = (df["total_spend"] > df["total_spend"].quantile(0.75)).astype(int)

    # Is new customer (tenure < 6 months)
    df["is_new_customer"] = (df["tenure_months"] < 6).astype(int)

    print(f"Feature engineering complete. New shape: {df.shape}")
    return df


def encode_features(df, fit=True, encoders=None):
    """Encode categorical features."""
    df = df.copy()

    if encoders is None:
        encoders = {}

    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            if fit:
                le = LabelEncoder()
                df[col + "_encoded"] = le.fit_transform(df[col].astype(str))
                encoders[col] = le
            else:
                le = encoders[col]
                df[col + "_encoded"] = le.transform(df[col].astype(str))

    return df, encoders


def scale_features(df, feature_cols, fit=True, scaler=None):
    """Scale numeric features."""
    df = df.copy()

    if fit:
        scaler = StandardScaler()
        df[feature_cols] = scaler.fit_transform(df[feature_cols])
        os.makedirs(MODELS_DIR, exist_ok=True)
        joblib.dump(scaler, SCALER_FILE)
        print(f"Scaler saved to {SCALER_FILE}")
    else:
        df[feature_cols] = scaler.transform(df[feature_cols])

    return df, scaler


def preprocess_pipeline(filepath=RAW_DATA_FILE, save=True):
    """Run full preprocessing pipeline."""
    # Load
    df = load_data(filepath)

    # Clean
    df = clean_data(df)

    # Feature engineering
    df = engineer_features(df)

    # Encode
    df, encoders = encode_features(df, fit=True)

    # Save processed data
    if save:
        os.makedirs(DATA_PROCESSED, exist_ok=True)
        df.to_csv(PROCESSED_DATA_FILE, index=False)
        print(f"Processed data saved to {PROCESSED_DATA_FILE}")

    return df, encoders


if __name__ == "__main__":
    df, encoders = preprocess_pipeline()
    print("\nProcessed data preview:")
    print(df.head())
    print(f"\nTarget distribution:\n{df[TARGET].value_counts(normalize=True)}")
