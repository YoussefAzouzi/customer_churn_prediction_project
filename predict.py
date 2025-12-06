"""
Prediction module for Customer Churn Prediction.
"""
import pandas as pd
import numpy as np
import joblib
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import MODEL_FILE, SCALER_FILE, CATEGORICAL_FEATURES, NUMERIC_FEATURES


def load_model(model_path=MODEL_FILE):
    """Load trained model from disk."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Please train the model first.")
    return joblib.load(model_path)


def load_scaler(scaler_path=SCALER_FILE):
    """Load fitted scaler from disk."""
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler not found at {scaler_path}. Please train the model first.")
    return joblib.load(scaler_path)


def predict_churn(customer_data, model=None, scaler=None):
    """
    Predict churn probability for customer(s).

    Args:
        customer_data: dict or DataFrame with customer features
        model: trained model (will load if not provided)
        scaler: fitted scaler (will load if not provided)

    Returns:
        DataFrame with predictions and probabilities
    """
    if model is None:
        model = load_model()
    if scaler is None:
        scaler = load_scaler()

    # Convert to DataFrame if dict
    if isinstance(customer_data, dict):
        df = pd.DataFrame([customer_data])
    else:
        df = customer_data.copy()

    # Get feature columns (same as training)
    feature_cols = scaler.feature_names_in_

    # Ensure all features exist
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0

    # Scale features
    X = scaler.transform(df[feature_cols])

    # Predict
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]

    results = pd.DataFrame({
        "churn_prediction": predictions,
        "churn_probability": probabilities,
        "risk_level": pd.cut(probabilities, bins=[0, 0.3, 0.6, 1.0], labels=["Low", "Medium", "High"])
    })

    return results


def interactive_predict():
    """Interactive prediction from user input."""
    print("\n" + "="*50)
    print("CUSTOMER CHURN PREDICTION")
    print("="*50)
    print("Enter customer information (press Enter for default values):\n")

    def get_input(prompt, default, dtype=float):
        val = input(f"{prompt} [{default}]: ").strip()
        if val == "":
            return default
        return dtype(val)

    customer = {
        "age": get_input("Age", 35, int),
        "tenure_months": get_input("Tenure (months)", 12, int),
        "total_purchases": get_input("Total purchases", 10, int),
        "total_spend": get_input("Total spend ($)", 500, float),
        "avg_order_value": get_input("Avg order value ($)", 50, float),
        "days_since_last_purchase": get_input("Days since last purchase", 30, int),
        "num_support_tickets": get_input("Number of support tickets", 2, int),
        "satisfaction_score": get_input("Satisfaction score (1-5)", 3.5, float),
        "email_open_rate": get_input("Email open rate (0-1)", 0.3, float),
        "website_visits_last_month": get_input("Website visits last month", 5, int),
        "discount_usage_rate": get_input("Discount usage rate (0-1)", 0.4, float),
    }

    # Add encoded categorical features (using default encoding)
    customer["gender_encoded"] = get_input("Gender (0=Female, 1=Male, 2=Other)", 1, int)
    customer["membership_tier_encoded"] = get_input("Membership (0=Bronze, 1=Gold, 2=Platinum, 3=Silver)", 0, int)
    customer["preferred_category_encoded"] = get_input("Category (0=Beauty, 1=Clothing, 2=Electronics, 3=Home, 4=Sports)", 2, int)
    customer["payment_method_encoded"] = get_input("Payment (0=Bank, 1=Credit, 2=Debit, 3=PayPal)", 1, int)

    # Add engineered features
    customer["spend_per_month"] = customer["total_spend"] / max(customer["tenure_months"], 1)
    customer["purchase_frequency"] = customer["total_purchases"] / max(customer["tenure_months"], 1)
    customer["engagement_score"] = (customer["email_open_rate"] * 50) + (customer["website_visits_last_month"] * 2)
    customer["recency_score"] = 100 / (customer["days_since_last_purchase"] + 1)
    customer["support_intensity"] = customer["num_support_tickets"] / max(customer["tenure_months"], 1)
    customer["is_high_value"] = 1 if customer["total_spend"] > 750 else 0
    customer["is_new_customer"] = 1 if customer["tenure_months"] < 6 else 0

    # Predict
    result = predict_churn(customer)

    print("\n" + "="*50)
    print("PREDICTION RESULT")
    print("="*50)
    print(f"Churn Probability: {result['churn_probability'].values[0]:.1%}")
    print(f"Risk Level: {result['risk_level'].values[0]}")
    print(f"Prediction: {'WILL CHURN' if result['churn_prediction'].values[0] == 1 else 'WILL NOT CHURN'}")

    return result


if __name__ == "__main__":
    interactive_predict()
