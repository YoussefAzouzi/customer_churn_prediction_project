"""
Configuration settings for Customer Churn Prediction project.
"""
import os

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW = os.path.join(BASE_DIR, "data", "raw")
DATA_PROCESSED = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Data files
RAW_DATA_FILE = os.path.join(DATA_RAW, "ecommerce_customers.csv")
PROCESSED_DATA_FILE = os.path.join(DATA_PROCESSED, "processed_customers.csv")
MODEL_FILE = os.path.join(MODELS_DIR, "churn_model.joblib")
SCALER_FILE = os.path.join(MODELS_DIR, "scaler.joblib")

# Feature columns
CATEGORICAL_FEATURES = ["gender", "membership_tier", "preferred_category", "payment_method"]
NUMERIC_FEATURES = [
    "age", "tenure_months", "total_purchases", "total_spend", 
    "avg_order_value", "days_since_last_purchase", "num_support_tickets",
    "satisfaction_score", "email_open_rate", "website_visits_last_month",
    "discount_usage_rate"
]
TARGET = "churned"
ID_COL = "customer_id"

# Model parameters
TEST_SIZE = 0.2
RANDOM_STATE = 42
