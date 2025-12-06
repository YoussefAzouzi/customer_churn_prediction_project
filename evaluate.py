"""
Model evaluation and analysis module.
"""
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve,
    confusion_matrix, classification_report
)
import joblib
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import MODEL_FILE, PROCESSED_DATA_FILE, TARGET


def load_artifacts():
    """Load model and processed data."""
    model = joblib.load(MODEL_FILE)
    df = pd.read_csv(PROCESSED_DATA_FILE)
    return model, df


def calculate_business_metrics(y_true, y_pred, y_prob, 
                                avg_customer_value=500, 
                                retention_cost=50,
                                retention_success_rate=0.3):
    """
    Calculate business impact metrics.

    Args:
        y_true: actual churn labels
        y_pred: predicted churn labels
        y_prob: churn probabilities
        avg_customer_value: average annual value of a customer
        retention_cost: cost of retention campaign per customer
        retention_success_rate: probability of successful retention
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # True positives: correctly identified churners we can try to retain
    potential_saves = tp * retention_success_rate
    value_saved = potential_saves * avg_customer_value

    # Cost of retention campaigns (targeting predicted churners)
    campaign_cost = (tp + fp) * retention_cost

    # False negatives: churners we missed
    lost_customers = fn
    lost_value = lost_customers * avg_customer_value

    # Net benefit
    net_benefit = value_saved - campaign_cost

    metrics = {
        "true_positives": tp,
        "false_positives": fp,
        "true_negatives": tn,
        "false_negatives": fn,
        "potential_customers_saved": int(potential_saves),
        "value_saved": value_saved,
        "campaign_cost": campaign_cost,
        "net_benefit": net_benefit,
        "lost_customers": lost_customers,
        "lost_value": lost_value,
        "roi": (value_saved - campaign_cost) / campaign_cost * 100 if campaign_cost > 0 else 0
    }

    return metrics


def print_evaluation_report(model, X_test, y_test):
    """Print comprehensive evaluation report."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("\n" + "="*60)
    print("MODEL EVALUATION REPORT")
    print("="*60)

    # Classification metrics
    print("\n--- Classification Metrics ---")
    print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
    print(f"F1 Score:  {f1_score(y_test, y_pred):.4f}")
    print(f"ROC AUC:   {roc_auc_score(y_test, y_prob):.4f}")

    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred, target_names=["Not Churned", "Churned"]))

    # Business metrics
    print("\n--- Business Impact Analysis ---")
    biz_metrics = calculate_business_metrics(y_test, y_pred, y_prob)
    print(f"Correctly identified churners: {biz_metrics['true_positives']}")
    print(f"False alarms: {biz_metrics['false_positives']}")
    print(f"Missed churners: {biz_metrics['false_negatives']}")
    print(f"Potential customers saved: {biz_metrics['potential_customers_saved']}")
    print(f"Estimated value saved: ${biz_metrics['value_saved']:,.2f}")
    print(f"Campaign cost: ${biz_metrics['campaign_cost']:,.2f}")
    print(f"Net benefit: ${biz_metrics['net_benefit']:,.2f}")
    print(f"ROI: {biz_metrics['roi']:.1f}%")

    return y_pred, y_prob, biz_metrics


if __name__ == "__main__":
    from sklearn.model_selection import train_test_split
    from data_preprocessing import preprocess_pipeline, scale_features
    from train import get_feature_columns, prepare_data
    from config import TEST_SIZE, RANDOM_STATE

    # Load and prepare data
    df, _ = preprocess_pipeline()
    X, y, feature_cols, scaler = prepare_data(df)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # Load model and evaluate
    model = joblib.load(MODEL_FILE)
    print_evaluation_report(model, X_test, y_test)
