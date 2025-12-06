"""
Model training module for Customer Churn Prediction.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)
import joblib
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import (
    PROCESSED_DATA_FILE, MODEL_FILE, MODELS_DIR,
    CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET, ID_COL,
    TEST_SIZE, RANDOM_STATE
)
from data_preprocessing import preprocess_pipeline, scale_features


def get_feature_columns(df):
    """Get list of feature columns for modeling."""
    encoded_cats = [col + "_encoded" for col in CATEGORICAL_FEATURES if col + "_encoded" in df.columns]
    engineered = ["spend_per_month", "purchase_frequency", "engagement_score", 
                  "recency_score", "support_intensity", "is_high_value", "is_new_customer"]
    engineered = [col for col in engineered if col in df.columns]

    return NUMERIC_FEATURES + encoded_cats + engineered


def prepare_data(df):
    """Prepare features and target for training."""
    feature_cols = get_feature_columns(df)

    X = df[feature_cols].copy()
    y = df[TARGET].copy()

    # Scale features
    X, scaler = scale_features(X, feature_cols, fit=True)

    return X, y, feature_cols, scaler


def train_models(X_train, y_train):
    """Train multiple models and return the best one."""
    models = {
        "Logistic Regression": LogisticRegression(random_state=RANDOM_STATE, max_iter=1000),
        "Random Forest": RandomForestClassifier(random_state=RANDOM_STATE, n_estimators=100),
        "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE, n_estimators=100)
    }

    results = {}
    for name, model in models.items():
        print(f"\nTraining {name}...")
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="roc_auc")
        model.fit(X_train, y_train)
        results[name] = {
            "model": model,
            "cv_score": cv_scores.mean(),
            "cv_std": cv_scores.std()
        }
        print(f"  CV ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    # Select best model
    best_name = max(results, key=lambda x: results[x]["cv_score"])
    print(f"\nBest model: {best_name}")

    return results[best_name]["model"], results


def evaluate_model(model, X_test, y_test):
    """Evaluate model on test set."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob)
    }

    print("\n" + "="*50)
    print("MODEL EVALUATION RESULTS")
    print("="*50)
    for metric, value in metrics.items():
        print(f"{metric.upper()}: {value:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Not Churned", "Churned"]))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    return metrics


def get_feature_importance(model, feature_cols):
    """Get feature importance from the model."""
    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
    elif hasattr(model, "coef_"):
        importance = np.abs(model.coef_[0])
    else:
        return None

    importance_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": importance
    }).sort_values("importance", ascending=False)

    print("\nTop 10 Important Features:")
    print(importance_df.head(10).to_string(index=False))

    return importance_df


def save_model(model, filepath=MODEL_FILE):
    """Save trained model to disk."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model, filepath)
    print(f"\nModel saved to {filepath}")


def main():
    """Main training pipeline."""
    print("="*50)
    print("CUSTOMER CHURN PREDICTION - MODEL TRAINING")
    print("="*50)

    # Preprocess data
    df, encoders = preprocess_pipeline()

    # Prepare features
    X, y, feature_cols, scaler = prepare_data(df)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"\nTraining set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    print(f"Churn rate in training: {y_train.mean():.2%}")

    # Train models
    best_model, all_results = train_models(X_train, y_train)

    # Evaluate
    metrics = evaluate_model(best_model, X_test, y_test)

    # Feature importance
    importance_df = get_feature_importance(best_model, feature_cols)

    # Save model
    save_model(best_model)

    return best_model, metrics, importance_df


if __name__ == "__main__":
    model, metrics, importance = main()
