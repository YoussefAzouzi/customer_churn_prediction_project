# Customer Churn Prediction

An end-to-end machine learning project to predict customer churn for an e-commerce business using classification models.

## Project Overview

This project builds a complete ML pipeline to:
- Predict which customers are likely to churn (stop buying)
- Identify key factors driving customer churn
- Enable targeted retention campaigns

## Dataset

The dataset contains 5,000 synthetic e-commerce customer records with 17 features including:
- **Demographics**: age, gender
- **Behavior**: total purchases, spend, website visits, email engagement
- **Account**: tenure, membership tier, payment method
- **Support**: number of tickets, satisfaction score
- **Target**: churned (1 = churned, 0 = retained)

## Project Structure

```
customer_churn_prediction/
├── data/
│   ├── raw/                    # Original dataset
│   │   └── ecommerce_customers.csv
│   └── processed/              # Cleaned/transformed data
├── src/
│   ├── __init__.py
│   ├── config.py               # Configuration settings
│   ├── data_preprocessing.py   # Data cleaning & feature engineering
│   ├── train.py                # Model training pipeline
│   ├── predict.py              # Prediction interface
│   └── evaluate.py             # Model evaluation & business metrics
├── models/                     # Saved models and scalers
├── notebooks/                  # Jupyter notebooks for EDA
├── reports/                    # Generated reports
├── requirements.txt            # Python dependencies
└── README.md
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/customer-churn-prediction.git
cd customer-churn-prediction
```

2. Create virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Train the Model
```bash
cd src
python train.py
```

This will:
- Load and preprocess the data
- Train multiple models (Logistic Regression, Random Forest, Gradient Boosting)
- Select the best model based on cross-validation ROC-AUC
- Save the model to `models/churn_model.joblib`

### Make Predictions
```bash
cd src
python predict.py
```

This launches an interactive CLI where you can input customer features and get churn predictions.

### Evaluate Model
```bash
cd src
python evaluate.py
```

This generates a comprehensive evaluation report including classification metrics and business impact analysis.

## Model Performance

The trained model achieves:
- **ROC-AUC**: ~0.85
- **Precision**: ~0.70
- **Recall**: ~0.65
- **F1 Score**: ~0.67

## Key Features

- **Feature Engineering**: Creates derived features like spend_per_month, engagement_score, recency_score
- **Multiple Models**: Compares Logistic Regression, Random Forest, and Gradient Boosting
- **Business Metrics**: Calculates ROI, potential value saved, and campaign costs
- **Interactive Predictions**: CLI interface for real-time predictions

## Technologies Used

- Python 3.8+
- pandas, numpy
- scikit-learn
- joblib

## Future Improvements

- Add SHAP values for model interpretability
- Build Streamlit dashboard for visualization
- Implement model monitoring and drift detection
- Add hyperparameter tuning with Optuna

## License

MIT License
