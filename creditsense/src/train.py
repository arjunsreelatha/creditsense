import numpy as np
import pandas as pd
import joblib
import os

from data_pipeline import load_csv, train_test_split
from metrics import calculate_metrics, roc_auc

from fraud_features import build_fraud_features
from Fraud_model import FraudModel
from risk_scorer import compute_trust_score
from alert_engine import get_alert_metadata
from explainability import explain_credit_score, explain_fraud_score, explain_verdict

# ── File paths ────────────────────────────────────────────────────────────────
CREDIT_DATA_PATH = r"C:\Users\Lenovo\Desktop\HOPE\creditsense\data\raw\cs-training.csv"
FRAUD_DATA_PATH  = r"C:\Users\Lenovo\Desktop\HOPE\creditsense\data\raw\train_transaction.csv"

# ── Saved model paths ─────────────────────────────────────────────────────────
MODEL_DIR = r"C:\Users\Lenovo\Desktop\HOPE\creditsense\models"
CREDIT_MODEL_PATH  = os.path.join(MODEL_DIR, "credit_model.joblib")
FRAUD_MODEL_PATH   = os.path.join(MODEL_DIR, "fraud_model.joblib")
CREDIT_XTRAIN_PATH = os.path.join(MODEL_DIR, "credit_xtrain.joblib")  # SHAP background data


# ── Credit pipeline (XGBoost, copied from your credit script) ─────────────────
from xgboost import XGBClassifier


def clean_data_for_xgboost(headers: list, data: np.ndarray) -> tuple:
    """No clipping, no normalizing - XGBoost handles NaNs and outliers natively.
    Only drops index column and masks 96/98 special codes as NaN."""
    data = data.copy()
    headers = headers.copy()

    data = np.delete(data, 0, axis=1)
    headers = headers[1:]

    special_code_cols = [3, 7, 9]
    for col_idx in special_code_cols:
        col_data = data[:, col_idx]
        data[:, col_idx] = np.where((col_data == 96) | (col_data == 98), np.nan, col_data)

    return headers, data


def feature_engineering(headers: list, data: np.ndarray) -> tuple:
    """Adds CombinedLate, MonthlyDebt, IncomePerPerson, LogIncome."""
    debt_ratio  = data[:, 4]
    income      = data[:, 5]
    dependents  = data[:, 10]

    combined_late = data[:, 3] + data[:, 7] + data[:, 9]

    monthly_debt = np.zeros(len(data))
    mask_normal = (income > 1) & (~np.isnan(income))
    monthly_debt[mask_normal] = debt_ratio[mask_normal] * income[mask_normal]
    mask_weird = (income <= 1) | (np.isnan(income))
    monthly_debt[mask_weird] = debt_ratio[mask_weird]

    income_per_person = income / (dependents + 1 + 1e-8)
    log_income = np.log1p(income)

    new_features = np.column_stack((combined_late, monthly_debt, income_per_person, log_income))
    data = np.hstack((data, new_features))

    new_headers = headers + ["CombinedLate", "MonthlyDebt", "IncomePerPerson", "LogIncome"]
    return new_headers, data


class CreditModel:
    """Thin wrapper matching the same pattern as FraudModel for consistency."""
    def __init__(self, n_estimators=500, learning_rate=0.04, max_depth=4, random_state=42, scale_pos_weight=15):
        self.model = XGBClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=random_state,
            scale_pos_weight=scale_pos_weight,
        )
        self.feature_names = None
        self.is_trained = False

    def fit(self, X, y, feature_names=None):
        self.feature_names = feature_names
        self.model.fit(X, y)
        self.is_trained = True

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]

    def score_single(self, borrower_features: np.ndarray) -> float:
        x = borrower_features.reshape(1, -1)
        return float(self.model.predict_proba(x)[0, 1])


# ── Training orchestration ─────────────────────────────────────────────────────

def train_credit_model() -> tuple:
    """Loads, engineers, cleans, trains the credit model.
    Returns (model, X_train, feature_names) - X_train is kept for SHAP background."""
    headers, data = load_csv(CREDIT_DATA_PATH)
    headers, data = feature_engineering(headers, data)
    headers, data = clean_data_for_xgboost(headers, data)

    X_train, X_test, y_train, y_test = train_test_split(data, test_size=0.265, shuffle=True, random_state=42)

    model = CreditModel()
    model.fit(X_train, y_train, feature_names=headers[1:])  # headers[0] is the target label

    y_pred = model.predict(X_test)
    accuracy, precision, recall, f1 = calculate_metrics(y_test, y_pred)
    print(f"[Credit Model] Accuracy: {accuracy:.4f}  Precision: {precision:.4f}  Recall: {recall:.4f}  F1: {f1:.4f}")

    return model, X_train, headers[1:]


def train_fraud_model() -> tuple:
    """Loads IEEE-CIS, builds features, trains fraud model.
    Returns (model, feature_names)."""
    X, y, feature_names = build_fraud_features(FRAUD_DATA_PATH)

    data = np.column_stack((y, X))
    X_train, X_test, y_train, y_test = train_test_split(data, test_size=0.2, shuffle=True, random_state=42)

    model = FraudModel()
    model.fit(X_train, y_train, feature_names=feature_names)

    y_pred = model.predict(X_test)
    accuracy, precision, recall, f1 = calculate_metrics(y_test, y_pred)
    print(f"[Fraud Model] Accuracy: {accuracy:.4f}  Precision: {precision:.4f}  Recall: {recall:.4f}  F1: {f1:.4f}")

    return model, feature_names


# ── Global state - models trained once, reused for every verdict call ────────
# Partner's FastAPI route should import get_risk_verdict and call it directly.
# Models train on first import (or call init_models() explicitly at app startup).

_credit_model = None
_credit_X_train = None
_credit_feature_names = None
_fraud_model = None
_fraud_feature_names = None


def init_models(force_retrain: bool = False):
    """
    Loads trained models from disk if they exist. Trains and saves them
    if they don't exist yet, or if force_retrain=True.

    Call this once at FastAPI startup (e.g. in an @app.on_event('startup')
    handler) so every /predict request reuses the same loaded models
    instead of retraining on every request or every server restart.
    """
    global _credit_model, _credit_X_train, _credit_feature_names
    global _fraud_model, _fraud_feature_names

    os.makedirs(MODEL_DIR, exist_ok=True)

    models_exist = (
        os.path.exists(CREDIT_MODEL_PATH)
        and os.path.exists(FRAUD_MODEL_PATH)
        and os.path.exists(CREDIT_XTRAIN_PATH)
    )

    if models_exist and not force_retrain:
        print("Loading saved models from disk...")
        _credit_model, _credit_feature_names = joblib.load(CREDIT_MODEL_PATH)
        _fraud_model, _fraud_feature_names = joblib.load(FRAUD_MODEL_PATH)
        _credit_X_train = joblib.load(CREDIT_XTRAIN_PATH)
        print("Models loaded.")
        return

    print("No saved models found (or force_retrain=True). Training from scratch...")

    _credit_model, _credit_X_train, _credit_feature_names = train_credit_model()
    _fraud_model, _fraud_feature_names = train_fraud_model()

    # Save everything to disk for next time
    joblib.dump((_credit_model, _credit_feature_names), CREDIT_MODEL_PATH)
    joblib.dump((_fraud_model, _fraud_feature_names), FRAUD_MODEL_PATH)
    joblib.dump(_credit_X_train, CREDIT_XTRAIN_PATH)
    print(f"Models saved to {MODEL_DIR}")


def get_risk_verdict(borrower_profile: dict, transaction: dict) -> dict:
    """
    THE single function partner's FastAPI route calls.

    Args:
        borrower_profile: dict matching credit feature names, e.g.
            {"RevolvingUtilizationOfUnsecuredLines": 0.3, "age": 45, ...}
        transaction: dict with fraud feature names, e.g.
            {"velocity": 2, "amount_deviation": 1.3, "balance_ratio": 0.04}

    Returns:
        dict - fully JSON-serializable. Partner does `return get_risk_verdict(...)`
        directly in the FastAPI route, no extra processing needed.
    """
    if _credit_model is None or _fraud_model is None:
        init_models()

    # Build feature vectors in the exact order the models were trained on
    borrower_vector = np.array([borrower_profile.get(f, 0.0) for f in _credit_feature_names])
    transaction_vector = np.array([transaction.get(f, 0.0) for f in _fraud_feature_names])

    credit_explanation = explain_credit_score(
        _credit_model.model, _credit_X_train, borrower_vector, _credit_feature_names
    )
    fraud_explanation = explain_fraud_score(
        _fraud_model, transaction_vector, _fraud_feature_names
    )

    scores = compute_trust_score(
        credit_score=credit_explanation["credit_score"],
        fraud_score=fraud_explanation["fraud_score"],
    )

    alert = get_alert_metadata(scores["trust_score"])

    verdict = explain_verdict(
        credit_explanation=credit_explanation,
        fraud_explanation=fraud_explanation,
        trust_score=scores["trust_score"],
        alert_level=alert["alert_level"],
    )

    # Merge in alert metadata (color, action, message) for the frontend
    verdict["color"] = alert["color"]
    verdict["action"] = alert["action"]
    verdict["message"] = alert["message"]

    return verdict


if __name__ == "__main__":
    init_models()

    # Smoke test with a sample borrower + transaction
    sample_borrower = {
        "RevolvingUtilizationOfUnsecuredLines": 0.85,
        "age": 35,
        "DebtRatio": 0.6,
        "MonthlyIncome": 3000,
        "NumberOfDependents": 2,
    }
    sample_transaction = {
        "velocity": 5,
        "amount_deviation": 3.2,
        "balance_ratio": 0.08,
    }

    result = get_risk_verdict(sample_borrower, sample_transaction)

    import json
    print("\n--- Sample Verdict ---")
    print(json.dumps(result, indent=2))