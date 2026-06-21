import shap
import numpy as np
import pandas as pd

# ── Explainability module ─────────────────────────────────────────────────────
# Credit layer: real SHAP (TreeExplainer via XGBoost model)
# Fraud layer: XGBoost feature_importances_ (SHAP optional upgrade later, same model type)
#
# IMPORTANT: every function here returns plain dicts/floats/strings only —
# no DataFrames, no numpy types — so the output is JSON-serializable.
# Partner's FastAPI route just does `return get_risk_verdict(...)` and
# FastAPI handles serialization automatically. No API code needed from Arjun.
# ─────────────────────────────────────────────────────────────────────────────

CREDIT_FEATURE_DESCRIPTIONS = {
    "RevolvingUtilizationOfUnsecuredLines": "Credit card usage relative to limit",
    "age":                                  "Borrower age",
    "NumberOfTime30-59DaysPastDueNotWorse": "Late payments (30-59 days)",
    "DebtRatio":                            "Monthly debt vs income ratio",
    "MonthlyIncome":                        "Monthly income",
    "NumberOfOpenCreditLinesAndLoans":      "Number of open credit lines",
    "NumberOfTimes90DaysLate":              "Severely late payments (90+ days)",
    "NumberRealEstateLoansOrLines":         "Real estate loans",
    "NumberOfTime60-89DaysPastDueNotWorse": "Late payments (60-89 days)",
    "NumberOfDependents":                   "Number of dependents",
    "CombinedLate":                         "Total late payment history",
    "MonthlyDebt":                          "Estimated monthly debt amount",
    "IncomePerPerson":                      "Income per family member",
    "LogIncome":                            "Income (log scaled)",
}

FRAUD_FEATURE_DESCRIPTIONS = {
    "velocity":          "Number of transactions in the last hour",
    "amount_deviation":  "How unusual this transaction amount is",
    "balance_ratio":     "Number of addresses linked to this card",
}


# ── Credit explainability (real SHAP) ─────────────────────────────────────────

def explain_credit_score(model, X_train: np.ndarray, X_single: np.ndarray, feature_names: list = None) -> dict:
    """
    Explains a single borrower's credit score using real SHAP values.

    Args:
        model        : trained credit model (XGBoost - must support shap.Explainer)
        X_train      : training feature matrix, used as SHAP background distribution
        X_single     : 1D numpy array of shape (n_features,) - one borrower
        feature_names: list of feature name strings

    Returns:
        dict - fully JSON-serializable, no DataFrames, ready for FastAPI to return directly
    """
    columns = feature_names if feature_names is not None else [f"f{i}" for i in range(X_train.shape[1])]

    X_train_df = pd.DataFrame(X_train, columns=columns)
    X_single_df = pd.DataFrame(X_single.reshape(1, -1), columns=columns)

    explainer = shap.Explainer(model, X_train_df)
    shap_values = explainer(X_single_df)

    base_value = float(shap_values.base_values[0])
    prediction = float(model.predict_proba(X_single_df)[0, 1])

    contribs = pd.DataFrame({
        "feature":    X_single_df.columns,
        "value":      X_single_df.iloc[0].values,
        "shap_value": shap_values.values[0],
    }).sort_values("shap_value", key=lambda s: s.abs(), ascending=False)

    top_factors = []
    for _, row in contribs.head(5).iterrows():
        feat = row["feature"]
        top_factors.append({
            "feature":     feat,
            "value":       round(float(row["value"]), 4),
            "shap_value":  round(float(row["shap_value"]), 4),
            "direction":   "increases risk" if row["shap_value"] > 0 else "decreases risk",
            "description": CREDIT_FEATURE_DESCRIPTIONS.get(feat, feat),
        })

    top_feat = top_factors[0]
    plain_explanation = (
        f"Credit score: {prediction:.2f} - "
        f"primarily driven by '{top_feat['description']}' "
        f"(value: {top_feat['value']}, {top_feat['direction']})."
    )

    return {
        "credit_score":       round(prediction, 4),
        "base_value":         round(base_value, 4),
        "top_factors":        top_factors,
        "plain_explanation":  plain_explanation,
    }


# ── Fraud explainability ──────────────────────────────────────────────────────

def explain_fraud_score(model, transaction_features: np.ndarray, feature_names: list = None) -> dict:
    """
    Explains a single transaction's fraud score using XGBoost feature importances.

    Args:
        model               : trained FraudModel instance
        transaction_features: 1D numpy array [velocity, amount_deviation, balance_ratio]
        feature_names       : list of feature names (default: standard fraud features)

    Returns:
        dict - fully JSON-serializable, same shape pattern as explain_credit_score()
    """
    if feature_names is None:
        feature_names = ["velocity", "amount_deviation", "balance_ratio"]

    x = transaction_features.reshape(1, -1)
    fraud_score = float(model.predict_proba(x)[0])

    importance_dict = model.get_feature_importance()

    top_factors = []
    for feat in feature_names:
        imp = float(importance_dict.get(feat, 0.0))
        val = float(transaction_features[feature_names.index(feat)])
        top_factors.append({
            "feature":     feat,
            "value":       round(val, 4),
            "importance":  round(imp, 4),
            "description": FRAUD_FEATURE_DESCRIPTIONS.get(feat, feat),
        })

    top_factors = sorted(top_factors, key=lambda t: t["importance"], reverse=True)

    top_feat = top_factors[0]
    plain_explanation = (
        f"Fraud score: {fraud_score:.2f} - "
        f"primarily driven by '{top_feat['description']}' "
        f"(value: {top_feat['value']})."
    )

    return {
        "fraud_score":        round(fraud_score, 4),
        "top_factors":        top_factors,
        "plain_explanation":  plain_explanation,
    }


# ── Combined explanation ──────────────────────────────────────────────────────

def explain_verdict(credit_explanation: dict, fraud_explanation: dict, trust_score: float, alert_level: str) -> dict:
    """
    Combines credit and fraud explanations into one final verdict explanation.
    This is the exact dict get_risk_verdict() returns. Partner's FastAPI route
    just does `return get_risk_verdict(...)` - fully JSON-serializable already.
    """
    credit_score = credit_explanation["credit_score"]
    fraud_score  = fraud_explanation["fraud_score"]

    if fraud_score > credit_score:
        primary_driver = "fraud"
        driver_text    = "suspicious transaction behavior"
    else:
        primary_driver = "credit"
        driver_text    = "borrower credit history"

    summary = (
        f"Trust score: {trust_score:.1f}/100 -> {alert_level}. "
        f"Primary risk factor: {driver_text}. "
        f"{credit_explanation['plain_explanation']} "
        f"{fraud_explanation['plain_explanation']}"
    )

    return {
        "trust_score":          round(trust_score, 2),
        "alert_level":          alert_level,
        "primary_driver":       primary_driver,
        "summary":              summary,
        "credit_explanation":   credit_explanation,
        "fraud_explanation":    fraud_explanation,
    }


if __name__ == "__main__":
    """
    Smoke test using dummy explanation dicts - verifies JSON-serializable structure.
    Replace with real model + real X_train once train.py wires everything together.
    """
    import json

    dummy_credit_explanation = {
        "credit_score": 0.72,
        "base_value": 0.08,
        "top_factors": [
            {"feature": "DebtRatio", "value": 0.85, "shap_value": 0.21, "direction": "increases risk", "description": "Monthly debt vs income ratio"},
            {"feature": "MonthlyIncome", "value": 0.12, "shap_value": -0.09, "direction": "decreases risk", "description": "Monthly income"},
        ],
        "plain_explanation": "Credit score: 0.72 - primarily driven by 'Monthly debt vs income ratio' (value: 0.85, increases risk).",
    }

    dummy_fraud_explanation = {
        "fraud_score": 0.81,
        "top_factors": [
            {"feature": "amount_deviation", "value": 4.21, "importance": 0.60, "description": "How unusual this transaction amount is"},
        ],
        "plain_explanation": "Fraud score: 0.81 - primarily driven by 'How unusual this transaction amount is' (value: 4.21).",
    }

    verdict = explain_verdict(dummy_credit_explanation, dummy_fraud_explanation, trust_score=23.4, alert_level="BLOCK")

    # Confirm it's actually JSON-serializable - this is what FastAPI will do internally
    print(json.dumps(verdict, indent=2))