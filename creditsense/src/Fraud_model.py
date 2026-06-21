import numpy as np
from pathlib import Path

# Project root (two levels up from src/)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
from xgboost import XGBClassifier
from fraud_features import build_fraud_features
from data_pipeline import train_test_split
from metrics import calculate_metrics, roc_auc
from sklearn.metrics import roc_auc_score

SCALE_POS_WEIGHT = 27


class FraudModel:
    def __init__(
        self,
        n_estimators=500,
        learning_rate=0.04,
        max_depth=4,
        random_state=42,
        scale_pos_weight=SCALE_POS_WEIGHT,
    ):
        self.model = XGBClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=random_state,
            scale_pos_weight=scale_pos_weight,
            use_label_encoder=False,
            eval_metric="auc",
        )
        self.feature_names = None
        self.is_trained = False

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: list = None):
        self.feature_names = feature_names or ["velocity", "amount_deviation", "balance_ratio"]
        self.model.fit(X, y)
        self.is_trained = True
        print("fraud model is trained")

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_trained:
            raise RuntimeError("Model not trained yet call fit() first.")
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_trained:
            raise RuntimeError("model is not trained yet call fit() first")
        return self.model.predict_proba(X)[:, 1]

    def score_single(self, transaction_features: np.ndarray) -> float:
        if not self.is_trained:
            raise RuntimeError("Model not trained yet. Call fit() first.")
        x = transaction_features.reshape(1, -1)
        return float(self.model.predict_proba(x)[0, 1])

    def get_feature_importance(self) -> dict:
        if not self.is_trained:
            raise RuntimeError("Model not trained yet. Call fit() first.")
        importances = self.model.feature_importances_
        return dict(zip(self.feature_names, importances))


if __name__ == "__main__":
    FILEPATH = str(PROJECT_ROOT / "data" / "raw" / "train_transaction.csv")

    print("Building fraud features...")
    X, y, feature_names = build_fraud_features(FILEPATH)

    data = np.column_stack((y, X))
    X_train, X_test, y_train, y_test = train_test_split(
        data, test_size=0.2, shuffle=True, random_state=42
    )

    model = FraudModel()
    model.fit(X_train, y_train, feature_names=feature_names)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    accuracy, precision, recall, f1 = calculate_metrics(y_test, y_pred)

    print(f"\n--- Fraud Model Results ---")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"My AUC:    {roc_auc(y_test, y_proba):.4f}")
    print(f"Sklearn AUC: {roc_auc_score(y_test, y_proba):.4f}")

    print(f"\n--- Feature Importance ---")
    for feat, score in model.get_feature_importance().items():
        print(f"  {feat:<25} {score:.4f}")

    print(f"\n--- Single Transaction Score ---")
    sample = X_test[0].reshape(-1)
    fraud_score = model.score_single(sample)
    print(f"  Features: velocity={sample[0]:.2f}, deviation={sample[1]:.2f}, balance_ratio={sample[2]:.4f}")
    print(f"  Fraud probability: {fraud_score:.4f}")