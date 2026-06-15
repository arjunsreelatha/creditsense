import numpy as np
from xgboost import XGBClassifier
from fraud_features import build_fraud_features
from data_pipeline import train_test_split
from metrics import calculate_metrics,roc_auc
from sklearn.metrics import roc_auc_score

#we know IEEE-CIS fraud rate is roughly 3.5%
#scale pos weight = non-fraud count/fraud count = 27
SCALE_POS_WEIGHT=27

class FraudMOdel:
    def __init__(
            self,
            n_estimators = 500,
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
        self.feature_names=None
        self.is_trained=False
    def fit(self,X:np.ndarray,y:np.ndarray,feature_names:list = None):
        """trains the model X is the feature matrix(n,3)velocity,amount_deviation,balance_ratio
        y:fraud labels 0 or 1"""
        self.feature_names = feature_names or ["velocity","amount_deviation","balance_ratio"]
        self.model.fit(X,y)
        self.is_trained=True
        print("fraud model is trained")
    def predict(self,X:np.ndarray)->np.ndarray:
        if not self.is_trained:
            raise RuntimeError("Model not trained yet call fit() first.")
        return self.model.predict(X)

    def predict_proba(self,X:np.ndarray)-> np.ndarray:
        if not self.is_trained:
            raise RuntimeError("model is not trained yet call fit() first")
        return self.model.predict_proba(X)[:,1]
    
    def score_single(self, transaction_features: np.ndarray) -> float:
        """
        Scores a single transaction. Used by get_risk_verdict() in train.py.
 
        Args:
            transaction_features: 1D array of shape (3,)
                                  [velocity, amount_deviation, balance_ratio]
 
        Returns:
            float between 0 and 1 — fraud probability
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained yet. Call fit() first.")
        x = transaction_features.reshape(1, -1)
        return float(self.model.predict_proba(x)[0, 1])
    def get_feature_importance(self) -> dict:
        """
        Returns feature importances from XGBoost.
        Used by explainability.py to explain fraud score.

        Returns:
            dict: { feature_name: importance_score }
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained yet. Call fit() first.")
        importances = self.model.feature_importances_
        return dict(zip(self.feature_names, importances))



