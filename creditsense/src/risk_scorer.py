import numpy as np

CREDIT_WEIGHT = 0.4
FRAUD_WEIGHT = 0.6
def normalize_to_100(score:float) -> float:
    score = float(score)
    score = max(0.0,min(1.0,score))
    return round(score*100,2)

def compute_trust_score(credit_score:float,fraud_score:float)->dict:
    credit_safety = 1.0 - credit_score
    fraud_safety = 1.0 - fraud_score

    raw_trust = (CREDIT_WEIGHT*credit_safety)+(FRAUD_WEIGHT*fraud_safety)

    return {
        "credit_score_raw": round(credit_score, 4),
        "fraud_score_raw": round(fraud_score, 4),
        "credit_risk": normalize_to_100(credit_safety),
        "fraud_risk": normalize_to_100(fraud_safety),
        "trust_score": normalize_to_100(raw_trust),
    }

def batch_trust_scores(credit_scores: np.ndarray, fraud_scores: np.ndarray) -> np.ndarray:
    """
    Vectorized version for scoring entire arrays at once.
    Used during training/evaluation, not live dashboard.
 
    Args:
        credit_scores : np.ndarray of shape (n,) — credit default probabilities
        fraud_scores  : np.ndarray of shape (n,) — fraud probabilities
 
    Returns:
        np.ndarray of shape (n,) — trust scores scaled 0–100
    """
    credit_scores = np.clip(credit_scores, 0.0, 1.0)
    fraud_scores = np.clip(fraud_scores, 0.0, 1.0)
 
    credit_safety = 1.0 - credit_scores
    fraud_safety = 1.0 - fraud_scores
 
    raw_trust = (CREDIT_WEIGHT * credit_safety) + (FRAUD_WEIGHT * fraud_safety)
 
    return np.round(raw_trust * 100, 2)



