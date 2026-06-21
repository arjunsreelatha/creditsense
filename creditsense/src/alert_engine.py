import numpy as np

# ── Alert thresholds ──────────────────────────────────────────────────────────
# Trust score is 0–100. Higher = safer.
# Change these constants to tune alert sensitivity without touching any logic.
BLOCK_THRESHOLD  = 40   # trust score below this → BLOCK
REVIEW_THRESHOLD = 70   # trust score below this → REVIEW (but above BLOCK)
                        # trust score above REVIEW_THRESHOLD → ALLOW
# ─────────────────────────────────────────────────────────────────────────────

# Alert level constants — use these everywhere instead of raw strings
ALERT_BLOCK  = "BLOCK"
ALERT_REVIEW = "REVIEW"
ALERT_ALLOW  = "ALLOW"


def get_alert_level(trust_score: float) -> str:
    """
    Classifies a single trust score into an alert level.

    Args:
        trust_score : float between 0 and 100

    Returns:
        "BLOCK"  — deny transaction, high risk
        "REVIEW" — flag for manual review
        "ALLOW"  — low risk, allow transaction
    """
    if trust_score < BLOCK_THRESHOLD:
        return ALERT_BLOCK
    elif trust_score < REVIEW_THRESHOLD:
        return ALERT_REVIEW
    else:
        return ALERT_ALLOW


def get_alert_metadata(trust_score: float) -> dict:
    """
    Returns full alert details for a single transaction.
    This is what get_risk_verdict() in train.py returns to the dashboard.

    Args:
        trust_score : float between 0 and 100

    Returns:
        dict with:
            trust_score  : the input score
            alert_level  : BLOCK / REVIEW / ALLOW
            message      : human readable explanation
            color        : UI color hint for Streamlit (red / orange / green)
            action       : what the bank should do
    """
    level = get_alert_level(trust_score)

    metadata = {
        ALERT_BLOCK: {
            "message": "High risk detected. Transaction blocked automatically.",
            "color": "red",
            "action": "Deny transaction. Freeze account for review.",
        },
        ALERT_REVIEW: {
            "message": "Suspicious activity detected. Flagged for manual review.",
            "color": "orange",
            "action": "Hold transaction. Analyst review required before processing.",
        },
        ALERT_ALLOW: {
            "message": "Transaction appears safe. No anomalies detected.",
            "color": "green",
            "action": "Allow transaction to proceed.",
        },
    }

    return {
        "trust_score": round(trust_score, 2),
        "alert_level": level,
        "message": metadata[level]["message"],
        "color": metadata[level]["color"],
        "action": metadata[level]["action"],
    }


def batch_alert_levels(trust_scores: np.ndarray) -> np.ndarray:
    """
    Vectorized alert classification for entire arrays.
    Used during evaluation to analyze threshold distribution.

    Args:
        trust_scores : np.ndarray of shape (n,)

    Returns:
        np.ndarray of shape (n,) — string array of alert levels
    """
    levels = np.where(
        trust_scores < BLOCK_THRESHOLD, ALERT_BLOCK,
        np.where(trust_scores < REVIEW_THRESHOLD, ALERT_REVIEW, ALERT_ALLOW)
    )
    return levels


def alert_distribution(trust_scores: np.ndarray) -> dict:
    """
    Summarizes how many transactions fall into each alert level.
    Useful for evaluating whether thresholds need tuning.

    Args:
        trust_scores : np.ndarray of shape (n,)

    Returns:
        dict with counts and percentages for each alert level
    """
    levels = batch_alert_levels(trust_scores)
    n = len(levels)

    block_count  = int(np.sum(levels == ALERT_BLOCK))
    review_count = int(np.sum(levels == ALERT_REVIEW))
    allow_count  = int(np.sum(levels == ALERT_ALLOW))

    return {
        "total": n,
        "BLOCK":  {"count": block_count,  "pct": round(block_count  / n * 100, 2)},
        "REVIEW": {"count": review_count, "pct": round(review_count / n * 100, 2)},
        "ALLOW":  {"count": allow_count,  "pct": round(allow_count  / n * 100, 2)},
    }


if __name__ == "__main__":
    # ── Single transaction tests ──────────────────────────────────────────────
    test_scores = [15.0, 38.5, 45.0, 69.9, 72.0, 95.5]

    print("--- Single Transaction Alerts ---")
    print(f"{'Trust Score':>12} {'Alert':>8} {'Action'}")
    print("-" * 70)
    for score in test_scores:
        result = get_alert_metadata(score)
        print(f"{result['trust_score']:>12} {result['alert_level']:>8}  {result['action']}")

    # ── Batch test ────────────────────────────────────────────────────────────
    print("\n--- Batch Distribution Test ---")
    np.random.seed(42)
    simulated_scores = np.random.uniform(0, 100, 1000)
    dist = alert_distribution(simulated_scores)

    print(f"Total transactions: {dist['total']}")
    for level in [ALERT_BLOCK, ALERT_REVIEW, ALERT_ALLOW]:
        print(f"  {level:<8}: {dist[level]['count']:>4} ({dist[level]['pct']}%)")