import numpy as np
from sklearn.metrics import roc_auc_score


def calculate_metrics(y_test, y_pred):
    tp = np.count_nonzero((y_pred == 1) & (y_test == 1))
    tn = np.count_nonzero((y_pred == 0) & (y_test == 0))
    fp = np.count_nonzero((y_pred == 1) & (y_test == 0))
    fn = np.count_nonzero((y_pred == 0) & (y_test == 1))

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)

    return accuracy, precision, recall, f1


def roc_auc(y_true, y_prob):
    """Mann Whitney U test"""
    order = np.argsort(y_prob)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(len(y_prob)) + 1

    pos = y_true == 1
    n_pos = np.sum(pos)
    n_neg = len(y_true) - n_pos

    rank_sum = np.sum(ranks[pos])

    return (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)



# Your implementation
def roc_auc_custom(y_true, y_prob):
    idx = np.argsort(-y_prob)
    y_true = y_true[idx]

    tp = np.cumsum(y_true == 1)
    fp = np.cumsum(y_true == 0)

    P = tp[-1]
    N = fp[-1]

    tpr = np.concatenate(([0.0], tp / P))
    fpr = np.concatenate(([0.0], fp / N))

    return np.trapz(tpr, fpr)
if __name__ == "__main__":

# Test many random datasets
    

    np.random.seed(42)

    for i in range(1000):
        n = 1000

        y_true = np.random.randint(0, 2, size=n)
        y_prob = np.random.rand(n)

        if len(np.unique(y_true)) < 2:
            continue

        custom = roc_auc_custom(y_true, y_prob)
        sklearn_auc = roc_auc_score(y_true, y_prob)
        mann_whitney = roc_auc(y_true, y_prob)

        if abs(custom - sklearn_auc) > 1e-10:
            print("Mismatch with sklearn!")
            print(custom, sklearn_auc)
            break

        if abs(custom - mann_whitney) > 1e-10:
            print("Mismatch with Mann-Whitney!")
            print(custom, mann_whitney)
            break
    else:
        print("All 1000 tests passed!")