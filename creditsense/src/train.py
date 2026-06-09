from logistic_regression import LogisticRegression
import numpy as np
from data_pipeline import load_csv, train_test_split

def train_logistic_regression(filepath: str) -> None:
    headers, data = load_csv(filepath)
    X_train, X_test, y_train, y_test = train_test_split(data, test_size=0.2, 
                                                         shuffle=True, random_state=42)

    # normalise — fit on train only, apply same stats to test
    mean = X_train.mean(axis=0)
    std  = X_train.std(axis=0)
    X_train = (X_train - mean) / (std + 1e-8)
    X_test  = (X_test  - mean) / (std + 1e-8)  # ← you were missing this

    # verify normalisation worked
    print("After normalisation X_train sample:", X_train[:2, :3])

    # train
    model = LogisticRegression(learning_rate=0.01, n_iterations=1000)
    model.fit(X_train, y_train)  # ← you were missing this

    # evaluate
    y_pred = model.predict(X_test)
    accuracy = np.mean(y_pred == y_test)
        # check what model is actually predicting
    y_pred_proba = model.predict_proba(X_test)
    print("Min proba:", y_pred_proba.min())
    print("Max proba:", y_pred_proba.max())
    print("Mean proba:", y_pred_proba.mean())
    print("Predicted class 0:", np.sum(y_pred == 0))
    print("Predicted class 1:", np.sum(y_pred == 1))
    print("Actual class 0:", np.sum(y_test == 0))
    print("Actual class 1:", np.sum(y_test == 1))

    print("=" * 35)
    print("  TRAINING RESULTS")
    print("=" * 35)
    print(f"  Train size   : {len(X_train)}")
    print(f"  Test size    : {len(X_test)}")
    print(f"  Final loss   : {model.loss_history[-1]:.4f}")
    print(f"  Accuracy     : {accuracy:.4f}")
    print("=" * 35)

if __name__ == "__main__":
    train_logistic_regression(r"C:\Users\Lenovo\Desktop\HOPE\creditsense\data\cleaned\cs-training-cleaned.csv")