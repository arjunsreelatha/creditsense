import csv
import numpy as np
import pandas as pd
from pathlib import Path


# Project root (two levels up from src/)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# columns we need from the IEEE-CIS dataset
USECOLS = ["TransactionID", "isFraud", "TransactionDT", "TransactionAmt", "card1", "C1", "D1"]

# Velocity window: 1 hour in seconds
VELOCITY_WINDOW = 3600


def load_ieee_cis(filepath: str) -> tuple[list[str], np.ndarray]:
    """
    Loads only the required columns from IEEE-CIS train_transaction.csv.
    Memory-safe: skips all 400+ irrelevant columns.
    Uses pandas for fast loading.
    """
    df = pd.read_csv(filepath, usecols=USECOLS)
    headers = list(df.columns)
    data = df.to_numpy(dtype=np.float64)
    print(f"✔ Loaded IEEE-CIS: {data.shape[0]} transactions, {data.shape[1]} columns.")
    return headers, data


def compute_velocity(data: np.ndarray, headers: list) -> np.ndarray:
    """
    Velocity: number of transactions by same card within last VELOCITY_WINDOW seconds.

    Fully vectorized — no Python loops.
    Uses pandas rolling count on time-sorted groups.
    """
    card1_idx = headers.index("card1")
    dt_idx    = headers.index("TransactionDT")

    df = pd.DataFrame({
        "card1":    data[:, card1_idx],
        "dt":       data[:, dt_idx],
        "orig_idx": np.arange(len(data))
    })

    df = df.sort_values(["card1", "dt"]).reset_index(drop=True)
    df["dt_td"] = pd.to_timedelta(df["dt"], unit="s")

    df["velocity"] = (
        df.groupby("card1", group_keys=False)
        .apply(lambda g: g.rolling(f"{VELOCITY_WINDOW}s", on="dt_td")["dt"].count() - 1)
        .fillna(0)
        .astype(int)
    )

    result = np.zeros(len(data))
    result[df["orig_idx"].values] = df["velocity"].values

    print(f"✔ Velocity computed. Mean: {result.mean():.2f}, Max: {result.max():.0f}")
    return result


def compute_amount_deviation(data: np.ndarray, headers: list) -> np.ndarray:
    """
    Amount deviation: z-score of transaction amount vs card's historical average.
    Higher deviation = more suspicious. Fully vectorized via pandas groupby.
    """
    card1_idx = headers.index("card1")
    amt_idx   = headers.index("TransactionAmt")

    df = pd.DataFrame({
        "card1":  data[:, card1_idx],
        "amount": data[:, amt_idx]
    })

    group_mean = df.groupby("card1")["amount"].transform("mean")
    group_std  = df.groupby("card1")["amount"].transform("std").fillna(0)

    deviation = (df["amount"] - group_mean) / (group_std + 1e-8)

    print(f"✔ Amount deviation computed. Mean: {deviation.mean():.2f}, Std: {deviation.std():.2f}")
    return deviation.to_numpy()


def compute_balance_ratio(data: np.ndarray, headers: list) -> np.ndarray:
    """
    Balance ratio: C1 (addresses linked to card) normalized to 0–1.
    High C1 = card linked to many addresses = suspicious.
    """
    c1_idx = headers.index("C1")
    c1     = np.where(np.isnan(data[:, c1_idx]), 0, data[:, c1_idx])

    max_c1 = np.nanmax(c1)
    if max_c1 == 0:
        return np.zeros(len(data))

    balance_ratio = c1 / max_c1

    print(f"✔ Balance ratio computed. Mean: {balance_ratio.mean():.4f}, Max: {balance_ratio.max():.4f}")
    return balance_ratio


def build_fraud_features(filepath: str) -> tuple[np.ndarray, np.ndarray, list]:
    """
    Master function. Loads IEEE-CIS, computes all three fraud features.

    Returns:
        X            : np.ndarray (n, 3) — [velocity, amount_deviation, balance_ratio]
        y            : np.ndarray (n,)  — isFraud labels
        feature_names: list of strings
    """
    headers, data = load_ieee_cis(filepath)

    velocity         = compute_velocity(data, headers)
    amount_deviation = compute_amount_deviation(data, headers)
    balance_ratio    = compute_balance_ratio(data, headers)

    X = np.column_stack((velocity, amount_deviation, balance_ratio))
    y = data[:, headers.index("isFraud")]

    feature_names = ["velocity", "amount_deviation", "balance_ratio"]

    print(f"\n✔ Fraud feature matrix built: {X.shape}")
    print(f"  Fraud rate: {y.mean() * 100:.2f}%")

    return X, y, feature_names


if __name__ == "__main__":
    FILEPATH = str(PROJECT_ROOT / "data" / "raw" / "train_transaction.csv")

    X, y, feature_names = build_fraud_features(FILEPATH)

    print("\n--- Sample (first 5 rows) ---")
    print(f"{'velocity':>15} {'amount_dev':>15} {'balance_ratio':>15} {'label':>8}")
    for i in range(min(5, len(X))):
        print(f"{X[i,0]:>15.4f} {X[i,1]:>15.4f} {X[i,2]:>15.4f} {int(y[i]):>8}")