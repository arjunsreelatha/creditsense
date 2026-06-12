import csv
import numpy as np
import pandas as pd

# ── IEEE-CIS columns we actually need ────────────────────────────────────────
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

"""other velocity implementations I tried (all too slow):"""

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

    # Sort by card + time
    df = df.sort_values(["card1", "dt"]).reset_index(drop=True)

    # Convert dt to datetime-like for rolling (treat seconds as milliseconds — same relative window)
    df["dt_td"] = pd.to_timedelta(df["dt"], unit="s")

    # Rolling count within window per card group — fully vectorized
    df["velocity"] = (
        df.groupby("card1", group_keys=False)
        .apply(lambda g: g.rolling(f"{VELOCITY_WINDOW}s", on="dt_td")["dt"].count() - 1)
        .fillna(0)
        .astype(int)
    )

    # Map back to original order
    result = np.zeros(len(data))
    result[df["orig_idx"].values] = df["velocity"].values

    print(f"✔ Velocity computed. Mean: {result.mean():.2f}, Max: {result.max():.0f}")
    return result


