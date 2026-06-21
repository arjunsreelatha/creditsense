import numpy as np
import csv
import time
from pathlib import Path

# Project root (two levels up from src/)
PROJECT_ROOT = Path(__file__).resolve().parents[1]



def load_csv(filepath:str) -> tuple[list[str], np.ndarray]:
    """Loads a CSV file and returns headers and data as a numpy array."""
    headers = []
    rows = []
    with open(filepath,'r') as file:
        reader = csv.reader(file)
        headers = next(reader)
        for row in reader:
            rows.append(row)
    data = []
    for row in rows:
        parsed = []
        for value in row:
            try:
                parsed.append(float(value))
            except ValueError:
                parsed.append(np.nan)
        data.append(parsed)

    return headers, np.array(data, dtype=np.float64)

def train_test_split(data: np.ndarray, test_size: float = 0.2,
                     shuffle: bool = True, random_state: int = 42) -> tuple:
    """Splits dataset into X_train, X_test, y_train, y_test."""
    if shuffle:
        np.random.seed(random_state)
        np.random.shuffle(data)

    split_idx = int(len(data) * (1 - test_size))
    train, test = data[:split_idx], data[split_idx:]

    X_train, y_train = train[:, 1:], train[:, 0]
    X_test,  y_test  = test[:, 1:],  test[:, 0]

    return X_train, X_test, y_train, y_test

    
   