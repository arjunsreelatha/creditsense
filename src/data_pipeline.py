import numpy as np
import csv
import time



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

    return headers, np.array(data)

def print_csv(header: list, data:np.ndarray) -> None:
    n_rows,n_cols = data.shape
    """ Prints dataset overview: number of rows/columns and column names with types."""
    print("="*45)
    print("DATASET")
    print("="*45)
    print(f"n_rows: {n_rows}, n_cols: {n_cols}")
    print()
    print(f"{'#':<5}{'coloumn':<30}{'Type'}")
    print(f"{'-'*5}{'-'*30}{'-'*8}")

    for i, col in enumerate(header):
        print(f"{i:<5}{col:<30} float")

def print_stats(headers: list, data: np.ndarray) -> None:
    """Prints statistics for each feature: count, missing, min, max, mean, std."""
    print("=" * 75)
    print("FEATURE STATISTICS")
    print("=" * 75)
    print(f"  {'Column':<40}{'Count':<10}{'Missing':<10}{'Min':<10}{'Max':<10}{'Mean':<10}{'Std':<10}")
    print(f"  {'-'*40}{'-'*10}{'-'*10}{'-'*10}{'-'*10}{'-'*10}{'-'*10}")

    for i, col in enumerate(headers):
        col_data = data[:, i]                        
        n_missing = np.sum(np.isnan(col_data))
        count = len(col_data) - n_missing            
        n_min = np.nanmin(col_data)
        n_max = np.nanmax(col_data)
        n_mean = np.nanmean(col_data)             
        n_std = np.nanstd(col_data)

        print(f"  {col:<40}{int(count):<10}{int(n_missing):<10}{n_min:<10.2f}{n_max:<10.2f}{n_mean:<10.4f}{n_std:<10.4f}")

        if n_missing / len(col_data) > 0.05:         # ⚠ warning if >5% missing
            print(f"  ⚠  Warning: {col} has {n_missing/len(col_data)*100:.1f}% missing values")

def clean_data(headers: list, data: np.ndarray) -> tuple[list[str], np.ndarray]:
    """Cleans the dataset by dropping index, imputing missing values, and clipping outliers."""
    data = data.copy()
    headers = headers.copy()

    # STEP 1 - drop index column
    data = np.delete(data, 0, axis=1)
    headers = headers[1:]
    print("✔ Dropped index column")

    # STEP 2 - impute missing values
    for i, col in enumerate(headers):
        col_data = data[:, i]
        if np.any(np.isnan(col_data)):
            n_missing = int(np.sum(np.isnan(col_data)))
            median = np.nanmedian(col_data)
            data[:, i] = np.where(np.isnan(col_data), median, col_data)
            print(f"✔ Imputed {col:<40} ({n_missing} values)")

     # STEP 3 - clip outliers
    for i, col in enumerate(headers):
        col_data = data[:, i]
        Q1 = np.nanpercentile(col_data, 25)
        Q3 = np.nanpercentile(col_data, 75)
        IQR = Q3 - Q1
        if IQR > 0:                                    # ✅ skip binary/zero columns
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            data[:, i] = np.clip(col_data, lower, upper)
            print(f"✔ Clipped outliers in {col}")
        else:
            print(f"  Skipped clipping for {col} (IQR = 0)")
    return headers, data

def minmax_normalize(data: np.ndarray) -> np.ndarray:
    """Applies min-max normalization to scale features to [0, 1]."""
    return (data - np.nanmin(data, axis=0)) / (np.nanmax(data, axis=0) - np.nanmin(data, axis=0) + 1e-8)

def z_score(data:np.ndarray) -> np.ndarray:
    """Applies z-score normalization to standardize features to mean=0 and std=1."""
    mean = np.nanmean(data, axis=0)
    std = np.nanstd(data, axis=0)
    return (data - mean) / (std + 1e-8)

def analyze_class_imbalance(headers: list, data: np.ndarray) -> None:
    """Analyzes class imbalance by counting class distribution and calculating imbalance ratio."""
    target = data[:,0]
    class_0 = np.sum(target == 0)
    class_1 = np.sum(target == 1)
    total = len(target)
    pct_0 = class_0 / total * 100
    pct_1 = class_1 / total * 100
    ratio = class_0 / class_1 if class_1 > 0 else float('inf')
    print("="*45)
    print("CLASS IMBALANCE ANALYSIS")
    print("="*45)
    print(f"Class 0 (No Default): {class_0} ({pct_0:.1f}%)")
    print(f"Class 1 (Default)   : {class_1} ({pct_1:.1f}%)")
    print(f"Imbalance Ratio     : {ratio:.2f} (Class 0 : Class 1)")
    if ratio > 5:
        print("⚠ Warning: Significant class imbalance detected!")

def oversample_minority(data:np.ndarray) -> np.ndarray:
    """Randomly oversamples the minority class by duplicating samples until classes are balanced."""
    majority = data[data[:,0] == 0]
    minority = data[data[:,0] == 1]
    n_needed = len(majority) - len(minority)
    if n_needed > 0:
        indices = np.random.choice(len(minority), size=n_needed, replace=True)
        oversampled_minority = minority[indices]
        return np.vstack((data, oversampled_minority))
    else:
        return data
    
def smote(data: np.ndarray, k: int = 5) -> np.ndarray:
    """Applies SMOTE algorithm to generate synthetic samples for the minority class."""
    majority = data[data[:, 0] == 0]
    minority = data[data[:, 0] == 1]

    # PART 2 - distance matrix (n_minority x n_minority)
    dist = np.sqrt(((minority[:, np.newaxis] - minority) ** 2).sum(axis=2))

    # k nearest neighbours (skip index 0 = self)
    knn_indices = np.argsort(dist, axis=1)[:, 1:k+1]

    # randomly pick one neighbour per sample
    random_nn = np.array([np.random.choice(knn_indices[i]) for i in range(len(minority))])
    neighbors = minority[random_nn]

    # PART 3 - interpolate
    gap = np.random.random((len(minority), 1))
    synthetic = minority + gap * (neighbors - minority)

    # PART 4 - pick n_needed synthetics
    n_needed = len(majority) - len(minority)
    if n_needed > 0:
        indices = np.random.choice(len(synthetic), size=n_needed, replace=True)
        oversampled = synthetic[indices]
        return np.vstack((data, oversampled))
    else:
        return data

def save_csv(headers: list, data: np.ndarray, filepath: str = "C:\\Users\\Lenovo\\Desktop\\HOPE\\creditsense\\data\\cleaned\\cs-training-cleaned.csv") -> None:
    """Saves the processed dataset to a new CSV file."""
    with open(filepath, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        for row in data:
            writer.writerow(row)
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

    
            
if __name__ == "__main__":
    # 1. Load
    headers, data = load_csv(r"C:\Users\Lenovo\Desktop\HOPE\creditsense\data\raw\cs-training.csv")
    
    # 2. Inspect raw
    print_csv(headers, data)
    print_stats(headers, data)
    
    # 3. Clean
    headers, data = clean_data(headers, data)
    
    # 4. Analyse imbalance
    analyze_class_imbalance(headers, data)
    
    # 5. Normalise
    minmax_data = minmax_normalize(data)
    zscore_data = z_score(data)
    print(f"Min-max range  : {minmax_data[:, 5].min():.2f} to {minmax_data[:, 5].max():.2f}")
    print(f"Zscore range   : {zscore_data[:, 5].min():.2f} to {zscore_data[:, 5].max():.2f}")
    
    # 6. Random oversample
    balanced = oversample_minority(data)
    print(f"\n--- Random Oversampling ---")
    print(f"  Class 0 : {int(np.sum(balanced[:, 0] == 0))}")
    print(f"  Class 1 : {int(np.sum(balanced[:, 0] == 1))}")
    print(f"  Total   : {len(balanced)}")
    
    # 7. SMOTE
    print(f"\n--- SMOTE ---")
    start = time.time()
    balanced_smote = smote(data, k=5)
    end = time.time()
    print(f"  Time taken : {end - start:.2f} seconds")
    print(f"  Class 0    : {int(np.sum(balanced_smote[:, 0] == 0))}")
    print(f"  Class 1    : {int(np.sum(balanced_smote[:, 0] == 1))}")
    print(f"  Total      : {len(balanced_smote)}")
    print(f"✔ SMOTE complete")
    #8. Save cleaned data
    save_csv(headers, balanced_smote)
    print(f"✔ Cleaned dataset saved to 'data/cleaned/cs-training-cleaned.csv'")