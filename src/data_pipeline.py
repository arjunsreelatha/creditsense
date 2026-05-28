import numpy as np
import csv
import os

def load_csv(filepath:str) -> tuple[list[str], np.ndarray]:
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
    return (data - np.nanmin(data, axis=0)) / (np.nanmax(data, axis=0) - np.nanmin(data, axis=0) + 1e-8)

def zscore(data:np.ndarray) -> np.ndarray:
    mean = np.nanmean(data, axis=0)
    std = np.nanstd(data, axis=0)
    return (data - mean) / (std + 1e-8)

def analyze_class_imbalance(headers: list, data: np.ndarray) -> None:
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
    majortiy = data[data[:,0] == 0]
    minority = data[data[:,0] == 1]
    n_needed = len(majortiy) - len(minority)
    if n_needed > 0:
        indices = np.random.choice(len(minority), size=n_needed, replace=True)
        oversampled_minority = minority[indices]
        return np.vstack((data, oversampled_minority))
    else:
        return data
    



if __name__ == "__main__":
    headers, data = load_csv(r"C:\Users\Lenovo\Desktop\HOPE\creditsense\data\raw\cs-training.csv")
    headers, data = clean_data(headers, data)
    analyze_class_imbalance(headers, data)      # ✅ added

    print(f"Original MonthlyIncome range : {data[:, 5].min():.2f} to {data[:, 5].max():.2f}")
    minmax_data = minmax_normalize(data)
    print(f"After min-max               : {minmax_data[:, 5].min():.2f} to {minmax_data[:, 5].max():.2f}")
    zscore_data = zscore(data)
    print(f"After zscore                : {zscore_data[:, 5].min():.2f} to {zscore_data[:, 5].max():.2f}")

    balanced = oversample_minority(data)
    print(f"\nAfter oversampling:")           # ✅ indented
    print(f"  Class 0 : {int(np.sum(balanced[:, 0] == 0))}")
    print(f"  Class 1 : {int(np.sum(balanced[:, 0] == 1))}")
    print(f"  Total   : {len(balanced)}")