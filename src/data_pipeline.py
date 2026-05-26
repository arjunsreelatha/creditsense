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
             

if __name__ == "__main__":
    filepath = "data/raw/give_me_some_credit.csv"
    
    headers, data = load_csv(r"C:\Users\Lenovo\Desktop\HOPE\creditsense\data\raw\cs-training.csv")
    clean_headers, clean_data = clean_data(headers, data)
    print_csv(clean_headers, clean_data)
    print()
    print_stats(clean_headers, clean_data)


    