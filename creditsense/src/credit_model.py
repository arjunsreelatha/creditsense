from xgboost import XGBClassifier
from metrics import calculate_metrics,roc_auc
import numpy as np
from data_pipeline import load_csv, train_test_split
from sklearn.metrics import roc_auc_score
from pathlib import Path

# Project root (two levels up from src/)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

def clean_data_for_xgboost(headers: list, data: np.ndarray) -> tuple[list[str], np.ndarray]:
    data = data.copy()
    headers = headers.copy()

    # 1. Drop index
    data = np.delete(data, 0, axis=1)
    headers = headers[1:]

    # 2. Handle the 96/98 "Special Codes" specifically
    # These indices refer to the delinquency columns in your dataset
    special_code_cols = [3, 7, 9] 
    for col_idx in special_code_cols:
        col_data = data[:, col_idx]
        # Replace 96/98 with NaN so XGBoost handles them as a separate category
        data[:, col_idx] = np.where((col_data == 96) | (col_data == 98), np.nan, col_data)

    # 3. DO NOT CLIP. DO NOT NORMALIZE.
    # Just return the data. XGBoost will handle the NaNs and Outliers natively.
    
    print("✔ Cleaned: Preserved outliers and masked special codes.")
    return headers, data


def feature_engineering(headers: list, data: np.ndarray) -> tuple[list, np.ndarray]:
    # 1. Define internal references to existing columns
    # Based on your previous schema: [3]=30-59days, [4]=DebtRatio, [5]=Income, [7]=90days, [9]=60-89days, [10]=Deps
    debt_ratio = data[:, 4]
    income = data[:, 5]
    dependents = data[:, 10]

    # 2. CombinedLate (Total history of friction)
    combined_late = data[:, 3] + data[:, 7] + data[:, 9]
    
    # 3. Smart Monthly Debt (The "Quirk" fix)
    monthly_debt = np.zeros(len(data))
    mask_normal = (income > 1) & (~np.isnan(income))
    monthly_debt[mask_normal] = debt_ratio[mask_normal] * income[mask_normal]
    
    mask_weird = (income <= 1) | (np.isnan(income))
    monthly_debt[mask_weird] = debt_ratio[mask_weird]

    # 4. Income Per Person (Family burden)
    # Use 1e-8 to prevent any rare division by zero errors
    income_per_person = income / (dependents + 1 + 1e-8)

    # 5. Log Transform Income (Handles the massive wealth gap in the data)
    log_income = np.log1p(income) # log1p handles 0 income safely

    # 6. Assemble
    new_features = np.column_stack((combined_late, monthly_debt, income_per_person, log_income))
    data = np.hstack((data, new_features))
    
    # Update headers so you don't lose track of columns
    new_headers = headers + ["CombinedLate", "MonthlyDebt", "IncomePerPerson", "LogIncome"]
    
    print(f"✔ Engineered 4 new features. Total features: {len(new_headers)}")
    return new_headers, data

class XGBoostClassifier:
    def __init__(
        self,
        n_estimators=500,
        learning_rate=0.04,
        max_depth=4,
        random_state=42,
        scale_pos_weight=15
    ):
        self.model = XGBClassifier(n_estimators=n_estimators,learning_rate=learning_rate,max_depth=max_depth,random_state=random_state,scale_pos_weight=scale_pos_weight)
    

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self,X_test):
        y_pred = self.model.predict(X_test)
        return y_pred
    def predict_proba(self,X_test):
        y_proab = self.model.predict_proba(X_test)[:,1]
        return y_proab
    
if __name__ ==  "__main__":
   
 
    headers, data = load_csv(str(PROJECT_ROOT / "data" / "raw" / "cs-training.csv"))
    feature_engineering(headers,data)
    headers, data = clean_data_for_xgboost(headers, data)
    X_train, X_test, y_train, y_test = train_test_split(data, test_size=0.265, shuffle=True, random_state=42)
    model = XGBoostClassifier()
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    y_proab = model.predict_proba(X_test)
    accuracy, precision, recall, f1 = calculate_metrics(y_test,y_pred) 
    print(f"--- Results ---")
   
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print("My AUC:", roc_auc(y_test, y_proab))
    print("Sklearn AUC:", roc_auc_score(y_test, y_proab))
    

   

