from xgboost import XGBClassifier
from metrics import calculate_metrics,roc_auc
import numpy as np
from data_pipeline import load_csv, train_test_split
from sklearn.metrics import roc_auc_score


class XGBoostClassifier:
    def __init__(
        self,
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        random_state=42
    ):
        self.model = XGBClassifier(n_estimators=n_estimators,learning_rate=learning_rate,max_depth=max_depth,random_state=random_state)
    

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self,X_test):
        y_pred = self.model.predict(X_test)
        return y_pred
    def predict_proba(self,X_test):
        y_proab = self.model.predict_proba(X_test)[:,1]
        return y_proab
    
if __name__ ==  "__main__":
   
 
    headers, data = load_csv(r"C:\Users\Lenovo\Desktop\HOPE\creditsense\data\cleaned\cs-training-cleaned.csv")
    X_train, X_test, y_train, y_test = train_test_split(data, test_size=0.2, shuffle=True, random_state=42)
    model = XGBoostClassifier()
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    y_proab = model.predict_proba(X_test)
    print("calculate metrices", calculate_metrics(y_test,y_pred) )
    print("My AUC:", roc_auc(y_test, y_proab))
    print("Sklearn AUC:", roc_auc_score(y_test, y_proab))
    

   

