# CreditSense

**Dual-layer fraud detection and credit risk scoring system for digital banking.**

CreditSense addresses a core gap in digital banking security: static, one-time identity checks cannot detect account takeover, behavioral drift, or suspicious transaction patterns that emerge over time. CreditSense scores every session in real time using two independent ML layers — credit history and live transaction behavior — combined into a single identity trust score.

Built for the **IIT Gandhinagar Hackathon 2026**.

---

## How It Works

```
Give Me Some Credit dataset          IEEE-CIS / Vesta Corporation dataset
         ↓                                        ↓
  Credit pipeline                        Fraud pipeline
  (XGBoost, feature engineering)         (velocity, amount deviation, balance ratio)
         ↓                                        ↓
         └──────────── risk_scorer.py ────────────┘
                              ↓
                    40% credit + 60% fraud
                              ↓
                       Trust score (0–100)
                              ↓
                  alert_engine.py → BLOCK / REVIEW / ALLOW
                              ↓
                    SHAP explainability (both layers)
                              ↓
                    FastAPI /predict endpoint
                              ↓
                       HTML/React frontend
```

---

## Alert Levels

| Trust Score | Decision |
|---|---|
| 70 – 100 | ✅ ALLOW — low risk |
| 40 – 69  | ⚠️ REVIEW — flag for manual check |
| 0 – 39   | 🚫 BLOCK — deny transaction |

---

## ML Pipeline

### Credit Scoring Layer
- **Dataset:** Give Me Some Credit (Kaggle) — 150,000 real borrower records
- **Model:** XGBoost (`scale_pos_weight=15` for class imbalance)
- **Feature engineering:** CombinedLate, MonthlyDebt, IncomePerPerson, LogIncome
- **Results:** Accuracy 0.7959 | Precision 0.2208 | Recall 0.7817 | F1 0.3444

### Fraud Detection Layer
- **Dataset:** IEEE-CIS Fraud Detection (Kaggle / Vesta Corporation) — 590,540 real transactions, 3.5% fraud rate
- **Model:** XGBoost (`scale_pos_weight=27` for 3.5% fraud rate)
- **Features:** Transaction velocity (1-hour rolling window), amount z-score deviation, address-link balance ratio
- **Results:** Accuracy 0.7495 | Precision 0.0693 | Recall 0.4965 | AUC 0.6857

### Risk Scorer
Combines both layers: `trust_score = 0.4 × credit_safety + 0.6 × fraud_safety`

Fraud weighted higher because it reflects live session behavior vs historical credit profile.

### Explainability
Real SHAP values for credit layer. XGBoost feature importances for fraud layer. Both layers explained separately — analyst can see which layer triggered the alert and why.

---

## Tech Stack

| Layer | Technology |
|---|---|
| ML | Python, XGBoost, NumPy, pandas, scikit-learn, SHAP |
| Backend | FastAPI, uvicorn, joblib |
| Frontend | HTML5, CSS3, JavaScript (prototype) |
| Datasets | Give Me Some Credit, IEEE-CIS / Vesta Corporation |
| Deployment | Render/Railway (API), Vercel (frontend) |
| Version control | GitHub (4 branches: main, ml-engine, fastapi-backend, frontend-ui) |

---

## Project Structure

```
creditsense/
├── data/
│   └── raw/
│       ├── cs-training.csv              # Give Me Some Credit
│       └── train_transaction.csv        # IEEE-CIS
│
├── models/                              # Saved joblib models (auto-generated)
│   ├── credit_model.joblib
│   ├── fraud_model.joblib
│   └── credit_xtrain.joblib
│
├── src/
│   ├── data_pipeline.py                 # CSV loading, train/test split
│   ├── metrics.py                       # AUC-ROC, F1, accuracy
│   ├── fraud_features.py                # Velocity, deviation, balance ratio
│   ├── fraud_model.py                   # FraudModel (XGBoost wrapper)
│   ├── risk_scorer.py                   # Trust score combiner
│   ├── alert_engine.py                  # Threshold-based alert classification
│   ├── explainability.py                # SHAP credit + fraud feature importance
│   ├── train.py                         # Training orchestration + get_risk_verdict()
│   └── main.py                          # FastAPI app — /predict endpoint
│
└── frontend/
    └── index.html                       # Analyst-facing test dashboard
```

---

## API

### `POST /predict`

**Request:**
```json
{
  "borrower_profile": {
    "RevolvingUtilizationOfUnsecuredLines": 0.85,
    "age": 35,
    "DebtRatio": 0.6,
    "MonthlyIncome": 3000,
    "NumberOfDependents": 2,
    "CombinedLate": 1,
    "MonthlyDebt": 1800,
    "IncomePerPerson": 1000,
    "LogIncome": 8.0
  },
  "transaction": {
    "velocity": 5,
    "amount_deviation": 3.2,
    "balance_ratio": 0.08
  }
}
```

**Response:**
```json
{
  "trust_score": 23.4,
  "alert_level": "BLOCK",
  "message": "High risk detected. Transaction blocked automatically.",
  "action": "Deny transaction. Freeze account for review.",
  "color": "red",
  "summary": "Trust score: 23.4/100 → BLOCK. Primary risk factor: suspicious transaction behavior.",
  "credit_explanation": { ... },
  "fraud_explanation": { ... }
}
```

---

## Running Locally

### 1. Install dependencies
```bash
pip install xgboost fastapi uvicorn joblib pandas numpy scikit-learn shap
```

### 2. Download datasets
```bash
kaggle competitions download -c GiveMeSomeCredit -p data/raw
kaggle competitions download -c ieee-fraud-detection -f train_transaction.csv -p data/raw
```

### 3. Start the API
```bash
cd src
python main.py
```
Models train automatically on first run and save to `models/`. Subsequent starts load from disk instantly.

### 4. Open the frontend
Open `frontend/index.html` in your browser. Click **"Fill sample data"** to test instantly.

API runs on `http://localhost:8000`. Interactive docs at `http://localhost:8000/docs`.

---

## Known Limitations (v1)

- Fraud model AUC 0.6857 — limited by only 3 input features. v2 will add `D15`, `addr1`, `card2`, `card4`, `dist1` for stronger signal (target AUC ~0.75–0.80)
- Frontend is a prototype HTML file — full React frontend in progress on `feature/frontend-ui`
- Models retrain from scratch if `.joblib` files are deleted

---

## Team

| Name | Role |
|---|---|
| Arjun Sreelatha | ML pipeline, FastAPI backend, prototype frontend |
| Agnivesh Thotumkara | React frontend (in progress) |
| Ayush | React frontend (in progress) |
| Anil | React frontend (in progress) |

B.Tech Computer Science, IIIT Vadodara

---

## License

Developed for educational and research purposes. IEEE-CIS dataset usage subject to Kaggle competition terms.
