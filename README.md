# CreditSense – AI-Powered Credit Risk Assessment

## Overview

CreditSense is a machine learning-powered credit risk assessment platform that predicts the probability of loan default and helps financial institutions make smarter lending decisions.

The platform consists of two web applications:

- User Portal – Allows customers to enter financial information and receive a credit risk assessment.
- Analyst Portal – Provides analysts with detailed risk insights, portfolio monitoring, and model performance analytics.

CreditSense combines predictive analytics with explainable AI to improve transparency, reduce financial risk, and support data-driven decision-making.

---

## Features

### User Portal

- Submit financial and credit-related information
- Real-time credit default prediction
- Risk classification (Low, Medium, High)
- Probability-based risk score
- Simple and intuitive interface
- Personalized prediction explanations

### Analyst Portal

- Applicant risk monitoring
- Portfolio risk analysis
- Feature importance visualization
- Model performance dashboard
- Historical prediction tracking
- Data analytics and reporting

---

## Machine Learning Pipeline

### Data Preprocessing

- Missing value handling
- Feature scaling
- Outlier treatment
- Feature engineering
- Class imbalance handling using oversampling

### Models Evaluated

- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost

### Final Model

**XGBoost with Oversampling**

Performance:

| Metric | Score |
|----------|--------|
| F1 Score | 0.80 |
| ROC-AUC | 0.89 |

---

## Tech Stack

### Frontend
- React.js
- Tailwind CSS
- Chart.js

### Backend
- FastAPI / Flask
- Python

### Machine Learning
- Scikit-learn
- XGBoost
- Pandas
- NumPy

### Visualization
- SHAP
- Matplotlib

### Database
- PostgreSQL / MongoDB

---

## System Architecture

User Portal
    ↓
Prediction API
    ↓
XGBoost Model
    ├── Risk Prediction
    └── SHAP Explanation
    ↓
Analyst Portal
    ↓
Analytics & Monitoring Dashboard

---

## Project Structure

CreditSense/

├── backend/
│   ├── api/
│   ├── models/
│   ├── preprocessing/
│   └── services/
│
├── frontend-user/
│   ├── src/
│   └── public/
│
├── frontend-analyst/
│   ├── src/
│   └── public/
│
├── datasets/
├── notebooks/
├── trained_models/
├── requirements.txt
└── README.md

---

## Installation

### Clone the Repository

git clone https://github.com/your-username/CreditSense.git

cd CreditSense

### Create Virtual Environment

python -m venv venv

### Activate Environment

Windows:
venv\Scripts\activate

Linux/Mac:
source venv/bin/activate

### Install Dependencies

pip install -r requirements.txt

---

## Running the Application

### Backend

uvicorn main:app --reload

### User Frontend

cd frontend-user
npm install
npm run dev

### Analyst Frontend

cd frontend-analyst
npm install
npm run dev

---

## Explainable AI

CreditSense uses SHAP (SHapley Additive Explanations) to provide transparency in model predictions.

Benefits:
- Understand feature contributions
- Increase trust in predictions
- Support regulatory compliance
- Improve decision-making

---

## Future Enhancements

- Real-time loan approval recommendations
- Fraud detection integration
- Automated report generation
- Alternative credit scoring methods
- Model drift monitoring
- Cloud deployment with Docker and Kubernetes

---

## Contributors

Arjun S

B.Tech Computer Science

---

## License

This project is developed for educational and research purposes.
