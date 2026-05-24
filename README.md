# CreditSense

A credit default prediction model built from scratch using Python and NumPy.
No Scikit-learn — every algorithm is implemented manually.

## Problem Statement
Predict the probability that a borrower will experience financial distress
in the next two years, using the Kaggle "Give Me Some Credit" dataset (150,000 borrowers).

## Approach
- Data pipeline: cleaning, outlier handling, normalisation, class imbalance handling
- Models: Logistic Regression + Decision Tree (both from scratch)
- Evaluation: AUC-ROC, F1, Confusion Matrix (all from scratch)
- Explainability: SHAP-style feature importance + single borrower explainer
- Deployment: Streamlit dashboard (live link coming soon)

## Repository Structure
creditsense/
├── data/
│   ├── raw/                  # Original Kaggle dataset
│   └── cleaned/              # Processed dataset
├── src/
│   ├── data_pipeline.py      # Loading, cleaning, normalisation
│   ├── eda.py                # Exploratory data analysis
│   ├── logistic_regression.py# LR from scratch
│   ├── decision_tree.py      # Decision tree from scratch
│   ├── metrics.py            # AUC-ROC, F1, confusion matrix
│   ├── explainability.py     # SHAP-style feature importance
│   └── train.py              # End-to-end training script
├── app/
│   └── streamlit_app.py      # Live dashboard
├── results/                  # Plots, reports, outputs
└── requirements.txt
## Results
*(To be updated as model is built)*

## Live Demo
*(Streamlit link coming soon)*

## Tech Stack
Python · NumPy · Matplotlib · Streamlit

## Dataset
[Give Me Some Credit — Kaggle](https://www.kaggle.com/c/GiveMeSomeCredit)