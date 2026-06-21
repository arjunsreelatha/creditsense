# EDA Findings for Credit Default Dataset

## Overview
This exploratory data analysis suggests that the strongest signals for default risk come from credit utilization and past-due behavior, while several other variables show weaker class separation. The attached plots indicate both predictive features and redundancy among some delinquency variables.

## Main patterns
`RevolvingUtilizationOfUnsecuredLines` appears to be the strongest individual predictor in the visual analysis, with the default class shifted toward much higher values than the no-default class. The top-features chart also shows it has the largest absolute correlation with the target, at approximately +0.52.

Age also shows meaningful separation between classes. The no-default group is centered at older ages, while the default group appears younger on average, which matches the negative correlation shown in the top-features plot.

The past-due variables — `NumberOfTime30-59DaysPastDueNotWorse`, `NumberOfTime60-89DaysPastDueNotWorse`, and `NumberOfTimes90DaysLate` — also stand out as useful predictors. Their distributions show that the default class contains more extreme late-payment values, and the feature-ranking plot places two of them among the top five features.

## Correlation insights
The correlation heatmap shows that the three delinquency variables are extremely highly correlated with one another, with values close to 0.99 or 1.00. This means they likely carry overlapping information and may introduce redundancy if all are used together in a model.

The heatmap also shows moderate relationships among a few other features. For example, `NumberOfOpenCreditLinesAndLoans` and `NumberRealEstateLoansOrLines` have a visible positive correlation, and age has a moderate negative correlation with `RevolvingUtilizationOfUnsecuredLines`.

## Weaker signals
`MonthlyIncome` and `DebtRatio` do show some class differences, but the overlap between the two classes is still substantial. That suggests they may help the model, but they are likely weaker standalone predictors than utilization and delinquency history.

`NumberOfDependents` appears relatively noisy and does not visually separate the classes well. Its distribution suggests limited predictive power on its own compared with the stronger variables above.

## Data quality warnings
Several plots show strong spikes at special-looking values such as 0 and 99 in delinquency-related features. That pattern often indicates outliers, placeholder values, or coded missing values, so these columns should be checked carefully before model training.

Some continuous features, especially utilization, debt ratio, and income, also appear skewed rather than normally distributed. This means transformations, clipping, or robust preprocessing may improve downstream modeling.

## Modeling implications
A first baseline model should likely prioritize `RevolvingUtilizationOfUnsecuredLines`, age, and the delinquency history variables, because those show the clearest separation and strongest target correlation in the plots.

At the same time, the nearly duplicate delinquency variables should be handled carefully. Keeping all of them may not add much new information, so feature selection, regularization, or model-based importance checks would be useful next steps.

## Class imbalance findings
The raw dataset contains approximately 93% no-default and 7% default cases — a significant imbalance. Three approaches were tested during training to understand the impact:

**Original imbalanced data** achieved 93.3% accuracy but predicted only 52 defaults out of 2,020 actual defaults in the test set. This result is misleading — the model learned to predict the majority class almost exclusively and is not useful for identifying real defaulters.

**Random oversampling** and **SMOTE** both produced balanced predictions closer to the true class distribution, with accuracy around 73–74% on the balanced test set. While this number looks lower, the model was actually attempting to identify defaulters rather than ignoring them.

This confirms that accuracy is not a reliable metric for imbalanced credit risk data. AUC-ROC and F1 score will be used as the primary evaluation metrics going forward, as they properly account for performance across both classes.

## Logistic regression baseline
After z-score normalisation and SMOTE balancing, logistic regression achieved a final binary cross-entropy loss of 0.527 and accuracy of 74.9% on the held-out test set. Probability scores ranged from 0.07 to 0.99, confirming the model is using the full prediction range rather than collapsing to one class.

The relatively modest accuracy is expected for a linear model on this dataset. The relationship between features and default risk is non-linear — particularly for delinquency counts and utilization — which a logistic regression cannot fully capture with a straight decision boundary. The decision tree built in Week 8 is expected to improve on this baseline by handling non-linear splits directly.

## Model selection note
The three late-payment features appear highly correlated and may cause multicollinearity. They will be evaluated for possible removal during model training based on their impact on AUC-ROC score. The model with the higher AUC-ROC after this evaluation will be selected as the better classifier.

Implemented Accuracy, Precision, Recall, F1-Score, and ROC-AUC from scratch using NumPy. ROC-AUC implementation was validated against scikit-learn and produced matching results within numerical precision.
error = 5.8e-08


# Fraud Model — v1 Results & v2 Plan

## v1 Results (current)
- Features: velocity, amount_deviation, balance_ratio (3 only)
- Accuracy: 0.7495
- Precision: 0.0693  ← weak
- Recall: 0.4965
- AUC: 0.6857  ← decent but not strong (0.85+ is good for fraud)
- balance_ratio alone carries 64% of feature importance — model is leaning on one weak signal

## Why v1 is weak
Three features aren't enough signal. Not a model problem — XGBoost is fine.
It's a feature problem.

## v2 Plan — add these columns to fraud_features.py
| Column | What it is | Why it helps |
|---|---|---|
| D15 | days since last transaction (diff from D1) | Strong known fraud signal |
| addr1 | billing address | Address mismatch = suspicious |
| card2, card4 | card type/category | Different card types = different fraud rates |
| dist1 | distance between billing/shipping | Large distance = classic fraud signal |

Expected result: AUC 0.68 → ~0.75-0.80+

## Decision
- v1 is enough for June 20 idea submission (idea stage, not model stage)
- Improve to v2 in Phase 2/3 (after shortlist, before prototype deadline July 26)
- Don't touch model type — XGBoost stays. Only add features.
