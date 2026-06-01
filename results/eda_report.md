# EDA Findings for Credit Default Dataset

## Overview
This exploratory data analysis suggests that the strongest signals for default risk come from credit utilization and past-due behavior, while several other variables show weaker class separation. The attached plots indicate both predictive features and redundancy among some delinquency variables.

## Main patterns
`RevolvingUtilizationOfUnsecuredLines` appears to be the strongest individual predictor in the visual analysis, with the default class shifted toward much higher values than the no-default class. The top-features chart also shows it has the largest absolute correlation with the target [1].

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

At the same time, the nearly duplicate delinquency variables should be handled carefully. Keeping all of them may not add much new information, so feature selection, regularization, or model-based importance checks would be useful next steps [6].

## Model selection note
The three late-payment features appear highly correlated, so they may cause multicollinearity and will be evaluated for possible removal during model training.

After comparing the AUC-ROC scores, the model with the higher score was selected as the better classifier.