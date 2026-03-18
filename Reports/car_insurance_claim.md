# Car Insurance Claim Prediction — Project Report

---

## Table of Contents

1. [Project Introduction & Objective](#1-project-introduction--objective)
2. [Data Overview](#2-data-overview)
3. [Basic Data Cleaning & Train/Test Split](#3-basic-data-cleaning--traintest-split)
4. [Exploratory Data Analysis (EDA)](#4-exploratory-data-analysis-eda)
5. [Data Preprocessing](#5-data-preprocessing)
6. [Feature Engineering & Multicollinearity Check](#6-feature-engineering--multicollinearity-check)
7. [Model Pipeline Construction](#7-model-pipeline-construction)
8. [Classification — Model Selection & Evaluation](#8-classification--model-selection--evaluation)
9. [Classification — Hyperparameter Optimization](#9-classification--hyperparameter-optimization)
10. [Regression — Model Selection & Evaluation](#10-regression--model-selection--evaluation)
11. [Regression — Hyperparameter Optimization](#11-regression--hyperparameter-optimization)

---

## 1. Project Introduction & Objective

**Dataset Source:** Kaggle — Car Insurance Claim Data (~10,302 records, 26 features + 1 target)

**Problem Statement:** The project is structured as a **two-part prediction system**:

- **Part 1 — Classification:** Predict *whether* an insurance claim will occur (`is_claim`: 0 or 1)
- **Part 2 — Regression:** Predict *how much* the claim will cost (`new_claim_value`), given that a claim has been made

**Libraries Used:** `pandas`, `numpy`, `seaborn`, `matplotlib`, `scikit-learn`, `statsmodels`, `xgboost`, `catboost`

---

## 2. Data Overview

The dataset contains policyholder information including:

| Feature Type | Examples |
|---|---|
| Demographics | age, gender, date_of_birth, marital_status |
| Financial | income, value_of_home, vehicle_value |
| Driving Profile | num_young_drivers, commute_dist, vehicle_use |
| Claim History | 5_year_total_claims_value, new_claim_value |
| Vehicle Info | vehicle_age, vehicle_color (red_vehicle) |
| Education/Occupation | education, occupation |

**Target Variables:**
- `is_claim` — binary flag (0 = no claim, 1 = claim)
- `new_claim_value` — continuous dollar amount of the claim

---

## 3. Basic Data Cleaning & Train/Test Split

This was the very first step applied to the raw data *before* any analysis.

**Steps Performed:**

- **Column Renaming:** Columns were renamed for readability and clarity (e.g., `KIDSDRIV` → `num_young_drivers`, `CLAIM_FLAG` → `is_claim`)
- **Duplicate Removal:** Dataset was checked for duplicate rows; **1 duplicate** was found and dropped
- **Currency Conversion:** Several monetary columns stored values as strings with `$` and `,` symbols. These were stripped and converted to numeric floats. Affected columns: `income`, `value_of_home`, `vehicle_value`, `5_year_total_claims_value`, `new_claim_value`
- **Train/Test Split:**
  - Ratio: **80% train / 20% test**
  - Method: Stratified sampling using a helper column `claim_value_cat` as the stratification key to preserve the distribution of claim outcomes across both sets
  - `random_state=42` for reproducibility

---

## 4. Exploratory Data Analysis (EDA)

EDA was performed exclusively on the **training set** to avoid data leakage.

**Steps Performed:**

- `X_train` and `y_train` were temporarily joined into a single DataFrame for correlation analysis
- A **Pearson correlation matrix** was computed for all numerical features against the target `is_claim`
- The matrix was visualized using a **Seaborn heatmap** (coolwarm colormap) to identify strong positive/negative correlations

**Key Findings:**

| Feature | Correlation with `is_claim` | Decision |
|---|---|---|
| `commute_dist` | Very weak | Dropped |
| `red_vehicle` | Negligible | Dropped |
| `gender` | Very weak | Noted, retained initially |
| `num_young_drivers`, `vehicle_use` | Moderate positive | Retained |

The EDA directly informed which features to drop in the next step.

---

## 5. Data Preprocessing

Preprocessing was broken into sub-steps:

### 5.1 Feature Dropping

The following columns were removed based on EDA results and logical reasoning:

- `ID` — unique identifier, not a predictive feature
- `red_vehicle` — negligible correlation with target
- `date_of_birth` — already captured by `age`
- `commute_dist` — very low correlation

### 5.2 Missing Value Identification

Missing values were found in 6 columns:

| Column | Missing Count |
|---|---|
| `age` | 7 |
| `years_job_held_for` | 548 |
| `income` | 570 |
| `value_of_home` | 575 |
| `vehicle_age` | 639 |
| `occupation` | 665 |

### 5.3 Imputation Strategy

Two different imputation strategies were used depending on feature type:

- **Numerical features** → `KNNImputer` with `n_neighbors=2` (uses similar records to infer missing values)
- **Categorical features** → `SimpleImputer` with `strategy='most_frequent'` (fills with the mode)

### 5.4 Categorical Encoding

Three encoding strategies were applied based on the nature of each categorical feature:

| Encoding Type | Applied To | Method |
|---|---|---|
| **Ordinal Encoding** | `education` | Custom ranking: `<High School < High School < Bachelors < Masters < PhD` |
| **Binary Encoding** | Yes/No columns (e.g., `is_urban`, `private_car`) | `OrdinalEncoder` mapping to 0/1 |
| **One-Hot Encoding** | Multi-class categoricals (e.g., `occupation`, `vehicle_use`) | `OneHotEncoder` with `handle_unknown='ignore'`, `drop='first'` to avoid dummy variable trap |

---

## 6. Feature Engineering & Multicollinearity Check

### 6.1 Variance Inflation Factor (VIF) Analysis

VIF was computed using `statsmodels.stats.outliers_influence.variance_inflation_factor` to detect multicollinearity — a condition where features are highly correlated with each other, which can destabilize linear models.

- VIF was calculated **before** and **after** one-hot encoding
- Features with high VIF scores (indicating redundancy) were identified and removed
- This step ensured that the `drop='first'` strategy in `OneHotEncoder` correctly handled the dummy variable trap

### 6.2 Feature Scaling

`StandardScaler` was applied to all numerical features:
- Standardizes features to zero mean and unit variance
- While minimal impact on tree-based models (which are scale-invariant), it was retained to benefit potential linear model comparisons during model selection

---

## 7. Model Pipeline Construction

A scikit-learn `Pipeline` + `ColumnTransformer` architecture was used to combine all preprocessing steps into a single, reusable pipeline. This prevents data leakage by ensuring all transformations are fitted only on training data.

**Four sub-pipelines were created:**

| Pipeline Name | Steps |
|---|---|
| **Numerical Pipeline** | `KNNImputer` → `StandardScaler` |
| **Ordinal Categorical Pipeline** | `SimpleImputer (most_frequent)` → `OrdinalEncoder` |
| **Binary Categorical Pipeline** | `SimpleImputer (most_frequent)` → `OrdinalEncoder` |
| **One-Hot Categorical Pipeline** | `SimpleImputer (most_frequent)` → `OneHotEncoder (drop='first')` |

A `ColumnTransformer` then routed each feature to its appropriate sub-pipeline. The final preprocessor was combined with each model into a unified end-to-end `Pipeline`.

---

## 8. Classification — Model Selection & Evaluation

**Objective:** Predict `is_claim` (binary: 0 or 1)

**Evaluation Strategy:** 10-fold cross-validation using `KFold(n_splits=10, shuffle=True, random_state=42)`

**Primary Metric:** F1-score (weighted average) — chosen because the dataset is likely imbalanced (more non-claims than claims), making accuracy a misleading metric

**Models Evaluated:**

| # | Model | Notes |
|---|---|---|
| 1 | Logistic Regression | `solver='liblinear'`, `max_iter=2000` |
| 2 | K-Nearest Neighbors | `KNeighborsClassifier` |
| 3 | Decision Tree | `DecisionTreeClassifier` |
| 4 | Random Forest | `RandomForestClassifier` |
| 5 | Linear SVM | `LinearSVC`, `max_iter=1000`, `dual='auto'` |
| 6 | XGBoost | `XGBClassifier` — **selected for tuning** |
| 7 | AdaBoost | `AdaBoostClassifier`, `algorithm='SAMME'` |
| 8 | Gradient Boosting | `GradientBoostingClassifier` |
| 9 | Bagging | `BaggingClassifier` |
| 10 | CatBoost | `CatBoostClassifier` |

**Outcome:** XGBoost achieved the best cross-validated F1-score and was selected for hyperparameter optimization.

---

## 9. Classification — Hyperparameter Optimization

A two-stage approach was used to efficiently tune the XGBoost classifier:

### Stage 1 — Random Search (`RandomizedSearchCV`)
- Explored a **broad parameter space** (many combinations, random sampling)
- Scoring: `F1-score (weighted)`
- Cross-validation: 10-fold
- Purpose: Narrow down the best parameter region

### Stage 2 — Grid Search (`GridSearchCV`)
- Explored a **fine-grained grid** around the best parameters found in Stage 1
- Scoring: `F1-score (weighted)`
- Purpose: Pinpoint the optimal hyperparameter values

**Final Evaluation on Test Set:**
- Confusion matrix displayed using `ConfusionMatrixDisplay`
- F1-score computed on held-out test data

---

## 10. Regression — Model Selection & Evaluation

**Objective:** Predict `new_claim_value` (continuous dollar amount), but *only for records where `is_claim = 1`*

**Evaluation Strategy:** 10-fold cross-validation

**Primary Metric:** Root Mean Squared Error (RMSE) — also reported MAE and MSE

**Models Evaluated:**

| # | Model | Notes |
|---|---|---|
| 1 | Linear Regression | Baseline |
| 2 | SGD Regressor | `random_state=42` — **selected for tuning** |
| 3 | Decision Tree Regressor | `DecisionTreeRegressor` |
| 4 | Random Forest Regressor | `RandomForestRegressor` |
| 5 | K-Nearest Neighbors Regressor | `KNeighborsRegressor` |
| 6 | SVR | `SVR(gamma=2, C=1)` |
| 7 | XGBoost Regressor | `XGBRegressor` |

**Outcome:** `SGDRegressor` was selected for hyperparameter tuning as it offered more tunable parameters than plain Linear Regression while remaining computationally efficient.

---

## 11. Regression — Hyperparameter Optimization

Same two-stage approach as classification:

### Stage 1 — Random Search (`RandomizedSearchCV`)
- Broad exploration of SGDRegressor's parameter space
- Scoring: `neg_root_mean_squared_error`
- Cross-validation: 10-fold

### Stage 2 — Grid Search (`GridSearchCV`)
- Fine-tuning on the narrowed parameter range from Stage 1
- Scoring: `neg_root_mean_squared_error`

**Final Evaluation on Test Set:**
- RMSE, MAE, and MSE computed on held-out test records (filtered to only claim records)

---

## Summary of Full Pipeline

```
Raw Data
  └── Column Renaming + Duplicate Removal + Currency Conversion
        └── Stratified Train/Test Split (80/20)
              └── EDA on Training Set (Correlation Analysis)
                    └── Feature Dropping (low-correlation features)
                          └── ColumnTransformer Pipeline
                                ├── KNNImputer + StandardScaler (numerical)
                                ├── SimpleImputer + OrdinalEncoder (ordinal)
                                ├── SimpleImputer + OrdinalEncoder (binary)
                                └── SimpleImputer + OneHotEncoder (nominal)
                                      └── VIF Check (multicollinearity removal)
                                            ├── Part 1: Classification
                                            │     ├── 10 models benchmarked (10-fold CV, F1)
                                            │     └── XGBoost tuned (RandomSearch → GridSearch)
                                            └── Part 2: Regression
                                                  ├── 7 models benchmarked (10-fold CV, RMSE)
                                                  └── SGDRegressor tuned (RandomSearch → GridSearch)
```
