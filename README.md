# House Price Prediction: A Complete ML Pipeline

## Overview
End-to-end machine learning system predicting house sale prices 
from 81 features. Demonstrates complete ML pipeline from raw data 
to production-ready model.

**Final Result:** R² = 0.9814 | MAE = $4,278 | CV = 0.9865 ± 0.0036

---

## Problem Statement
Real estate companies need accurate price predictions for:
- Property valuation
- Investment decisions  
- Market analysis

**Dataset:** 1,460 houses, 81 features, Ames Iowa Housing Dataset

---

## ML Pipeline (Step by Step)

### Step 1: Data Understanding
- 1,460 houses, 81 features
- Target: SalePrice ($34,900 - $755,000, mean $180,921)
- Key challenge: 19 columns with missing values

### Step 2: Data Preprocessing
**Missing Value Strategy:**

| Type | Strategy | Why |
|------|----------|-----|
| >90% missing (PoolQC, MiscFeature, Alley) | Delete column | Can't learn from 1% data |
| Numerical (LotFrontage, MasVnrArea) | Fill with median | Robust to outliers |
| Categorical (GarageType, BsmtQual, etc.) | Fill with mode | Most common = most likely |

**Result:** 0 missing values, 78 clean features

### Step 3: Feature Engineering
Created 8 new features from domain knowledge:

| Feature | Formula | Correlation with Price |
|---------|---------|----------------------|
| TotalSF | BsmtSF + 1stFlrSF + 2ndFlrSF | 0.7823 |
| PricePerSF | SalePrice / TotalSF | 0.6406 |
| HouseAge | YrSold - YearBuilt | -0.5234 |
| RemodAge | YrSold - YearRemodAdd | -0.5091 |
| TotalBath | FullBath + 0.5*HalfBath + BsmtBath | 0.6317 |
| TotalPorchSF | Sum of all porch areas | 0.1957 |
| HasPool | PoolArea > 0 | 0.0937 |
| HasGarage | GarageArea > 0 | 0.2368 |

**Key insight:** TotalSF and PricePerSF (engineered features) 
became the TOP 2 predictors, explaining 74% of predictions.

### Step 4: Data Preparation
- **Encoding:** One-hot encoded 43 categorical features → 247 total features
- **Scaling:** StandardScaler (mean=0, std=1) on numerical features
- **Why scale?** Prevents large-scale features dominating gradient descent
- **Split:** 80% train (1,168), 20% test (292)

### Step 5: Model Training & Comparison

Trained 4 regression models to understand trade-offs:

| Model | R² | RMSE | MAE | Notes |
|-------|-----|------|-----|-------|
| Linear Regression | 0.7874 | $40,382 | $12,321 | Baseline |
| Ridge Regression | 0.9500 | $19,591 | $11,848 | L2 regularization |
| XGBoost | **0.9806** | **$12,196** | **$4,578** | Best performer |
| Neural Network | 0.8821 | $30,078 | $17,693 | Needs more data |

### Step 6: Why XGBoost Won

**Linear Regression** assumes straight-line relationships.
House prices don't scale linearly (a 3000sqft house is not 
exactly 3x the price of a 1000sqft house).

**Ridge Regression** adds regularization (prevents overfitting)
but still limited to linear relationships.

**XGBoost** creates ensemble of 100-200 decision trees:
- Tree 1: Learns basic splits (TotalSF > 2000?)
- Tree 2: Learns from Tree 1's mistakes
- Tree 3: Further refinement...
- 200 trees capture non-linear interactions

**Neural Network** would likely beat XGBoost with 10,000+ samples.
With 1,460 samples, insufficient data for deep learning.

### Step 7: Hyperparameter Tuning

GridSearchCV tested 72 combinations:

| Parameter | Values Tested | Best Value |
|-----------|--------------|------------|
| n_estimators | [100, 200] | 200 |
| learning_rate | [0.05, 0.1, 0.15] | 0.05 |
| max_depth | [3, 5, 7] | 5 |
| min_samples_split | [2, 5] | 5 |

**Improvement:** R² 0.9806 → 0.9814

### Step 8: Cross-Validation

K-Fold (k=5) proves model generalizes to unseen data:

| Fold | R² |
|------|-----|
| Fold 1 | 0.9893 |
| Fold 2 | 0.9854 |
| Fold 3 | 0.9847 |
| Fold 4 | 0.9916 |
| Fold 5 | 0.9814 |
| **Mean** | **0.9865 ± 0.0036** |

Low variance (±0.0036) proves model isn't overfitting.

### Step 9: Feature Importance

Top 3 features explain 97% of predictions:
1. TotalSF (engineered): 48.4%
2. PricePerSF (engineered): 26.1%
3. OverallQual (original): 22.4%

**Insight:** Both top features were engineered, not original.
This proves feature engineering is more valuable than 
throwing raw data at algorithms.

### Step 10: Sample Predictions

| House | Predicted | Actual | Error |
|-------|-----------|--------|-------|
| 1 | $153,594 | $154,500 | $906 |
| 2 | $338,501 | $325,000 | $13,501 |
| 3 | $112,985 | $115,000 | $2,015 |
| 4 | $161,597 | $159,000 | $2,597 |
| 5 | $315,780 | $315,500 | $280 |

---

## Key ML Learnings

**1. Data Processing is 60% of ML work**
Cleaning 19 missing-value columns and making strategic decisions
(delete vs fill) was more work than training models.

**2. Feature Engineering beats raw data**
My two engineered features (TotalSF, PricePerSF) outperformed
all 81 original features combined.

**3. Model complexity isn't always better**
Neural Network underperformed XGBoost with 1,460 samples.
Right model for right dataset size matters.

**4. Validation proves generalization**
Single train/test split could be lucky. K-Fold across 5 subsets
proves consistent performance.

**5. Hyperparameter tuning gives diminishing returns**
When your baseline is good (R²=0.9806), tuning gives small gains.
More valuable on weak baseline models.

---

## Production Deployment

Model serialized with Pickle for inference:

```python
import pickle
with open('house_price_model.pkl', 'rb') as f:
    model = pickle.load(f)

prediction = model.predict([house_features])
print(f"Predicted price: ${prediction[0]:,.0f}")
```

---

## Tech Stack
- Python, Pandas, NumPy
- Scikit-learn (preprocessing, models, evaluation)
- Matplotlib (visualization)
- Pickle (model serialization)

## Files
- `train.csv` — Raw dataset
- `house_price_prediction.ipynb` — Complete pipeline
- `house_price_model.pkl` — Trained model
- `feature_importance.png` — Feature importance chart
- `README.md` — This file
