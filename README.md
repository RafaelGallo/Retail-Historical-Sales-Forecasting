## 📊 Forecasting Weekly Department Sales Across 45 Retail Stores Using Time Series, Gradient Boosting, and Deep Learning

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![Kaggle](https://img.shields.io/badge/Kaggle-Notebook-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-FF6600?style=for-the-badge&logo=xgboost&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-02569B?style=for-the-badge&logo=lightgbm&logoColor=white)
![CatBoost](https://img.shields.io/badge/CatBoost-FFD700?style=for-the-badge&logo=catboost&logoColor=black)
![SciPy](https://img.shields.io/badge/SciPy-8CAAE6?style=for-the-badge&logo=scipy&logoColor=white)
![Statsmodels](https://img.shields.io/badge/Statsmodels-4051B5?style=for-the-badge&logo=python&logoColor=white)
![pmdarima](https://img.shields.io/badge/pmdarima-AutoARIMA-FF4B4B?style=for-the-badge&logo=python&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=python&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-4C72B0?style=for-the-badge&logo=python&logoColor=white)
![Pickle](https://img.shields.io/badge/Pickle-Model%20Serialization-FF69B4?style=for-the-badge&logo=python&logoColor=white)
![Joblib](https://img.shields.io/badge/Joblib-Model%20Save-8B0000?style=for-the-badge&logo=python&logoColor=white)
![ARIMA](https://img.shields.io/badge/ARIMA-Time%20Series-4B0082?style=for-the-badge&logo=python&logoColor=white)
![SARIMA](https://img.shields.io/badge/SARIMA-Seasonal-6A0DAD?style=for-the-badge&logo=python&logoColor=white)
![AutoARIMA](https://img.shields.io/badge/AutoARIMA-pmdarima-FF4500?style=for-the-badge&logo=python&logoColor=white)
![Anomaly](https://img.shields.io/badge/Anomaly%20Detection-Z--Score%20%7C%20IQR%20%7C%20Rolling-E84545?style=for-the-badge&logo=python&logoColor=white)
![Retail](https://img.shields.io/badge/Domain-Retail%20Analytics-16A085?style=for-the-badge&logo=shoppingcart&logoColor=white)
![Forecasting](https://img.shields.io/badge/Task-Sales%20Forecasting-2C3E50?style=for-the-badge&logo=python&logoColor=white)
![TimeSeries](https://img.shields.io/badge/Time%20Series-Weekly%20Sales-E67E22?style=for-the-badge&logo=python&logoColor=white)
![Holiday](https://img.shields.io/badge/Holiday%20Impact-Thanksgiving%20%7C%20Christmas-E84545?style=for-the-badge&logo=python&logoColor=white)
![MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge&logo=opensourceinitiative&logoColor=white)

![](https://github.com/RafaelGallo/Retail-Historical-Sales-Forecasting/blob/main/img/log.png?raw=true)

## 🏪 Business Problem

A retail chain operating **45 stores across different regions** needs to forecast weekly department-level sales to:

- Optimize inventory replenishment ahead of peak demand periods
- Plan promotional markdown budgets around major holiday weeks
- Allocate staffing and resources during high-impact events such as Super Bowl, Labor Day, Thanksgiving, and Christmas
- Predict which departments will be most affected by markdowns and to what extent

The four major holiday weeks are weighted **5x higher** in business evaluation than non-holiday weeks, making accurate holiday forecasting a critical priority.

## 📂 Dataset

**Source:** [Walmart Retail Data Analytics — Kaggle](https://www.kaggle.com/datasets/manjeetsingh/retaildataset)
**License:** CC0 Public Domain

| File | Description |
|---|---|
| `sales data-set.csv` | Weekly sales per store and department (Feb 2010 – Nov 2012) |
| `features data set.csv` | External features: temperature, fuel price, markdowns, CPI, unemployment |
| `stores data-set.csv` | Store type (A, B, C) and size |

**Key Variables**

| Variable | Description |
|---|---|
| `Weekly_Sales` | Target variable — weekly sales in USD |
| `Store` | Store identifier (1–45) |
| `Dept` | Department identifier |
| `IsHoliday` | Whether the week includes a major holiday |
| `Type` | Store type: A (largest), B, C (smallest) |
| `Size` | Store size in square feet |
| `MarkDown1–5` | Anonymized promotional markdown data |
| `Temperature` | Regional average temperature |
| `Fuel_Price` | Regional fuel price |
| `CPI` | Consumer Price Index |
| `Unemployment` | Regional unemployment rate |

## 🔧 Pipeline Overview

```
Raw Data (3 CSV files)
        ↓
   Merge & Parse Dates
        ↓
   Outlier Treatment (Z-Score |z| > 3)
        ↓
   Dataset Restructuring (leakage-free)
        ↓
   Exploratory Data Analysis
        ↓
   Anomaly Detection (Z-Score | IQR | Rolling ±2σ)
        ↓
   Correlation Analysis
        ↓
┌────────────────────────────────────┐
│  Section A — Time Series           │
│  AutoARIMA → ARIMA / SARIMA        │
└────────────────────────────────────┘
        ↓
┌────────────────────────────────────┐
│  Section B — Gradient Boosting     │
│  XGBoost | LightGBM | CatBoost     │
│  Gradient Boosting | AdaBoost      │
└────────────────────────────────────┘
        ↓
   Model Evaluation (MAE, MSE, RMSE, R²)
        ↓
   Future Forecast (+12 weeks)
        ↓
   Business Insights & Recommendations
```

## 📈 Exploratory Data Analysis

The EDA covered the full merged dataset of **421,570 rows** across 45 stores, 99 departments and 143 weeks.

**Key findings from EDA:**

- **Store Type A** generates the highest average weekly sales, followed by Type B and Type C
- **Top departments by revenue** are concentrated in grocery, electronics and apparel categories
- Weekly sales follow a strong **annual seasonal pattern** with consistent peaks in November–December
- The overall mean weekly sales per department is approximately **$15,981**
- Holiday weeks account for only **~7% of all weeks** but drive disproportionately high revenue

## 🔍 Anomaly Detection

Three methods were applied in combination:

| Method | Criterion |
|---|---|
| Z-Score | \|z\| > 2.5 |
| IQR | Outside Q1 − 1.5×IQR and Q3 + 1.5×IQR |
| Rolling Mean ±2σ | Outside 8-week rolling band |

**Consensus anomalies** (flagged by at least 2 of 3 methods):

| Date | Event | Methods | Sales |
|---|---|---|---|
| 2010-12-24 | Christmas Eve Peak | Z-Score + IQR | $80.9M |
| 2011-12-23 | Christmas Peak | Z-Score + IQR | $77.0M |
| 2011-11-25 | Thanksgiving / Black Friday | **All 3** | $66.6M |
| 2010-11-26 | Thanksgiving / Black Friday | **All 3** | $65.8M |
| 2010-12-17 | Pre-Christmas Shopping | Z-Score + IQR | $61.8M |
| 2012-04-06 | Easter Weekend | IQR + Rolling | $53.5M |

All anomalies are **genuine business events**, not data errors — confirming the health of the dataset and the importance of holiday modeling.

## 🧹 Outlier Treatment

Z-Score treatment with threshold |z| > 3.0 applied across all numerical columns:

| Feature | Outliers Removed | Outlier % |
|---|---|---|
| Unemployment | 13,756 | 3.26% |
| Weekly_Sales | 8,848 | 2.10% |
| MarkDown1 | 8,723 | 2.07% |
| MarkDown4 | 6,731 | 1.60% |
| MarkDown2 | 5,655 | 1.34% |
| MarkDown5 | 5,455 | 1.29% |
| MarkDown3 | 2,773 | 0.66% |
| Temperature | 69 | 0.02% |
| Fuel_Price | 0 | Clean ✓ |
| CPI | 0 | Clean ✓ |
| Size | 0 | Clean ✓ |

**Rows after treatment:** 386,614 (removed 8.29% of original data)

## 📉 Section A — Time Series Models

AutoARIMA was used to automatically search for optimal `(p, d, q)` and seasonal `(P, D, Q, m)` parameters using AIC as the optimization criterion.

| Model | Order | AIC |
|---|---|---|
| ARIMA | (1, 0, 0) | — |
| SARIMA | (1, 0, 1)(0, 1, 2, 52) | — |

**Results on 12-week holdout:**

| Model | MAE | RMSE | R² |
|---|---|---|---|
| ARIMA | 9,108,061 | 10,445,031 | -1.41 |
| SARIMA | 14,910,919 | 16,269,091 | -4.85 |

Both models produced **negative R²**, indicating they perform worse than a simple mean baseline. At the aggregated multi-store scale with 143 weekly observations, univariate time series models cannot capture the structural complexity without exogenous features. SARIMA additionally generated **negative sales forecasts** in future weeks due to instability in the seasonal component estimation with m=52 and limited training data.

## 🚀 Section B — Gradient Boosting Models

Five models trained on the **store-department level dataset** (386,614 rows) using only original source columns — no engineered features to prevent data leakage.

**Features used:**

```
Store | Dept | IsHoliday | Type | Size |
MarkDown1 | MarkDown2 | MarkDown3 | MarkDown4 | MarkDown5
```

**Temporal train/test split:**
- Train: Feb 2010 → Aug 2012
- Test (holdout): Sep 2012 → Nov 2012 (12 weeks)

## 🏆 Model Results

| Ranking | Model | R² | MAE | RMSE |
|---|---|---|---|---|
| 🥇 | XGBoost | **0.9818** | 690,877 | 906,077 |
| 🥈 | LightGBM | 0.9821 | 699,636 | 899,093 |
| 🥉 | Gradient Boosting | 0.9811 | 743,501 | 924,396 |
| 4° | CatBoost | 0.9786 | 760,032 | 983,272 |
| 5° | AdaBoost | -1.37 | 10,183,355 | 10,358,455 |
| 6° | ARIMA | -1.41 | 9,108,061 | 10,445,031 |
| 7° | SARIMA | -4.85 | 14,910,919 | 16,269,091 |

> **Note:** XGBoost and LightGBM are virtually tied. XGBoost wins on MAE (lowest absolute error per prediction), while LightGBM wins on RMSE (lower sensitivity to large errors). Both are production-ready.

## 🔮 Forecast Results

Future forecasts were generated for **12 weeks beyond the dataset end date** (Nov 2012 → Jan 2013) using same-period-last-year values as the feature anchor — a strategy that allows Gradient Boosting models to reproduce seasonal patterns without extrapolation.

**Forecast behavior by model:**

| Model | Forecast Quality | Notes |
|---|---|---|
| XGBoost | ✅ Excellent | Captures seasonal dip and Thanksgiving peak |
| LightGBM | ✅ Excellent | Very close to XGBoost, stable CI band |
| Gradient Boosting | ✅ Good | Slightly conservative on peak magnitude |
| CatBoost | ✅ Good | Wider variance on holiday weeks |
| AdaBoost | ⚠️ Unstable | High amplitude swings, not recommended |
| ARIMA | ❌ Poor | Flat declining line, extremely wide CI |
| SARIMA | ❌ Failed | Negative forecasts, unstable seasonal component |

**Visual evidence from forecast plots confirms:**
- Top 4 Gradient Boosting models **agree on direction and magnitude**, forming a reliable consensus
- All top models correctly anticipate the **Thanksgiving/Christmas sales surge** in late November–December
- The grey reference line (same period last year) sits within the forecast cluster, validating seasonal calibration

## 💡 Business Insights

**1. Holiday weeks are the most critical forecasting period**
Thanksgiving and Christmas Eve weeks consistently exceed $65M–$80M in total sales across all stores — 40% to 71% above the weekly average. These six anomalous weeks require dedicated inventory and staffing strategies.

**2. Store Type A dominates revenue**
Type A stores generate significantly higher weekly sales than Type B and C. Inventory pre-positioning and markdown campaigns should prioritize Type A stores for maximum ROI.

**3. Markdowns have measurable promotional lift**
MarkDown1 and MarkDown2 show positive correlation with Weekly Sales. Promotional campaigns launched 2 weeks before major holiday windows generate the highest measurable lift.

**4. Macroeconomic variables have negligible predictive power**
CPI, Unemployment, Fuel Price and Temperature showed weak or negative correlation with Weekly Sales and were excluded from the final model. Store identity, department and size are far stronger predictors.

**5. Department-level forecasting outperforms aggregated models**
Training at the Store × Department level (386K rows) versus aggregated weekly series (143 rows) produced a dramatic improvement — from R² = −1.41 (ARIMA) to R² = 0.98 (XGBoost).

## ✅ Recommendations

| Priority | Action | Expected Impact |
|---|---|---|
| 🔴 High | Deploy XGBoost in production with weekly refit | Consistent R²=0.98 forecasts |
| 🔴 High | Pre-position inventory 6–8 weeks before Nov | Avoid stockouts during Thanksgiving/Christmas |
| 🟡 Medium | Launch markdowns 2 weeks before holiday peaks | Maximize promotional lift from MarkDown1/2 |
| 🟡 Medium | Prioritize Type A stores in planning | Highest revenue concentration |
| 🟢 Low | Implement automated anomaly alerts | Flag weeks forecast >10% above baseline |
| 🟢 Low | Monitor AdaBoost and remove from production | Unstable forecasts, R²=0.22 |

## 🛠️ Tech Stack

```
Python 3.10+
├── Data
│   ├── pandas
│   └── numpy
├── Visualization
│   ├── matplotlib
│   └── seaborn
├── Statistics
│   └── scipy
├── Time Series
│   ├── pmdarima (AutoARIMA)
│   └── statsmodels (SARIMAX)
├── Machine Learning
│   ├── scikit-learn
│   ├── xgboost
│   ├── lightgbm
│   └── catboost
└── Explainability
    └── shap
```

## 📁 Project Structure

```
retail-sales-forecasting/
│
├── data/
│   ├── sales data-set.csv
│   ├── features data set.csv
│   ├── stores data-set.csv
│   ├── dataset_before_treatment.csv
│   └── dataset_after_treatment.csv
│
├── models/
│   ├── xgboost.pkl
│   ├── lightgbm.pkl
│   ├── catboost.pkl
│   ├── gradient_boosting.pkl
│   ├── adaboost.pkl
│   ├── arima.pkl
│   ├── sarima.pkl
│   └── metadata.json
│
├── notebook/
│   └── retail_sales_forecasting.ipynb
│
├── requirements.txt
└── README.md
```

## ▶️ How to Run

**1. Clone the repository**
```bash
git clone https://github.com/RafaelGallo/retail-sales-forecasting.git
cd retail-sales-forecasting
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run on Kaggle (recommended)**

The notebook is optimized for the Kaggle environment with GPU support. Upload to Kaggle and set the dataset path to:
```
/kaggle/input/retail-data-analytics/
```

**4. Run locally**
```bash
jupyter notebook notebook/retail_sales_forecasting.ipynb
```
## 📄 License

This project is licensed under the MIT License. The dataset is licensed under CC0 Public Domain.
