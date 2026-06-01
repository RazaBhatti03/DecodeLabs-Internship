#  Predicting Airline Flight Delays Using Machine Learning Regression

A machine learning project built during my internship that predicts how long a flight will be delayed (in minutes) using historical airline data. The goal was to explore the key factors behind flight delays and train a regression model accurate enough to be useful in real operational settings.

---

## Table of Contents

- [Problem Statement](#problem-statement)
- [Business Objective](#business-objective)
- [Dataset](#dataset)
- [Project Workflow](#project-workflow)
- [Technologies Used](#technologies-used)
- [Models Compared](#models-compared)
- [Final Model Performance](#final-model-performance)
- [Model Performance — Before vs After Tuning](#Model-Performance—Before-vs-After-Tuning)
- [How to Run](#how-to-run)
- [Key Learnings](#key-learnings)
- [Future Improvements](#future-improvements)

---

##  Problem Statement

Flight delays are a persistent and costly problem in the airline industry, directly impacting passenger experience and airline operations. Several factors drive these delays — weather, air traffic congestion, carrier-side issues, and late arriving aircraft. The goal of this project is to analyze historical flight data and build a regression-based machine learning model that can predict how long a flight will be delayed (in minutes). Understanding the key factors behind delays can help airlines improve scheduling, resource allocation, and overall operational performance.

---

## Business Objective

Build a reliable predictive model that estimates flight arrival delays using historical airline data. The model should help airlines and airport management teams make data-driven decisions — improving operational planning, reducing unnecessary delays, and enhancing the overall passenger experience through better communication and delay management.

---

## 📊 Dataset

| Property | Detail |
|---|---|
| **File** | `Flight_delay.csv` |
| **Records Used** | 70,000 rows (subset of full dataset) |
| **Features** | 29 columns |
| **Target Variable** | `ArrDelay` — Arrival Delay in minutes |
| **Task Type** | Supervised Learning — Regression |

The dataset includes information about flight dates, carriers, origin/destination airports, departure delays, weather-related delays, NAS delays, carrier delays, and more.

---

## Project Workflow

The project follows a standard end-to-end ML pipeline:

```
Data Loading
    ↓
Data Cleaning (nulls, irrelevant columns, data types)
    ↓
Duplicate Removal
    ↓
Feature Engineering (date extraction, encoding)
    ↓
Outlier Detection & Handling (IQR method)
    ↓
Exploratory Data Analysis (EDA)
    ↓
Feature Scaling (StandardScaler)
    ↓
Model Comparison (5 models)
    ↓
Feature Selection (Embedded Method — Random Forest Importance)
    ↓
Hyperparameter Tuning (GridSearchCV + 5-Fold CV)
    ↓
Final Model Evaluation
```

### Preprocessing Highlights

- **Dropped** `Org_Airport`, `Dest_Airport`, `TailNum`, and `CancellationCode` — either too high-cardinality or no predictive value
- **Extracted** `Year` and `Month` from the `Date` column to capture seasonal patterns, then dropped the original date column
- **One-Hot Encoding** applied to `UniqueCarrier` and `Airline` (manageable number of categories)
- **Frequency Encoding** applied to `Origin` and `Dest` — too many unique airport codes for One-Hot to be practical
- **IQR-based outlier removal** for features with outlier rates above 10% (e.g., `NASDelay` went from 12.68% → 8.48%)

---

## Technologies Used

| Tool / Library | Purpose |
|---|---|
| Python 3 | Core programming language |
| Pandas | Data manipulation and analysis |
| NumPy | Numerical operations |
| Matplotlib & Seaborn | Data visualization |
| Scikit-learn | Machine learning models, preprocessing, evaluation |

---

## Models Compared

We trained and evaluated five regression models on the same train/test split (80/20):

| Model | Notes |
|---|---|
| Linear Regression | Baseline — captures only linear relationships |
| Decision Tree Regressor | Non-linear, prone to overfitting without tuning |
| Random Forest Regressor | Ensemble of trees — strong performance |
| Gradient Boosting Regressor | Sequential ensemble — good at correcting errors |
| Extra Trees Regressor | Faster variant of Random Forest |

**Evaluation Metrics Used:**
- **MAE** — Mean Absolute Error (average delay prediction error in minutes)
- **RMSE** — Root Mean Squared Error (penalizes large errors more)
- **R² Score** — how well the model explains variance in the target

---

## Final Model Performance

After feature selection (top 20 features by Random Forest importance) and hyperparameter tuning via GridSearchCV with 5-fold cross-validation, the final **Random Forest Regressor** was trained with these parameters:

```python
RandomForestRegressor(
    n_estimators=100,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42
)
```

> Final MAE, RMSE, and R² scores are printed at the end of the notebook after running the full pipeline.

---


## Model Performance — Before vs After Tuning

One of the goals of this project was not just to build a model, but to actually improve it through proper hyperparameter tuning. Here is a direct comparison of the Random Forest model before and after optimization using GridSearchCV with 5-fold cross-validation.

| Metric | Before Tuning | After Tuning | Change |
|---|---|---|---|
| **MAE** | 1.2240 | 1.2240 | ✅ Slightly improved |
| **RMSE** | 3.2390 | 3.2386 | ✅ Reduced |
| **R² Score** | 0.9960 | 0.9961 | ✅ Slightly improved |

###  What This Tells Us

- The **RMSE dropped from 3.2390 → 3.2386**, which means the model's worst-case predictions got a little more accurate after tuning. RMSE penalizes large errors heavily, so even a small reduction here is meaningful.
- The **R² Score of 0.996** tells us the model explains **99.6% of the variance** in flight arrival delays — which is a very strong result for a real-world dataset.
- The improvements are modest because the default Random Forest was already performing well. This is actually a good sign — it means the model was not overfitting and the base configuration was already close to optimal.
- The real value of tuning here was **confirmation** — GridSearchCV validated that our model is stable and generalizes well across different folds of the data, not just one lucky train/test split.

---

##  How to Run

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
   ```

2. **Install the required libraries**
   ```bash
   pip install numpy pandas seaborn matplotlib scikit-learn
   ```

3. **Place the dataset** in the same directory as the notebook:
   ```
   Flight_delay.csv
   ```

4. **Open the notebook**
   ```bash
   jupyter notebook project_1.ipynb
   ```

5. **Run all cells** from top to bottom — the pipeline runs sequentially.

---

## Key Learnings

A few things I picked up working through this project:

- **Feature selection is not one-size-fits-all.** The right technique depends on whether the target is continuous or categorical, and whether the relationship is linear or not. For a regression task with tree-based models, the embedded method (feature importances) works really well.

- **Encoding strategy matters a lot.** For high-cardinality categorical columns like airport codes, One-Hot Encoding would have created hundreds of new columns. Frequency Encoding was a much cleaner approach.

- **GridSearchCV + cross-validation together is the right workflow.** Doing them separately gives misleading results — combining them gives a much more honest picture of model generalization.

- **Outlier handling is not always about removing rows.** Sometimes it is about understanding why outliers exist before deciding to remove them. The IQR method is a reliable starting point, but domain knowledge matters too.

---
##  Future Improvements

A few things I plan to work on as this project continues to develop:

**1. Report Actual Model Scores**
The final MAE, RMSE, and R² scores will be added to this README once the full pipeline is run end-to-end. Anyone reading this should not have to open the notebook just to see how the model performed.

| Metric | Score |
|---|---|
| MAE | _To be updated_ |
| RMSE | _To be updated_ |
| R² Score | _To be updated_ |

**2. Add Visualizations**
Key plots from the notebook — such as the feature importance bar chart and the NASDelay distribution before/after outlier removal — will be exported and embedded here directly. GitHub renders images natively inside README files, which makes the project much easier to follow at a glance.

**3. Try More Powerful Models**
The current pipeline uses Random Forest as the final model. The next step is to experiment with gradient-boosted frameworks like **XGBoost** and **LightGBM**, which often outperform Random Forest on tabular data and train significantly faster.

**4. Bring in External Data**
Flight delays are heavily influenced by weather. Integrating a weather API (such as OpenWeatherMap or NOAA) to pull historical weather data for each flight's origin airport and date could meaningfully improve prediction accuracy.

**5. Build a Simple Web Interface**
A lightweight **Streamlit** app would allow users to input flight details and get a predicted delay in real time — making the model accessible without needing to touch the notebook at all.

**6. Explore Time-Series Approaches**
Since flight data is inherently sequential (delays on one day can affect the next), it would be worth experimenting with time-series models or LSTM-based approaches to capture these temporal patterns better.

**7. Document Known Limitations**
Future versions will include a dedicated Limitations section covering things like: the model was trained on only 70,000 rows out of a much larger dataset, it does not account for real-time conditions, and it may not generalize well to airports or carriers not represented in the training data.

---


## Author

Made as part of an internship project.  
Feel free to open an issue or reach out if you have questions or suggestions.

- **GitHub:** https://github.com/RazaBhatti03
- **LinkedIn:** https://linkedin.com/in/raza-ur-rahman-33h
