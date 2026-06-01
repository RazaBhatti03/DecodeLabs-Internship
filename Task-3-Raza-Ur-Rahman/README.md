#  Task 3 — Exploratory Data Analysis (EDA)

> **Part of the Internship Project Series:** Predicting Airline Flight Delays Using Machine Learning Regression

---

##  Goal

Once the data is clean, the next step is to truly understand it before building any model. EDA is where we ask the right questions — what does the data look like, how are the features distributed, are there outliers, how do features relate to each other, and which ones are actually useful for predicting flight delays?

Skipping EDA and jumping straight to modeling is one of the most common mistakes in data science. The decisions made here directly shape every modeling choice that comes after.

---

##  What This Task Covers

- Basic structure inspection after cleaning
- Statistical summary of all features
- Feature encoding (preparing categorical columns for numeric analysis)
- Skewness and distribution analysis
- Coefficient of Variation (CV) analysis
- Outlier detection and handling using the IQR method
- Correlation analysis for feature selection

---

##  EDA Steps — In Order

### Step 1 — Basic Structure Inspection

After all the cleaning done in Task 2, we started by rechecking the shape, a sample of the data, and column types to make sure everything looked as expected before going deeper.

```python
# Shape of the cleaned dataset
data.shape

# Quick look at the first two rows
data.head(2)

# Column-level info: types and non-null counts
data.info()

# Data types of all columns
data.dtypes
```

This confirmed the dataset was clean, consistently typed, and ready for deeper analysis.

---

### Step 2 — Statistical Summary

We generated a full statistical summary across all numeric features — mean, standard deviation, min, max, and percentiles. This gives a first feel for the scale, spread, and range of each column.

```python
data.describe()
```

A few things stood out at this stage:

- Several delay columns (`CarrierDelay`, `WeatherDelay`, `NASDelay`, etc.) had very high max values compared to their means — a strong signal of skewness and outliers.
- `DayOfWeek` had a very symmetric distribution — mean close to 4, spread evenly across 1–7.
- The target variable `ArrDelay` had both negative values (early arrivals) and large positive values (severe delays).

---

### Step 3 — Feature Encoding

Before we could run any numeric analysis — skewness checks, correlation, outlier detection — all categorical columns needed to be converted to numbers. EDA methods like IQR and Pearson correlation only work on numeric data.

**This is why encoding comes before outlier handling and correlation analysis, not after.**

#### One-Hot Encoding — for low cardinality columns

Applied to `UniqueCarrier` and `Airline` — manageable number of unique values.

```python
data = pd.get_dummies(data, columns=['UniqueCarrier'], drop_first=True)
data = pd.get_dummies(data, columns=['Airline'], drop_first=True)
```

`drop_first=True` drops one dummy column per feature to avoid multicollinearity.

#### Frequency Encoding — for high cardinality columns

Applied to `Origin` and `Dest`. Both had hundreds of unique airport codes — One-Hot Encoding would have created hundreds of extra columns, making the dataset unnecessarily wide and hard to work with.

Instead, each airport code was replaced with how often it appeared in the dataset.

```python
freq = data['Origin'].value_counts()
data['Origin'] = data['Origin'].map(freq)

freq = data['Dest'].value_counts()
data['Dest'] = data['Dest'].map(freq)
```

#### Converting Boolean Columns to Integer

`get_dummies` produces boolean (True/False) columns by default. We converted all of them to integer (0/1).

```python
bool_cols = data.select_dtypes(include='bool').columns
data[bool_cols] = data[bool_cols].astype(int)
```

After this step, all columns were fully numeric — confirmed via `data.info()`.

---

### Step 4 — Distribution and Skewness Analysis

Understanding the shape of each feature's distribution helps us decide how to handle outliers and whether any transformations are needed.

We used `DayOfWeek` as a representative example to demonstrate the full analysis workflow.

#### Skewness Check

```python
data['DayOfWeek'].skew()
```

| Skew Range | Distribution Nature |
|---|---|
| -0.5 to 0.5 | Approximately Normal |
| 0.5 to 1.0 | Moderately Skewed |
| Above 1.0 | Highly Skewed |

#### Visual Inspection

```python
sns.histplot(data['DayOfWeek'], kde=True)
plt.show()
```

The histogram with KDE curve shows the shape of the distribution visually — whether it is symmetric, left-skewed, or right-skewed.

#### Coefficient of Variation (CV) Analysis

CV measures how spread out the data is relative to its mean. It is a useful way to compare variability across features that have different scales.

```python
# CV = (Standard Deviation / Mean) × 100
1.92 / 3.85 * 100
```

| CV Value | Meaning |
|---|---|
| Less than 20% | Very stable, low spread |
| 20% – 40% | Moderate variation |
| Above 40% | High variation |

#### Mean vs Median Comparison

In a perfectly normal distribution, Mean ≈ Median. A gap between the two tells us about skew direction.

For `DayOfWeek`:
- Mean = 3.5
- Median = 4.0

This small gap confirmed a slight **left skew** — the distribution leans marginally to the left.

---

### Step 5 — Outlier Detection and Handling (IQR Method)

Outliers are extreme values that sit far outside the general pattern of the data. In a flight delay dataset, these often represent unusual events — severe storms, system-wide groundings, or data entry errors. If left in, they can pull the model's predictions in the wrong direction.

We used the **IQR (Interquartile Range)** method to detect outliers across all features.

```python
def calculate_outlier_percentages(data):
    outlier_percentages = {}
    for col in data.columns:
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = data[(data[col] < lower_bound) | (data[col] > upper_bound)][col]
        outlier_percentages[col] = len(outliers) / len(data) * 100
    return outlier_percentages

outlier_percentages_all = calculate_outlier_percentages(data)
```

**Decision Rule:** Any feature with an outlier percentage above **10%** was treated as a priority for cleaning.

`NASDelay` was the main column that crossed this threshold.

```python
def iqr_outlier_range(data):
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    min_range = Q1 - 1.5 * IQR
    max_range = Q3 + 1.5 * IQR
    return min_range, max_range

min_range, max_range = iqr_outlier_range(data['NASDelay'])
data = data[(data['NASDelay'] >= min_range) & (data['NASDelay'] <= max_range)]
```

We visualized `NASDelay` before and after removal using both a boxplot and a distribution plot to confirm the improvement.

```python
# Boxplot — shows the spread and outlier points clearly
sns.boxplot(x='NASDelay', data=data, color='lightblue')
plt.show()

# Distribution plot — shows how the shape changed after cleaning
sns.distplot(data['NASDelay'])
plt.show()
```

**Result:**

| Column | Before Cleaning | After Cleaning |
|---|---|---|
| `NASDelay` | 12.68% outliers | 8.48% outliers ✅ |

---

### Step 6 — Correlation Analysis

With all features now numeric and outliers handled, we ran a correlation analysis to identify redundant features — pairs of columns that carry the same information. Keeping both in the model adds noise without adding value.

We used a **threshold of 0.90** — any two features correlated above this were flagged as redundant.

```python
import numpy as np

threshold = 0.90
corr_matrix = data.corr(numeric_only=True).abs()

# Look at the upper triangle only to avoid counting each pair twice
upper = corr_matrix.where(
    np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
)

high_corr_pairs = []
for col in upper.columns:
    for row in upper.index:
        corr_value = upper.loc[row, col]
        if pd.notna(corr_value) and corr_value > threshold:
            high_corr_pairs.append([row, col, round(corr_value, 4)])

high_corr_df = pd.DataFrame(
    high_corr_pairs,
    columns=['Feature_1', 'Feature_2', 'Correlation']
).sort_values(by='Correlation', ascending=False)

print(high_corr_df)
```

**Important note on Pearson Correlation:**
Pearson only captures **linear relationships**. For checking feature-vs-feature redundancy this is fine — we are just looking for duplicate signals. For measuring feature importance against the target, we later used Random Forest importance scores (Task 5), which capture non-linear relationships as well.

---

## ✅ Task 3 — Summary

| Check | Outcome |
|---|---|
| Basic structure inspected | ✅ Shape, dtypes, and info confirmed |
| Statistical summary generated | ✅ `describe()` run across all features |
| Categorical columns encoded | ✅ One-Hot for low-cardinality, Frequency for high-cardinality |
| Boolean columns converted to int | ✅ All `bool` columns cast to `int` |
| Skewness checked | ✅ `DayOfWeek` analyzed as representative example |
| CV analysis completed | ✅ Spread measured relative to mean |
| Outliers detected and handled | ✅ IQR method applied — `NASDelay` cleaned from 12.68% → 8.48% |
| Correlation analysis completed | ✅ High-corr pairs identified using 0.90 threshold |

> **Next Step → Task 4: Feature Selection & Scaling** — using the insights from EDA, we now select the most important features and scale them before passing them into the model.

---


---

## 👤 Author

Made as part of an internship project.  
Feel free to open an issue or reach out if you have questions or suggestions.

- **GitHub:** [https://github.com/your-username](https://github.com/RazaBhatti03)
- **LinkedIn:** [https://linkedin.com/in/your-profile](https://linkedin.com/in/raza-ur-rahman-33h)

