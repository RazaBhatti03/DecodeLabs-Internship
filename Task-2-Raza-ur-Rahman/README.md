# Task 2 — Data Cleaning & Preprocessing

> **Part of the Internship Project Series:** Predicting Airline Flight Delays Using Machine Learning Regression

---

## Goal

Raw data is almost never ready for a machine learning model straight out of the box. It contains missing values, duplicate records, wrong data types, irrelevant columns, and inconsistent labels. If we skip this step or rush through it, the model ends up learning from noise — and no amount of fancy algorithms can fix that later.

This task is entirely about making the data trustworthy before we do anything else with it.

---

## What This Task Covers

- Checking and handling missing values
- Dropping irrelevant or high-cardinality columns
- Removing duplicate rows
- Fixing data types
- Identifying and handling inconsistent or near-constant columns
- Feature engineering from the `Date` column

---

## Preprocessing Steps — In Order

### Step 1 — Handling Missing Values

The first thing we did was check which columns had missing values and how many.

```python
data.isnull().sum()
```

After inspecting the results, we found that `Org_Airport` and `Dest_Airport` both had too many unique categorical values with no clean encoding path — full airport names are not something a model can use directly. Since `Origin` and `Dest` already carry the airport codes, these two columns were redundant and dropped.

```python
data.drop(['Org_Airport', 'Dest_Airport'], axis=1, inplace=True)
```

**Result:** Shape went from `(70000, 29)` → `(70000, 27)`

---

### Step 2 — Removing Duplicate Rows

Duplicate records can mislead the model by artificially inflating certain patterns in the data. We checked for exact duplicates and removed them.

```python
duplicates = data.duplicated().sum()
data.drop_duplicates(inplace=True)
```

---

### Step 3 — Fixing Data Types and Removing Irrelevant Columns

We inspected the data types of all columns and identified a few that needed attention:

| Column | Issue | Action Taken |
|---|---|---|
| `TailNum` | Aircraft registration number — too many unique values, no predictive value | Dropped |
| `CancellationCode` | Almost entirely one value — no discriminating power | Dropped |
| `Date` | Stored as a string, not usable as-is | Converted to datetime, extracted `Year` and `Month` |

```python
# Drop TailNum
data.drop('TailNum', axis=1, inplace=True)

# Convert Date and extract useful features
data['Date'] = pd.to_datetime(data['Date'], dayfirst=True)
data['Year']  = data['Date'].dt.year
data['Month'] = data['Date'].dt.month
data.drop('Date', axis=1, inplace=True)

# Drop CancellationCode
data.drop('CancellationCode', axis=1, inplace=True)
```

**Why extract Year and Month from Date?**
A raw date string is meaningless to a model. But Year and Month carry real signal — flight delays tend to spike during holiday seasons and certain weather months. By extracting these as separate numeric columns, we give the model a way to learn those seasonal patterns.

---

### Step 4 — Handling Data Inconsistencies

After fixing data types, we did a deeper inspection to catch any remaining inconsistencies — things like columns with too many unique values, near-constant columns, or labels that don't make sense in context.

```python
# Check unique value counts across all columns
for col in data.columns:
    print(f"{col}: {data[col].nunique()} unique values")

# Inspect unique values in all categorical columns
cat_cols = data.select_dtypes(include='object').columns
for col in cat_cols:
    print(f"\n{col}:", data[col].unique())
```

**What we found and fixed:**

| Column | Inconsistency Found | Action Taken |
|---|---|---|
| `TailNum` | Thousands of unique aircraft codes — no usable pattern | Already dropped in Step 3 |
| `CancellationCode` | One dominant value across almost all rows | Already dropped in Step 3 |
| `Org_Airport` | Full airport name strings — too many unique values | Already dropped in Step 1 |
| `Dest_Airport` | Same issue as `Org_Airport` | Already dropped in Step 1 |

After all these steps, we confirmed that the remaining columns were clean, consistently labeled, and ready for the next stage.

```python
# Final check — confirm data types and non-null counts across all remaining columns
data.info()
```

---

##  Before vs After — Dataset Shape

| Stage | Shape |
|---|---|
| Raw dataset loaded | (70,000 × 29) |
| After dropping `Org_Airport`, `Dest_Airport` | (70,000 × 27) |
| After dropping `TailNum`, `CancellationCode`, `Date` | (70,000 × 24) |
| After removing duplicate rows | Rows slightly reduced |

---

## ✅ Task 2 — Summary

| Check | Outcome |
|---|---|
| Missing values handled | ✅ Redundant and noisy columns dropped |
| Duplicate rows removed | ✅ `drop_duplicates()` applied |
| Data types corrected | ✅ `Date` converted, `Year` and `Month` extracted |
| Irrelevant columns dropped | ✅ `TailNum`, `CancellationCode`, `Org_Airport`, `Dest_Airport` |
| Inconsistencies identified and resolved | ✅ Near-constant and high-cardinality columns handled |
| Dataset confirmed clean | ✅ Verified via `data.info()` |

> **Next Step → Task 3: Exploratory Data Analysis (EDA)** — now that the data is clean and consistent, we explore distributions, skewness, correlations, and outliers before moving into modeling.

---

## 📁 Project Structure
