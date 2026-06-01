# Task 1 — Data Collection & Dataset Understanding

> **Part of the Internship Project Series:** Predicting Airline Flight Delays Using Machine Learning Regression

---

##  Goal

The first step in any machine learning project is to load the data and truly understand what you are working with before touching a single line of preprocessing code. Jumping straight into cleaning or modeling without understanding the data is one of the most common mistakes in data science. This task is entirely about slowing down and asking the right questions first.

---

## What This Task Covers

- Loading the raw dataset into a Pandas DataFrame
- Checking the size and shape of the data
- Identifying all columns and their data types
- Understanding what each feature represents
- Getting a first look at the statistical properties of the data

---

## Dataset Overview

| Property | Detail |
|---|---|
| **File** | `Flight_delay.csv` |
| **Total Records (Full)** | More than 70,000 rows |
| **Records Used** | 70,000 rows (first 70,000 selected for this project) |
| **Total Columns** | 29 |
| **Target Variable** | `ArrDelay` — Arrival Delay in minutes |
| **Task Type** | Supervised Learning — Regression |

The dataset contains historical domestic airline flight records. Each row represents a single flight, and the columns capture various aspects of that flight — from scheduling details to delay breakdowns by cause.

---

## What the Data Represents

This is a flight operations dataset. It tells us:

- **When** the flight operated — date, day of the week, departure and arrival times
- **Who** operated it — the airline carrier and aircraft tail number
- **Where** it went — origin and destination airports
- **How it performed** — whether it was delayed, cancelled, or diverted, and by how much
- **Why it was delayed** — broken down into specific causes such as carrier delay, weather delay, NAS delay, security delay, and late aircraft delay

This kind of data is used by airlines, airport authorities, and regulatory bodies to track on-time performance and identify systemic issues in operations.

---

## Key Columns Identified

| Column | Type | Description |
|---|---|---|
| `Date` | Object | Date of the flight |
| `DayOfWeek` | Integer | Day of the week (1 = Monday, 7 = Sunday) |
| `UniqueCarrier` | Object | Airline carrier code (e.g., WN, AA, DL) |
| `Airline` | Object | Full airline name |
| `TailNum` | Object | Aircraft registration number |
| `Origin` | Object | Origin airport code |
| `Dest` | Object | Destination airport code |
| `Org_Airport` | Object | Full name of origin airport |
| `Dest_Airport` | Object | Full name of destination airport |
| `CRSDepTime` | Integer | Scheduled departure time |
| `DepTime` | Float | Actual departure time |
| `DepDelay` | Float | Departure delay in minutes |
| `CRSArrTime` | Integer | Scheduled arrival time |
| `ArrTime` | Float | Actual arrival time |
| `ArrDelay` | Float | **Target — Arrival delay in minutes** |
| `Cancelled` | Integer | Whether the flight was cancelled (1 = Yes) |
| `CancellationCode` | Object | Reason for cancellation |
| `Diverted` | Integer | Whether the flight was diverted (1 = Yes) |
| `CarrierDelay` | Float | Delay caused by the carrier (minutes) |
| `WeatherDelay` | Float | Delay caused by weather (minutes) |
| `NASDelay` | Float | Delay caused by National Air System (minutes) |
| `SecurityDelay` | Float | Delay caused by security issues (minutes) |
| `LateAircraftDelay` | Float | Delay caused by a late arriving aircraft (minutes) |

---

## Initial Observations

A few things stood out during the first look at the data:

- The **target column `ArrDelay`** was not at the end of the DataFrame by default. It was moved to the last position to make the feature/target separation cleaner in later steps.
- **`TailNum`** has a very high number of unique values (one per aircraft) — this raised an early flag that it may not be useful as a feature.
- **`Org_Airport` and `Dest_Airport`** are text-heavy columns with full airport names. These are likely redundant given that `Origin` and `Dest` already carry the airport codes.
- **`CancellationCode`** is almost entirely one value — which means it carries very little information for a model.
- Several delay columns (`CarrierDelay`, `WeatherDelay`, `NASDelay`, etc.) contain missing values — likely because those fields are only filled when a delay of that type actually occurred.

These observations directly shaped the decisions made in Task 2 (Data Preprocessing).

---

## Code Snapshot

```python
import numpy as np
import pandas as pd

# Load the dataset
df = pd.read_csv("Flight_delay.csv")

# Work with the first 70,000 rows
data = df.head(70000)

# Move target column to the end
cols = [col for col in data.columns if col != 'ArrDelay'] + ['ArrDelay']
data = data[cols]

# Check shape, column names, and data types
print(data.shape)
print(data.columns)
print(data.dtypes)
print(data.describe())
```

---

## ✅ Task 1 — Summary

| Check | Outcome |
|---|---|
| Dataset loaded successfully | ✅ |
| Shape confirmed | ✅ 70,000 rows × 29 columns |
| Column names identified | ✅ 23 meaningful features + target |
| Data types reviewed | ✅ Mix of int, float, and object columns |
| Target variable identified | ✅ `ArrDelay` (continuous — regression task) |
| Early problem areas flagged | ✅ High-cardinality columns, mostly-null delay fields, near-constant columns |

> **Next Step → Task 2: Data Preprocessing** — where we handle missing values, remove duplicates, fix data types, and clean up all the issues identified here.

---

---

## 👤 Author

Made as part of an internship project.  
Feel free to open an issue or reach out if you have questions or suggestions.

- **GitHub:** [your-username](https://github.com/your-username)
- **LinkedIn:** [your-linkedin-profile](https://linkedin.com/in/your-profile)
