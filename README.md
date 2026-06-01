#  DecodeLabs Internship — Predicting Airline Flight Delays

This repository contains the complete work done during my internship at **DecodeLabs**. The project was structured across five progressive tasks — from raw data collection all the way to a fully optimized machine learning model. Each task has its own dedicated repository with its own code, notebook, and documentation.

---

##  Project Summary

**Problem:** Flight delays cost airlines and passengers billions every year. Predicting how long a flight will be delayed — before it happens — can help airlines plan better and help passengers manage their time.

**Solution:** We built a regression-based machine learning model trained on 70,000 historical flight records that predicts arrival delay in minutes using features like departure delay, carrier, origin airport, time of year, and delay type breakdowns.

**Final Result:** The optimized Random Forest Regressor achieved an R² Score of **0.996** — meaning the model explains 99.6% of the variance in flight arrival delays.

| Metric | Score |
|---|---|
| MAE | 1.2240 |
| RMSE | 3.2386 |
| R² Score | 0.9961 |

---

##  Repository Structure

This main repository is the starting point. All five tasks are organized into their own separate repositories below:

| Task | Repository | Description |
|---|---|---|
| Task 1 | [Task-1-Data-Collection](./Task-1-Data-Collection) | Data loading and initial dataset understanding |
| Task 2 | [Task-2-Data-Preprocessing](./Task-2-Data-Preprocessing) | Data cleaning, duplicates, inconsistencies, and formatting |
| Task 3 | [Task-3-EDA](./Task-3-EDA) | Exploratory Data Analysis, encoding, outliers, and correlation |
| Task 5 | [Task-5-Final-Project-Model-Building-Optimization](./Task-5-Final-Project-Model-Building-Optimization) | Model comparison, hyperparameter tuning, and final evaluation |

---

##  Project Workflow

The project follows a clean end-to-end machine learning pipeline:

```
Task 1 — Data Collection & Understanding
         ↓
Task 2 — Data Cleaning & Preprocessing
         ↓
Task 3 — Exploratory Data Analysis (EDA)
         ↓
Task 5 — Model Building & Optimization
```

Each task builds directly on the one before it. Nothing was skipped and nothing was done out of order — the pipeline was followed exactly as a real-world data science project would be.

---

##  Dataset

| Property | Detail |
|---|---|
| **File** | `Flight_delay.csv` |
| **Records Used** | 70,000 rows |
| **Total Features** | 29 columns |
| **Target Variable** | `ArrDelay` — Arrival Delay in minutes |
| **Task Type** | Supervised Learning — Regression |

The dataset includes historical domestic airline flight records covering scheduling details, carrier information, origin and destination airports, and delay breakdowns by cause (carrier, weather, NAS, security, late aircraft).

---

##  Technologies Used

| Tool / Library | Purpose |
|---|---|
| Python 3 | Core programming language |
| Pandas | Data manipulation and analysis |
| NumPy | Numerical operations |
| Matplotlib & Seaborn | Data visualization |
| Scikit-learn | Preprocessing, modeling, and evaluation |

---

##  Models Compared

Five regression models were trained and evaluated on the same 80/20 train/test split:

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Linear Regression | — | — | — |
| Decision Tree | — | — | — |
| Random Forest | 1.224 | 3.239 | 0.9960 |
| Gradient Boosting | — | — | — |
| Extra Trees | — | — | — |

> Random Forest was selected as the final model and further optimized using GridSearchCV with 5-fold cross-validation.

---

##  Before vs After Tuning

| Metric | Before Tuning | After Tuning |
|---|---|---|
| MAE | 1.2240 | 1.2240 |
| RMSE | 3.2390 | 3.2386 ✅ |
| R² Score | 0.9960 | 0.9961 ✅ |

The improvements are intentionally modest — the default Random Forest was already well-configured. The real value of tuning was **confirmation through cross-validation** that the model generalizes well and is not just performing well on one lucky train/test split.

---

##  Key Learnings

- **EDA is not optional.** Every preprocessing and modeling decision in this project came from something we saw during exploratory analysis. Rushing past it would have meant building a model on shaky foundations.

- **Encoding strategy is a real decision.** One-Hot Encoding for low-cardinality columns and Frequency Encoding for high-cardinality ones — choosing the wrong approach would have either exploded the feature space or lost important information.

- **GridSearchCV + Cross-Validation together is the right workflow.** Running them separately gives an overly optimistic and unreliable picture of model performance.

- **A high R² score needs context.** 0.996 is a strong result, but it is important to remember the model was trained on a specific subset of 70,000 rows. Generalization to unseen airlines, airports, or time periods would need further validation.

---

##  Future Improvements

- Fill in the full model comparison table with actual scores for all five models
- Integrate real-time weather data as an additional feature
- Build a Streamlit web app so the model can be used without opening a notebook
- Experiment with XGBoost and LightGBM for potential performance gains
- Expand the dataset beyond 70,000 rows for better generalization

---

##  Full Project Structure

```
DecodeLabs-Internship/
│
├── Task-1-Raza-Ur-Rahman/
│   ├── Task-1-Raza-Ur-Rahman.ipynb
│   └── README.md
│
├── Task-2-Raza-Ur-Rahman/
│   ├── Task-2-Raza-Ur-Rahman.ipynb
│   └── README.md
│
├── Task-3-Raza-Ur-Rahman/
│   ├── Task-3-Raza-Ur-Rahman3.ipynb
│   └── README.md
│
├── Task-5-Raza-Ur-Rahman/
│   ├── Final_Project_Task_1_Documented.ipynb
│   └── README.md
│
└── README.md  ← You are here
```

---

## 👤 Author

Made as part of an internship at **DecodeLabs**.  
Feel free to open an issue or reach out if you have questions or suggestions.

- **GitHub:** https://github.com/RazaBhatti03
- **LinkedIn:** https://linkedin.com/in/raza-ur-rahman-33h
