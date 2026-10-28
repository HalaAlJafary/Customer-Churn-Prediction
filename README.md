# 📉 Customer Churn Prediction using Data Mining

A data mining project that predicts which telecom customers are likely to **leave the company (churn)**, using **Logistic Regression** on the Telco Customer Churn dataset.

**Authors:** Hala AlJafary. Aseel Hassan · Aseel Alshishani . Maria Jaber

---

## 📌 Problem Definition

Customer churn is one of the most critical challenges for telecommunications companies: losing customers means financial losses and higher costs to acquire new ones. Predicting churn early helps the company:

- Detect at-risk customers before they leave
- Focus retention efforts (and incentives) where they matter most
- Improve service quality and protect revenue

The goal is a **binary classification** model: `1` = customer will churn, `0` = customer will stay.

---

## 📊 Dataset

- **Name:** Telco Customer Churn ([Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn))
- **Size:** 7,043 customers × 21 columns (types: object, float, integer)
- **Description:** a fictitious telecom company that provided home phone and Internet services to customers in California. Each row shows a customer's profile, subscribed services, account information, and whether they churned.
- **Target:** `Churn` (Yes / No)

> The dataset file is **not included** in this repository. Download it and place it next to the notebook (see [Getting Started](#-getting-started)).
>
> The copy used in this project contains 70 missing `tenure` values and 105 `Unknown` entries in `InternetService`; the notebook handles both (and also works if they are absent).

---

## 🧹 Data Preprocessing

| Step | What was done |
|---|---|
| Remove unnecessary columns | Dropped `customerID` |
| Fix data type mismatch | `TotalCharges` (object) converted to float; 11 non-numeric values became `NaN` |
| Handle missing values | Missing `TotalCharges` and `tenure` (70 values) filled with the **median** |
| Handle unknown classes | `Unknown` in `InternetService` (105 rows) replaced with the **mode** |
| Binary encoding | `Partner`, `Dependents`, `PhoneService`, `PaperlessBilling`, `Churn`: `Yes/No` → `1/0` |
| One-hot encoding | All remaining categorical columns |
| Outlier inspection | Boxplots of `tenure`, `MonthlyCharges`, `TotalCharges` (no rows removed) |

After preprocessing the dataset has **42 columns** (41 features + the `Churn` target). There were no duplicate rows.

---

## 🧠 Model

**Logistic Regression** (scikit-learn, default settings):

- Suited to a binary outcome (leave / stay)
- Outputs a churn **probability** for each customer (default threshold 0.5)
- Easy to interpret and explain to business stakeholders

**Train / test split:** 80% / 20% (`random_state=42`) → 5,634 training and 1,409 test samples.

---

## ✅ Results

| Metric | Score |
|---|---|
| **Accuracy** | 0.8155 |
| **Precision** | 0.6828 |
| **Recall** | 0.5657 |
| **F1-Score** | 0.6188 |

**Confusion matrix** (test set):

|  | Predicted: Stay | Predicted: Churn |
|---|---|---|
| **Actual: Stay** | 938 | 98 |
| **Actual: Churn** | 162 | 211 |

**How to read it:**
- **Accuracy** looks good, but the dataset is imbalanced (373 of the 1,409 test customers churned), so accuracy alone can be misleading.
- **Precision (68%)**: when the model predicts churn, it is right about 68% of the time, so relatively few retention offers are wasted.
- **Recall (57%)**: the model catches 211 of the 373 customers who actually left; 162 are missed. Recall is the key metric to improve, since every missed churner is lost revenue.
- **F1-Score (62%)** balances the two.

---

## 📂 Repository Structure

```
.
├── Customer-Churn-Prediction.ipynb      # EDA, preprocessing, model, evaluation
├── Customer-Churn-Presentation.pdf      # Project presentation slides
└── README.md
```

---

## 🛠️ Tech Stack

Python · pandas · seaborn · matplotlib · scikit-learn · Jupyter

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
```

### 2. Install dependencies
```bash
pip install pandas seaborn matplotlib scikit-learn jupyter
```

### 3. Add the dataset
Download the Telco Customer Churn CSV from [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn), place it in the project folder, and make sure `DATA_PATH` in the notebook's settings cell points to it:

```python
DATA_PATH = "Telco_Customer_Churn.csv"
```

### 4. Run
```bash
jupyter notebook Customer-Churn-Prediction.ipynb
```

---

## 🔮 Possible Improvements

- **Improve recall:** handle class imbalance (`class_weight='balanced'`, SMOTE) and/or tune the decision threshold below 0.5.
- **Fix the convergence warning:** the model reports that `lbfgs` did not converge because the numeric features are on different scales. Scaling them (e.g., `StandardScaler`) or increasing `max_iter` would fix this.
- **Avoid slight data leakage:** compute the median values on the training set only (e.g., with a scikit-learn `Pipeline`) instead of on the full dataset.
- **Compare other models:** Random Forest, Gradient Boosting / XGBoost, and use cross-validation for more reliable estimates.

---

## 📄 License

Add your preferred license here (e.g., MIT).
