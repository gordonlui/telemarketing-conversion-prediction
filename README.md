# 📞 Telemarketing Subscription Prediction using GBM: A Full Machine Learning Workflow

## Overview

This project predicts whether a customer will subscribe to a term deposit following a telemarketing campaign. The final solution uses a tuned **Gradient Boosting Machine (GBM)** model combined with extensive data cleaning, feature engineering, and threshold optimization.

The model achieved an **F1 Score of 0.69814** on the public leaderboard, representing a **28.9% improvement** over the benchmark score of **0.54150**.

---

## Results

| Metric | Score |
|----------|---------|
| Benchmark F1 Score | 0.54150 |
| Final GBM F1 Score | 0.69814 |
| Improvement | 28.9% |

### Key Takeaway

The largest performance gain came from **threshold optimization**, which improved validation F1 by approximately **3.1%**, outperforming the gains achieved through hyperparameter tuning alone.

---

## Data Preparation

The dataset contained 15 predictor variables consisting of both numerical and categorical features. Significant preprocessing was required before modelling.

### Data Cleaning

- Corrected incorrect data types:
  - `avg_balance` → numeric
  - `last_contact_date` → datetime
- Standardised inconsistent date formats between train and test datasets.
- Corrected spacing, capitalization, and typographical errors in categorical variables.
- Fixed records where `age` contained year of birth instead of actual age.
- Treated invalid values (`-1`, `0`) as missing where appropriate.

### Missing Value Imputation

A combination of imputation methods was used:

| Variable Type | Method |
|--------------|---------|
| Age, Average Balance | MICE Imputation |
| Call Duration, Campaign Contacts | Median Imputation |
| Binary & Encoded Variables | Most Frequent Value |

### Previous Campaign Variables

Special preprocessing was required for:

- `prev_days`
- `n_prev_campaign_contacts`
- `prev_outcome`

Customers with incomplete campaign history information were classified as:

```python
prev_outcome = "not_contacted"
```

to ensure consistency across records.

---

## Feature Engineering

### Occupation Binning

Low-performing occupations were grouped into a single category:

```text
services
housemaid
entrepreneur
blue-collar
```

↓

```text
others
```

This reduced cardinality from **12 categories to 9**.

### Average Call Time

Created a new feature:

```python
avg_call_time = total_call_time / n_curr_campaign_contacts
```

This reduced the influence of extreme call durations while providing a more meaningful measure of customer engagement.

### Date Feature Extraction

Extracted the following variables from `last_contact_date`:

- Month
- Day
- Weekday

### Cyclical Encoding

Applied sine and cosine transformations to capture periodic relationships:

```python
month_sin, month_cos
day_sin, day_cos
weekday_sin, weekday_cos
```

This allows the model to recognise continuity between months, days, and weekdays.

### Additional Transformations

Applied:

- One-Hot Encoding
- Log Transformation
- Min-Max Scaling

For `avg_balance`, an offset of +9000 was added before log transformation to accommodate negative values.

### Feature Selection

Removed highly correlated features:

| Removed Feature | Reason |
|---------------|---------|
| total_call_time | Highly correlated with avg_call_time |
| prev_outcome_not_contacted | Highly correlated with prev_days |

This reduced model complexity and improved stability.

---

## Model Development

Several machine learning algorithms were evaluated:

- Logistic Regression
- Decision Tree
- Random Forest
- Neural Network
- Gradient Boosting Machine (GBM)

### Final Model

The best-performing model was a **Gradient Boosting Machine (GBM)** with the following tuned hyperparameters:

```python
max_depth = 14
max_leaf_nodes = 45
min_samples_leaf = 300
max_iter = 250
learning_rate = 0.15
l2_regularization = 20
```

### Classification Threshold

```python
threshold = 0.342
```

The threshold was tuned using repeated stratified K-fold cross-validation to maximise F1 score.

---

## Model Comparison

### Logistic Regression
- Underfit the dataset.
- Lower training and validation F1 scores.

### Decision Tree
- Insufficient complexity to capture nonlinear relationships.

### Random Forest
- Strong performance.
- Higher precision but lower recall than GBM.

### Neural Network
- Promising results.
- Computationally expensive and difficult to tune within project constraints.

### Gradient Boosting Machine (GBM)
- Best overall F1 score.
- Superior balance between precision and recall.
- Sequential learning improved identification of positive cases.

---

## Key Findings

### Threshold Tuning Had the Largest Impact

Validation F1 improved from:

```text
0.685 → 0.706
```

representing a **3.1% improvement**.

Interestingly, threshold tuning contributed more to performance gains than hyperparameter tuning.

### Class Imbalance Methods Did Not Improve GBM Performance

The following approaches were tested:

- `class_weight = balanced`
- SMOTE

Both methods slightly reduced validation F1 performance for GBM, although they improved Logistic Regression performance.

A likely explanation is that GBM's sequential learning process naturally focuses on difficult observations, making it less sensitive to class imbalance.

### Importance of Regularisation

Removing L2 regularisation slightly improved validation F1 but substantially increased variability:

| Model | Validation F1 Standard Deviation |
|---------|---------|
| Without Regularisation | 0.0088 |
| With Regularisation | 0.0013 |

The regularised model was selected for its superior stability and expected generalisation performance.

### Deep Trees Improved Performance

The optimal model selected:

```python
max_depth = 14
```

which is considerably deeper than the weak learners traditionally used in boosting.

Possible explanations include:

- Complex nonlinear relationships between variables.
- Large feature space.
- Most feature selection techniques reduced predictive performance.

Although the final model exhibited slight overfitting:

```text
Mean Training CV F1 = 0.734
```

it achieved the strongest validation and leaderboard performance.

---

## Repository Structure

```text
├── data/
│   ├── train.csv
│   └── test.csv
├── notebooks/
│   └── modelling.ipynb
├── src/
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── train_model.py
│   └── predict.py
├── outputs/
│   ├── submissions/
│   └── figures/
├── README.md
└── requirements.txt
```

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-Learn
- HistGradientBoostingClassifier
- MICE Imputation
- Cross Validation
- Feature Engineering

---

## Future Improvements

- Explore advanced neural network architectures.
- Compare with:
  - XGBoost
  - LightGBM
  - CatBoost
- Investigate automated feature selection techniques.
- Build ensemble models to further improve predictive performance.

---

## Author

**Gordon Lui**

Actuarial Science & Data Analytics Portfolio
