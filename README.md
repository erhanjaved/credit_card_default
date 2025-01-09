# Blog Post: Predicting Credit Card Defaults with Machine Learning

This blog post provides an overview of our analysis and modeling efforts to predict credit card defaults using the **Default of Credit Card Clients Dataset**. The target audience is someone with a solid technical foundation but limited prior exposure to machine learning. The dataset can be accessed [here](https://www.kaggle.com/uciml/default-of-credit-card-clients-dataset).

---

## Problem Description
The objective was to predict whether a client will default on their credit card payment next month. This is a binary classification problem where the target variable, `default.payment.next.month`, is labeled as `1` (default) or `0` (no default). The analysis aimed to assist financial institutions in identifying high-risk customers.

---

## Dataset Description
The dataset contains 30,000 samples and 24 features, including:
- **Demographics**: Gender, age, education, marital status.
- **Financial Information**: Credit limit (`LIMIT_BAL`), past bill amounts (`BILL_AMT1` to `BILL_AMT6`), and repayment status (`PAY_0` to `PAY_6`).
- **Target Variable**: `default.payment.next.month`.

### Key Observations:
1. **Class Imbalance**:  
   | Class                  | Count   | Percentage |
   |------------------------|---------|------------|
   | Non-defaulting clients | 23,310  | 77.7%      |
   | Defaulting clients     | 6,690   | 22.3%      |

2. **Demographics**:
   - Majority of clients are in their 20s and 30s, with an average age of 35.5.
   - Slightly more female clients than male.
   - Most clients have a university education, with fewer at graduate or high school levels.

3. **Repayment Status**:
   - Delays in repayment correlate with higher default probabilities.

---

## Exploratory Data Analysis (EDA)
### Visualizations:
1. **Class Distribution**: Highlighted the imbalance between defaulting and non-defaulting clients.
2. **Repayment Status**: Delays in September (most recent data point) showed a strong association with default rates.  
3. **Age and Credit Limit Trends**: Explored demographic patterns affecting default behavior.

---

## Feature Engineering and Preprocessing
### Steps Taken:
1. **Dimensionality Reduction**:
   - Averaged bill amounts (`AVG_BILL_AMT`) and payment amounts (`AVG_PAY_AMT`).
2. **Scaling**:
   - Normalized numeric features (`LIMIT_BAL`, `AGE`, `AVG_BILL_AMT`, `AVG_PAY_AMT`).
3. **Encoding**:
   - Categorical variables (`MARRIAGE`, `EDUCATION`) were encoded ordinally to preserve hierarchy.

---

## Model Selection and Results
We tested multiple models and tuned hyperparameters to optimize performance:

### Model Comparison Table:
| Model                 | Best Hyperparameters                    | Cross-Validation Score | Test Score |
|-----------------------|------------------------------------------|-------------------------|------------|
| Random Forest         | `n_estimators=50, max_depth=5`          | 0.83                    | 0.82       |
| Decision Tree         | `max_depth=5`                           | 0.819                   | 0.819      |
| Logistic Regression   | `C=1`                                   | 0.810                   | 0.810      |
| kNN                   | `n_neighbors=17`                        | 0.813                   | 0.813      |

### Random Forest Performance Metrics:
| Metric    | Score |
|-----------|-------|
| Precision | 0.66  |
| Recall    | 0.37  |
| F1 Score  | 0.48  |

---

## Caveats
1. **Class Imbalance**:
   - Models may focus too heavily on the majority class (non-defaulting clients).
2. **Feature Averaging**:
   - Combining `BILL_AMT` and `PAY_AMT` features might have lost crucial temporal details.
3. **Limited Generalizability**:
   - Dataset was specific to a single time and region, possibly reducing its applicability elsewhere.

---

## Communication Technique
We employed the **"bottom-up explanations"** approach:
- Started with data exploration and EDA.
- Progressed logically through feature engineering, model selection, and evaluation.
- Used embedded visualizations to make the narrative more intuitive and engaging.

---

This analysis demonstrates the practical application of machine learning in financial risk management. By identifying patterns in client behavior, institutions can make more informed decisions and mitigate risk effectively.
