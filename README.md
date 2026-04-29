# Machine Learning Portfolio
### Vijayalakshmi Veeraiyan (Viji)

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue)](www.linkedin.com/in/vijayalakshmi-veeraiyan-6761421a1)
[![GitHub](https://img.shields.io/badge/GitHub-Profile-black)](https://github.com/Viji-Veer)
[![Location](https://img.shields.io/badge/Location-Vantaa%2C%20Finland-green)](https://github.com/Viji-Veer)

---

## About This Repository

This repository contains my machine learning 
projects — each one built from scratch, 
fully documented, and validated before 
being presented publicly.

Every project follows the same professional 
standard:

- ✅ Real datasets with honest analysis
- ✅ Clear business problem and measurable impact
- ✅ End-to-end pipeline from raw data to insights
- ✅ Validated results with automated checks
- ✅ Documented notebooks explaining every decision

I believe data science is not just about 
building models that score well on paper, but 
it is about solving real business problems 
with data and communicating findings clearly 
to people who need to act on them.

---

## Projects

### 1. User Retention & Churn Analysis
**Status:** ✅ Complete | **Streamlit App:** 🔄 In Progress

**Business Problem:**
A telecom company wants to identify customers 
at risk of leaving before they actually do — 
enabling targeted retention strategies that 
protect revenue.

**Dataset:** 7,043 customers, 38 features 
(IBM Telecom Customer Churn — Kaggle)

**Approach:**
- Exploratory data analysis revealing key 
  churn drivers
- Feature engineering — 6 new variables 
  including Monthly Spend Ratio
- SMOTE for class imbalance handling
- Gradient Boosting selected from 3 algorithms
- Hyperparameter tuning with RandomizedSearchCV
- Threshold optimisation for maximum business value

**Key Results:**

| Metric | Value |
|---|---|
| Algorithm | Gradient Boosting |
| ROC AUC | 0.8931 |
| Accuracy at optimal threshold | 82.61% |
| Recall (churners caught) | 82% |
| Optimal threshold | 0.2735 |
| Churners identified | 306 of 373 |

**Business Impact:**
*(Assumed benchmarks: CLV $1,680, retention cost $50)*

| Metric | Value |
|---|---|
| Value from retained customers | $498,780 |
| Cost of retention offers | $8,900 |
| Missed churner costs | $112,560 |
| **Net benefit** | **$377,320** |
| **ROI** | **1,961%** |

**Top Churn Predictors:**
1. Month-to-Month contracts (importance: 0.384)
2. Number of referrals (importance: 0.108)
3. Monthly Spend Ratio (importance: 0.073)

**Tools & Technologies:**
Python, Scikit-learn, Pandas, NumPy, 
Matplotlib, Seaborn, Imbalanced-learn, Joblib

**Project Structure:**
```
telecom-churn-prediction/
├── notebooks/
│   ├── 1_data_exploration.ipynb
│   ├── 2_preprocessing.ipynb
│   └── 3_modeling.ipynb
├── data/
│   └── raw/
│       └── telecom_customer_churn.csv
├── images/
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── feature_importance.png
│   └── business_impact.png
├── requirements.txt
└── README.md
```

[📂 View Project](https://github.com/Viji-Veer/MLprojects/tree/main/telecom-churn-prediction)

---

### 2. Customer Segmentation Analysis
**Status:** 🔄 Coming Soon

**Business Problem:**
Understanding distinct customer groups 
to enable personalised marketing, 
targeted offers, and improved retention 
strategies for each segment.

**Planned Approach:**
- K-Means clustering
- Optimal cluster selection using 
  Elbow method and Silhouette score
- Segment profiling and business 
  recommendations
- Interactive visualisation of segments

**Tools planned:** Python, Scikit-learn, 
Pandas, Matplotlib, Seaborn

---


### 3. Player Behaviour Analysis
**Status:** 🔄 Coming Soon

**Business Problem:**
Understanding what drives player 
engagement, session length, and 
retention in gaming — identifying 
patterns that distinguish highly 
engaged players from those at risk 
of dropping off.

**Planned Approach:**
- Exploratory analysis of player 
  session data
- Engagement pattern identification
- Churn risk scoring for players
- Feature importance analysis

**Tools planned:** Python, Pandas, 
Scikit-learn, Matplotlib, Seaborn

---

## Technical Skills Demonstrated

| Skill | Projects |
|---|---|
| Classification | User Retention & Churn |
| Clustering | Customer Segmentation (coming) |
| Time Series | Sales Forecasting (coming) |
| Feature Engineering | All projects |
| Class Imbalance (SMOTE) | User Retention & Churn |
| Hyperparameter Tuning | User Retention & Churn |
| Business Impact Analysis | User Retention & Churn |
| Model Deployment (Streamlit) | Coming soon |

---

## How I Work

Every project in this repository follows 
the same disciplined approach:

**1. Understand the business problem first**
Before writing any code — what decision 
will this analysis support? What does 
success look like in business terms?

**2. Explore before modeling**
Thorough exploratory data analysis 
before any model is built. The patterns 
found here guide every subsequent decision.

**3. Document every decision**
Every preprocessing choice, every 
algorithm selection, every threshold 
decision has a reason — documented 
clearly in the notebooks.

**4. Validate before presenting**
Every number presented publicly has 
been verified through automated 
validation checks. No unverified 
claims.

**5. Translate into business value**
Model performance metrics are always 
translated into financial or operational 
terms that business stakeholders can 
understand and act on.

---



---

*Last updated: 2025*  
*All projects built and validated independently*
