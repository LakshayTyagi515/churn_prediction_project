# 📱 Telecom Customer Churn Prediction

## Problem Statement

A telecom company wants to identify which customers are likely to cancel their subscription (churn) based on their service usage, contract details, and payment methods, so it can intervene with targeted retention offers before they leave and reduce customer loss.

## Dataset

- **Name:** Telco Customer Churn
- **Source:** Kaggle (blastchar)
- **Link:** https://www.kaggle.com/datasets/blastchar/telco-customer-churn
- **Rows / Columns:** 7,043 customer records / 21 features

## Tools Used

- Python 3.11
- Pandas, NumPy (Data manipulation)
- Matplotlib, Seaborn (Data visualization)
- Scikit-learn (Machine Learning)
- Streamlit (Web Application)
- Pickle (Model serialization)

## Workflow

1. **Data Collection** - Downloaded Telco Customer Churn dataset from Kaggle
2. **Data Cleaning** - Converted TotalCharges to numeric, handled missing values, standardized categories
3. **Exploratory Data Analysis (EDA)** - Created 6 visualizations analyzing churn patterns by contract type, tenure, internet service, and payment method
4. **Feature Engineering** - Created tenure_group buckets, calculated avg_monthly_spend, performed one-hot encoding on 10+ categorical features
5. **Model Building** - Trained Logistic Regression with balanced class weights to handle 27% churn rate
6. **Evaluation** - Assessed model using Accuracy, Precision, Recall, F1 Score, and Confusion Matrix
7. **Insights & Recommendations** - Identified 3 data-driven retention strategies with expected ROI

## Results

- **Model:** Logistic Regression (with class_weight='balanced')
- **Key Metrics:**
  - **Accuracy:** 82.15% - Correctly predicts churn status for 82% of customers
  - **Precision:** 65.32% - When model predicts churn, it's correct 65% of the time
  - **Recall:** 58.47% - Catches 58% of customers who will actually churn
  - **F1 Score:** 0.6178 - Balanced performance metric

- **Top Churn Drivers:**
  1. **Month-to-Month Contract** - 42.71% churn rate (15x higher than 2-year contracts)
  2. **Low Tenure / New Customers** - Churned customers average 17.9 months vs 32.4 months for retained
  3. **Manual Payment Methods** - Electronic check/mail check show higher churn than auto-pay

## Screenshots

### Churn by Contract Type

![Churn by Contract Type](Images/01_churn_by_contract.png)

### Tenure Distribution

![Tenure Impact](Images/02_tenure_distribution.png)

### Model Performance - Confusion Matrix

![Confusion Matrix](Images/05_confusion_matrix.png)

### Top Churn Drivers

![Churn Drivers](Images/06_churn_drivers.png)

## Recommended Retention Strategies

### Strategy 1: Contract Conversion Campaign ⭐ **HIGHEST PRIORITY**

- **Target:** Month-to-month customers (especially 0-12 months)
- **Action:** Offer 15-20% discount for 12-month contract
- **Expected Impact:** 25-35% churn reduction, 300-400% ROI

### Strategy 2: Fiber Optic Service Excellence Program

- **Target:** Fiber optic service users
- **Action:** Proactive monitoring, satisfaction guarantees, free upgrades
- **Expected Impact:** 15-20% churn reduction

### Strategy 3: Auto-Pay Incentive Program ⭐ **QUICK WIN**

- **Target:** Manual payment method customers
- **Action:** 5-10% discount for automatic payment
- **Expected Impact:** 25-35% adoption increase, 10-15% churn reduction

## Future Improvements

- Test Random Forest and XGBoost models for potentially higher accuracy
- Implement SMOTE for better handling of class imbalance
- Collect additional features (customer satisfaction, support tickets)
- Set up automated monthly churn risk scoring for all customers
- Run A/B testing to validate retention strategy effectiveness
- Deploy mobile app for on-the-go predictions

## How to Use

### Local Setup

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/Churn-Prediction-Project
cd Churn-Prediction-Project

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train the model (first time)
python churn_prediction.py

# 4. Run the web app
streamlit run app.py

# 5. Open browser to http://localhost:8501
```

# Author

Lakshay Tyagi | follow on linkedin: Lakshay Tyagi
