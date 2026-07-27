"""
Customer Churn Prediction Project
Using Logistic Regression to predict customer churn and identify retention strategies
Dataset: Telco Customer Churn (Kaggle)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, classification_report)

import warnings
warnings.filterwarnings('ignore')

# Set style for visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

print("="*80)
print("CUSTOMER CHURN PREDICTION PROJECT")
print("="*80)

# =============================================================================
# STEP 1: DOWNLOAD AND LOAD DATASET
# =============================================================================
print("\n[STEP 1] Loading Dataset...")

# Download from Kaggle (ensure you have kaggle API configured)
# For this demo, we'll load from a direct URL
try:
    url = "https://raw.githubusercontent.com/IBM/telco-customer-churn/main/data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    df = pd.read_csv('dataset/WA_Fn-UseC_-Telco-Customer-Churn.csv')
    print(f"✓ Dataset loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
except:
    print("Note: Could not download from URL. Using kaggle CLI alternative:")
    print("Run: kaggle datasets download -d blastchar/telco-customer-churn")
    print("Then load with: df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')")
    exit()

print(f"\nFirst few rows:")
print(df.head())
print(f"\nDataset Info:")
print(df.info())
print(f"\nMissing values:\n{df.isnull().sum()}")

# =============================================================================
# STEP 2: DATA CLEANING
# =============================================================================
print("\n" + "="*80)
print("[STEP 2] Data Cleaning")
print("="*80)

# Drop customerID
if 'customerID' in df.columns:
    df = df.drop('customerID', axis=1)
    print("✓ Removed customerID column")

# Convert TotalCharges to numeric
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
print(f"✓ Converted TotalCharges to numeric")

# Handle blank/NaN values in TotalCharges (typically new customers)
print(f"  - NaN values in TotalCharges: {df['TotalCharges'].isnull().sum()}")
df['TotalCharges'] = df['TotalCharges'].fillna(0)
print(f"  - Filled NaN with 0 (new customers)")

# Standardize 'No internet service' / 'No phone service'
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    if df[col].dtype == 'object':
        df[col] = df[col].replace(['No internet service', 'No phone service'], 'No')
print(f"✓ Standardized service 'No' categories")

# Check for duplicates
duplicates = df.duplicated().sum()
if duplicates > 0:
    df = df.drop_duplicates()
    print(f"✓ Removed {duplicates} duplicate rows")
else:
    print(f"✓ No duplicate rows found")

print(f"\nCleaned dataset shape: {df.shape}")

# =============================================================================
# STEP 3: EXPLORATORY DATA ANALYSIS (EDA)
# =============================================================================
print("\n" + "="*80)
print("[STEP 3] Exploratory Data Analysis")
print("="*80)

# Overall churn rate
churn_rate = (df['Churn'] == 'Yes').sum() / len(df) * 100
print(f"\n📊 Overall Churn Rate: {churn_rate:.2f}%")

# EDA Visualization 1: Churn by Contract Type
print("\n[EDA-1] Churn Rate by Contract Type")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

contract_churn = pd.crosstab(df['Contract'], df['Churn'], normalize='index') * 100
contract_churn.plot(kind='bar', ax=axes[0], color=['#2ecc71', '#e74c3c'])
axes[0].set_title('Churn Rate by Contract Type (%)', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Contract Type')
axes[0].set_ylabel('Percentage (%)')
axes[0].legend(['No Churn', 'Churn'], loc='upper right')
axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=0)

# Count plot
contract_counts = pd.crosstab(df['Contract'], df['Churn'])
contract_counts.plot(kind='bar', ax=axes[1], color=['#2ecc71', '#e74c3c'])
axes[1].set_title('Customer Count by Contract Type', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Contract Type')
axes[1].set_ylabel('Count')
axes[1].legend(['No Churn', 'Churn'], loc='upper right')
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=0)

plt.tight_layout()
plt.savefig('images/01_churn_by_contract.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 01_churn_by_contract.png")
plt.show()

# EDA Visualization 2: Tenure vs Churn
print("\n[EDA-2] Tenure Distribution by Churn Status")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Distribution
churned = df[df['Churn'] == 'Yes']['tenure']
retained = df[df['Churn'] == 'No']['tenure']

axes[0].hist([retained, churned], bins=30, label=['Retained', 'Churned'], 
             color=['#2ecc71', '#e74c3c'], alpha=0.7)
axes[0].set_title('Tenure Distribution by Churn Status', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Tenure (months)')
axes[0].set_ylabel('Frequency')
axes[0].legend()
axes[0].grid(alpha=0.3)

# Box plot
df.boxplot(column='tenure', by='Churn', ax=axes[1])
axes[1].set_title('Tenure by Churn Status', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Churn Status')
axes[1].set_ylabel('Tenure (months)')
plt.suptitle('')  # Remove default title

plt.tight_layout()
plt.savefig('images/02_tenure_distribution.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 02_tenure_distribution.png")
plt.show()

# EDA Visualization 3: Churn by Internet Service and Payment Method
print("\n[EDA-3] Churn by Internet Service & Payment Method")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Internet Service
internet_churn = pd.crosstab(df['InternetService'], df['Churn'], normalize='index') * 100
internet_churn.plot(kind='bar', ax=axes[0], color=['#2ecc71', '#e74c3c'])
axes[0].set_title('Churn Rate by Internet Service (%)', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Internet Service')
axes[0].set_ylabel('Percentage (%)')
axes[0].legend(['No Churn', 'Churn'])
axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45)

# Payment Method
payment_churn = pd.crosstab(df['PaymentMethod'], df['Churn'], normalize='index') * 100
payment_churn.plot(kind='bar', ax=axes[1], color=['#2ecc71', '#e74c3c'])
axes[1].set_title('Churn Rate by Payment Method (%)', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Payment Method')
axes[1].set_ylabel('Percentage (%)')
axes[1].legend(['No Churn', 'Churn'])
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45)

plt.tight_layout()
plt.savefig('images/03_churn_by_service.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 03_churn_by_service.png")
plt.show()

# EDA Visualization 4: Correlation Heatmap
print("\n[EDA-4] Correlation Heatmap (Numeric Features)")
numeric_df = df[['tenure', 'MonthlyCharges', 'TotalCharges']].copy()
numeric_df['Churn_numeric'] = (df['Churn'] == 'Yes').astype(int)

plt.figure(figsize=(8, 6))
correlation = numeric_df.corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0, 
            square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Correlation Heatmap - Numeric Features vs Churn', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('images/04_correlation_heatmap.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 04_correlation_heatmap.png")
plt.show()

# Print key insights
print("\n📈 KEY EDA INSIGHTS:")
print(f"  - Month-to-Month contracts have {contract_churn.loc['Month-to-month', 'Yes']:.1f}% churn rate")
print(f"  - One/Two-Year contracts have lower churn rates (~{contract_churn.loc['One year', 'Yes']:.1f}% & {contract_churn.loc['Two year', 'Yes']:.1f}%)")
print(f"  - Avg tenure for churned customers: {churned.mean():.1f} months")
print(f"  - Avg tenure for retained customers: {retained.mean():.1f} months")

# =============================================================================
# STEP 4: FEATURE ENGINEERING
# =============================================================================
print("\n" + "="*80)
print("[STEP 4] Feature Engineering")
print("="*80)

df_model = df.copy()

# Create tenure_group bucket
print("\n[FE-1] Creating tenure groups...")
df_model['tenure_group'] = pd.cut(df_model['tenure'], 
                                   bins=[0, 12, 24, 48, np.inf],
                                   labels=['0-12 months', '13-24 months', 
                                          '25-48 months', '49+ months'],
                                   include_lowest=True)
print("✓ Created tenure_group feature")

# Create avg_monthly_spend (handle divide-by-zero)
print("[FE-2] Creating avg_monthly_spend feature...")
df_model['avg_monthly_spend'] = df_model.apply(
    lambda row: row['TotalCharges'] / row['tenure'] if row['tenure'] > 0 else 0,
    axis=1
)
print("✓ Created avg_monthly_spend feature")

# Identify categorical columns for encoding
categorical_features = df_model.select_dtypes(include=['object' , 'category']).columns.tolist()
if 'Churn' in categorical_features:
    categorical_features.remove('Churn')

print(f"\n[FE-3] One-hot encoding {len(categorical_features)} categorical features...")

# One-hot encode categorical features
df_encoded = pd.get_dummies(df_model, columns=categorical_features, drop_first=True)
print(f"✓ One-hot encoding complete. New shape: {df_encoded.shape}")

# Encode target variable
df_encoded['Churn'] = (df_encoded['Churn'] == 'Yes').astype(int)
print("✓ Encoded Churn as binary (Yes=1, No=0)")

# =============================================================================
# STEP 5: MODEL BUILDING
# =============================================================================
print("\n" + "="*80)
print("[STEP 5] Model Building & Training")
print("="*80)

# Separate features and target
X = df_encoded.drop('Churn', axis=1)
y = df_encoded['Churn']

print(f"\nFeature matrix shape: {X.shape}")
print(f"Target distribution:\n{y.value_counts()}")
print(f"Class imbalance ratio: {(y==0).sum()} : {(y==1).sum()} "
      f"(Retained : Churned)")

# Train-test split with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\n✓ Train-test split (80/20) with stratification:")
print(f"  - Training set: {X_train.shape[0]} samples")
print(f"  - Test set: {X_test.shape[0]} samples")

# Train Logistic Regression
print("\n[MODEL] Training Logistic Regression...")
lr_model = LogisticRegression(
    max_iter=1000,
    random_state=42,
    class_weight='balanced'  # Handle class imbalance
)
lr_model.fit(X_train, y_train)
print(X_train.dtypes)
print("✓ Model training complete!")

# =============================================================================
# STEP 6: MODEL EVALUATION
# =============================================================================
print("\n" + "="*80)
print("[STEP 6] Model Evaluation")
print("="*80)

# Make predictions
y_pred = lr_model.predict(X_test)
y_pred_proba = lr_model.predict_proba(X_test)[:, 1]

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n📊 CLASSIFICATION METRICS:")
print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"  Precision: {precision:.4f} ({precision*100:.2f}%)")
print(f"  Recall:    {recall:.4f} ({recall*100:.2f}%)")
print(f"  F1 Score:  {f1:.4f}")

print("\n📋 CLASSIFICATION REPORT:")
print(classification_report(y_test, y_pred, 
                           target_names=['Retained', 'Churned'],
                           digits=4))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\n🔲 CONFUSION MATRIX:")
print(f"  True Negatives:  {cm[0,0]}")
print(f"  False Positives: {cm[0,1]}")
print(f"  False Negatives: {cm[1,0]}")
print(f"  True Positives:  {cm[1,1]}")

# Visualize Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Retained', 'Churned'],
            yticklabels=['Retained', 'Churned'],
            cbar_kws={'label': 'Count'})
plt.title('Confusion Matrix - Churn Prediction', fontsize=12, fontweight='bold')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('images/05_confusion_matrix.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: 05_confusion_matrix.png")
plt.show()

# =============================================================================
# STEP 7: IDENTIFY CHURN DRIVERS
# =============================================================================
print("\n" + "="*80)
print("[STEP 7] Churn Drivers Analysis (Model Coefficients)")
print("="*80)

# Extract coefficients and feature names
coefficients = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': lr_model.coef_[0]
})

# Sort by absolute coefficient value
coefficients['Abs_Coefficient'] = coefficients['Coefficient'].abs()
coefficients_sorted = coefficients.sort_values('Abs_Coefficient', ascending=False)

print("\n🎯 TOP 15 CHURN DRIVERS (by absolute coefficient magnitude):")
print("-" * 70)
for idx, (_, row) in enumerate(coefficients_sorted.head(15).iterrows(), 1):
    direction = "↑ INCREASES" if row['Coefficient'] > 0 else "↓ DECREASES"
    print(f"{idx:2}. {row['Feature']:40} {direction:15} churn "
          f"(coef: {row['Coefficient']:+.4f})")

# Visualize top churn drivers
fig, ax = plt.subplots(figsize=(10, 8))
top_drivers = coefficients_sorted.head(12)
colors = ['#e74c3c' if x > 0 else '#2ecc71' for x in top_drivers['Coefficient']]

ax.barh(range(len(top_drivers)), top_drivers['Coefficient'], color=colors)
ax.set_yticks(range(len(top_drivers)))
ax.set_yticklabels(top_drivers['Feature'])
ax.set_xlabel('Coefficient Value', fontsize=11, fontweight='bold')
ax.set_title('Top 12 Churn Drivers (Red = Increases Churn, Green = Decreases)', 
             fontsize=12, fontweight='bold')
ax.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
plt.tight_layout()
plt.savefig('images/06_churn_drivers.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: 06_churn_drivers.png")
plt.show()

# =============================================================================
# STEP 8: RETENTION STRATEGIES
# =============================================================================
print("\n" + "="*80)
print("[STEP 8] RETENTION STRATEGIES Based on Churn Drivers")
print("="*80)

strategies = """
🎯 RETENTION STRATEGY RECOMMENDATIONS
====================================

Based on the analysis of the top churn drivers, we recommend:

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

STRATEGY 1: CONTRACT CONVERSION CAMPAIGN
────────────────────────────────────────
📍 Target: Month-to-month customers (especially in first 12 months)
💡 Insight: Month-to-month contracts have 2-3x higher churn rate than long-term contracts
🎁 Tactic:
   • Offer 15-20% discount for switching to 12-month contract
   • Early retention reach out at 1-3 month mark
   • Highlight commitment benefits & cost savings
📊 Expected Impact: 
   • Reduce churn in this segment by 25-35%
   • ~200-300 additional retained customers (if 20% conversion)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

STRATEGY 2: FIBER OPTIC SERVICE OPTIMIZATION
──────────────────────────────────────────────
📍 Target: Fiber optic internet customers (if showing high churn)
💡 Insight: Fiber customers often churn due to service quality or billing issues
🎁 Tactic:
   • Implement proactive service quality monitoring
   • Offer "service satisfaction guarantee" rebate program
   • Provide free equipment upgrades
   • Create fiber-specific loyalty rewards
📊 Expected Impact:
   • Reduce Fiber segment churn by 15-20%
   • Increased customer lifetime value through service bundling

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

STRATEGY 3: PAYMENT METHOD & BILLING OPTIMIZATION
──────────────────────────────────────────────────
📍 Target: Customers using automatic bank transfer / mail check (if high churn)
💡 Insight: Electronic payment adoption correlates with higher retention
🎁 Tactic:
   • Incentivize paperless billing (automatic payment) with 5% discount
   • Simplify billing experience with clear, itemized statements
   • Offer flexible payment dates/amounts
   • Autopay adoption bonus: 1-month free premium service
📊 Expected Impact:
   • Increase autopay adoption by 20%
   • Reduce churn by 10-15% in this segment

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

IMPLEMENTATION ROADMAP:
─────────────────────
Month 1: Launch Strategy 1 (Contract Conversion) - Quick win, high ROI
Month 2: Monitor results, refine messaging, launch Strategy 3 (Billing)
Month 3: Launch Strategy 2 (Service Optimization), measure all impacts
Month 4+: Scale winning tactics, A/B test new retention offers

SUCCESS METRICS TO TRACK:
────────────────────────
✓ Churn rate by cohort (contract type, service type, tenure group)
✓ Campaign conversion rates (% switching to longer contracts)
✓ Customer acquisition cost (CAC) payback period
✓ Lifetime value (LTV) improvement
✓ Segment-specific retention uplift
"""

print(strategies)

# Save strategies to file
with open('retention_strategies.txt', 'w', encoding = "UTF-8") as f:
    f.write(strategies)
print("\n✓ Saved retention strategies to: retention_strategies.txt")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "="*80)
print("PROJECT SUMMARY & OUTPUT FILES")
print("="*80)

summary = f"""
✅ PROJECT COMPLETION SUMMARY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📊 DATA OVERVIEW
  • Dataset: Telco Customer Churn (Kaggle)
  • Total Customers: {len(df)}
  • Features: {df.shape[1]} columns
  • Churn Rate: {churn_rate:.2f}%

🧹 DATA CLEANING
  ✓ Converted TotalCharges to numeric
  ✓ Handled blank values (filled with 0)
  ✓ Standardized "No service" categories
  ✓ Removed duplicates
  ✓ Final dataset: {df.shape[0]} rows × {df.shape[1]} columns

📈 EDA VISUALIZATIONS CREATED
  ✓ 01_churn_by_contract.png - Churn rate by contract type
  ✓ 02_tenure_distribution.png - Tenure vs churn correlation
  ✓ 03_churn_by_service.png - Internet service & payment method analysis
  ✓ 04_correlation_heatmap.png - Feature correlations
  ✓ 05_confusion_matrix.png - Model performance matrix
  ✓ 06_churn_drivers.png - Top churn drivers ranked

🔧 FEATURE ENGINEERING
  ✓ Created tenure_group (0-12, 13-24, 25-48, 49+ months)
  ✓ Created avg_monthly_spend (TotalCharges / tenure)
  ✓ One-hot encoded all categorical features
  ✓ Final features: {X.shape[1]}

🤖 MODEL TRAINING
  • Algorithm: Logistic Regression (with class_weight='balanced')
  • Training samples: {X_train.shape[0]}
  • Test samples: {X_test.shape[0]}
  • Train-test split: 80/20 (stratified by churn)

📊 MODEL PERFORMANCE
  • Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)
  • Precision: {precision:.4f} ({precision*100:.2f}%)
  • Recall:    {recall:.4f} ({recall*100:.2f}%)
  • F1 Score:  {f1:.4f}

🎯 TOP 3 CHURN DRIVERS
  1. {coefficients_sorted.iloc[0]['Feature']} (coef: {coefficients_sorted.iloc[0]['Coefficient']:+.4f})
  2. {coefficients_sorted.iloc[1]['Feature']} (coef: {coefficients_sorted.iloc[1]['Coefficient']:+.4f})
  3. {coefficients_sorted.iloc[2]['Feature']} (coef: {coefficients_sorted.iloc[2]['Coefficient']:+.4f})

💡 RETENTION STRATEGIES
  ✓ Strategy 1: Contract Conversion Campaign (Month-to-month → Long-term)
  ✓ Strategy 2: Fiber Optic Service Optimization
  ✓ Strategy 3: Payment Method & Billing Optimization

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📁 OUTPUT FILES GENERATED:
  ✓ 01_churn_by_contract.png
  ✓ 02_tenure_distribution.png
  ✓ 03_churn_by_service.png
  ✓ 04_correlation_heatmap.png
  ✓ 05_confusion_matrix.png
  ✓ 06_churn_drivers.png
  ✓ retention_strategies.txt

✨ Next Steps:
  1. Review all visualizations
  2. Implement retention strategies
  3. Set up monitoring dashboard for churn metrics
  4. Schedule A/B tests for top-performing tactics
  5. Create README.md for GitHub submission
"""

print(summary)

# Save summary to file
with open('project_summary.txt', 'w', encoding = "UTF-8") as f:
    f.write(summary)

print("\n✓ Project complete! All outputs saved")

with open('churn_model.pkl', 'wb') as f:
    pickle.dump(lr_model, f)
print("✓ Model saved to churn_model.pkl")
