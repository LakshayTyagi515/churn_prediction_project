"""
Streamlit Web App for Customer Churn Prediction
Interactive UI to predict customer churn
"""

import streamlit as st
import pandas as pd
from backend import predict_churn, get_churn_risk_level
import os

# Page configuration
st.set_page_config(
    page_title="Churn Predictor",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

# Title and header
st.title("📱 Telecom Customer Churn Prediction")
st.markdown("---")
st.write("**Predict if a customer will churn with our ML model (82% Accuracy)**")

# Sidebar for user input
st.sidebar.header("📋 Enter Customer Details")

col1, col2 = st.sidebar.columns(2)

with col1:
    tenure = st.number_input("Tenure (months)", min_value=0, max_value=72, value=12)
    monthly_charges = st.number_input("Monthly Charges ($)", min_value=20.0, max_value=120.0, value=65.0)

with col2:
    total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=780.0)
    
st.sidebar.markdown("---")

contract = st.sidebar.selectbox(
    "Contract Type",
    ["Month-to-month", "One year", "Two year"],
    help="Longer contracts have lower churn"
)

internet_service = st.sidebar.selectbox(
    "Internet Service",
    ["DSL", "Fiber optic", "No internet service"],
    help="Service type affects churn risk"
)

payment_method = st.sidebar.selectbox(
    "Payment Method",
    ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
    help="Automatic payments reduce churn"
)

st.sidebar.markdown("---")
st.sidebar.subheader("Optional Services")

col1, col2 = st.sidebar.columns(2)

with col1:
    phone_service = st.selectbox("Phone Service", ["Yes", "No"], key="phone")
    online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"], key="security")
    online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"], key="backup")
    device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"], key="device")

with col2:
    tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"], key="support")
    streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"], key="tv")
    streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"], key="movies")

st.sidebar.markdown("---")

# Create customer data dictionary
customer_data = {
    'tenure': tenure,
    'MonthlyCharges': monthly_charges,
    'TotalCharges': total_charges,
    'Contract': contract,
    'InternetService': internet_service,
    'PaymentMethod': payment_method,
    'PhoneService': phone_service,
    'OnlineSecurity': online_security,
    'OnlineBackup': online_backup,
    'DeviceProtection': device_protection,
    'TechSupport': tech_support,
    'StreamingTV': streaming_tv,
    'StreamingMovies': streaming_movies,
}

# Prediction section
st.sidebar.markdown("---")
if st.sidebar.button("🔮 PREDICT CHURN", use_container_width=True, key="predict_btn"):
    
    result = predict_churn(customer_data)
    
    if result['error']:
        st.sidebar.error(f"❌ {result['error']}")
    else:
        # Display result
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Will Churn?", result['will_churn'])
        
        with col2:
            st.metric("Churn Probability", f"{result['probability']}%")
        
        with col3:
            st.metric("Risk Level", get_churn_risk_level(result['probability']))
        
        st.markdown("---")
        
        # Detailed result box
        if result['will_churn'] == 'Yes':
            st.error(f"""
            ### ⚠️ HIGH CHURN RISK DETECTED
            
            This customer has a **{result['probability']}% probability** of churning.
            
            **Recommended Actions:**
            - 🎁 Offer retention discount (15-20% off)
            - 📞 Proactive customer service outreach
            - 📦 Upgrade service packages
            - 🔄 Contract conversion offer
            """)
        else:
            st.success(f"""
            ### ✅ LOW CHURN RISK
            
            This customer has a **{result['probability']}% probability** of churning.
            
            **Recommended Actions:**
            - 📈 Opportunity for upsell
            - 💰 Cross-sell additional services
            - 🎖️ Loyalty rewards program
            - 📧 Continue regular engagement
            """)

# Main content area - Visualizations
st.markdown("---")
st.header("📊 Historical Analysis & Insights")

# Check if visualization files exist
image_files = [
    'images/01_churn_by_contract.png',
    'images/02_tenure_distribution.png',
    'images/03_churn_by_service.png',
    'images/04_correlation_heatmap.png',
    'images/05_confusion_matrix.png',
    'images/06_churn_drivers.png'
]

missing_files = [f for f in image_files if not os.path.exists(f)]

if missing_files:
    st.warning(f"""
    ⚠️ Missing visualization files: {', '.join(missing_files)}
    
    Run `python churn_prediction.py` first to generate visualizations.
    """)
else:
    # Display visualizations in grid
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Churn by Contract Type")
        st.image('images/01_churn_by_contract.png', use_column_width=True)
        st.caption("Month-to-month contracts have 42.71% churn rate (15x higher than 2-year)")
        
        st.subheader("Churn by Service & Payment")
        st.image('images/03_churn_by_service.png', use_column_width=True)
        st.caption("Electronic check payments and fiber optic show higher churn")
    
    with col2:
        st.subheader("Tenure Impact on Churn")
        st.image('images/02_tenure_distribution.png', use_column_width=True)
        st.caption("Churned customers average 17.9 months vs 32.4 for retained")
        
        st.subheader("Model Performance")
        st.image('images/05_confusion_matrix.png', use_column_width=True)
        st.caption("Model achieves 82.15% accuracy in churn prediction")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Feature Correlations")
        st.image('images/04_correlation_heatmap.png', use_column_width=True)
        st.caption("Tenure shows strong negative correlation with churn")
    
    with col2:
        st.subheader("Top Churn Drivers")
        st.image('images/06_churn_drivers.png', use_column_width=True)
        st.caption("Month-to-month contracts are the strongest churn driver")

# Key insights
st.markdown("---")
st.header("🎯 Key Insights from Analysis")

col1, col2, col3 = st.columns(3)

with col1:
    st.info("""
    ### 📌 Contract Impact
    - **Month-to-month:** 42.71% churn
    - **One year:** 11.27% churn
    - **Two year:** 2.83% churn
    
    **Action:** Convert to annual plans
    """)

with col2:
    st.info("""
    ### ⏱️ Tenure Critical
    - **First 12 months:** Highest churn
    - **After 24 months:** Churn drops 80%
    
    **Action:** Focus retention efforts on new customers
    """)

with col3:
    st.info("""
    ### 💳 Payment Method
    - Auto-pay: 20-30% lower churn
    - Manual payment: Higher churn
    
    **Action:** Incentivize automatic payments
    """)

# Retention strategies
st.markdown("---")
st.header("💡 Recommended Retention Strategies")

with st.expander("📌 Strategy 1: Contract Conversion Campaign (HIGHEST PRIORITY)", expanded=False):
    st.markdown("""
    **Target:** Month-to-month customers (especially 0-12 months)
    
    **Why:** Month-to-month contracts have 42.71% churn rate (15x higher than 2-year)
    
    **Tactics:**
    - Offer 15-20% discount for switching to 12-month contract
    - Early outreach at 1-3 month mark
    - "Lock in savings" messaging
    - One-click upgrade in portal
    
    **Expected Impact:**
    - 25-35% churn reduction in M2M segment
    - 200-300 retained customers annually
    - 300-400% ROI
    
    **Timeline:** Week 1-4 launch, then scale
    """)

with st.expander("📌 Strategy 2: Fiber Optic Service Excellence Program", expanded=False):
    st.markdown("""
    **Target:** Fiber optic internet customers
    
    **Why:** Fiber customers show elevated churn (service quality or value issues)
    
    **Tactics:**
    - Proactive service monitoring with automatic credits for outages
    - "Satisfaction guarantee" - 30-day refund option
    - Free equipment upgrades every 2 years
    - "Fiber Plus" loyalty rewards program
    
    **Expected Impact:**
    - 15-20% churn reduction in fiber segment
    - +10-15 point NPS improvement
    - 10-15% upgrade rate to higher speeds
    
    **Timeline:** Month 1-3 implementation
    """)

with st.expander("📌 Strategy 3: Auto-Pay Incentive Program (QUICK WIN)", expanded=False):
    st.markdown("""
    **Target:** Manual payment method customers
    
    **Why:** Auto-pay adoption correlates with 20-30% higher retention
    
    **Tactics:**
    - 5-10% discount for switching to automatic payment
    - "1 Month Free" for auto-pay adoption
    - Paperless billing incentive (additional 3% discount)
    - Flexible billing date selection
    
    **Expected Impact:**
    - 25-35% auto-pay adoption increase
    - 10-15% churn reduction in this segment
    - -20% fewer billing-related support calls
    - Improved cash flow predictability
    
    **Timeline:** Week 1 launch (IMMEDIATE), high ROI
    """)

# Model Performance section
st.markdown("---")
st.header("📈 Model Performance Metrics")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Accuracy", "82.15%", help="Overall correctness of predictions")

with col2:
    st.metric("Precision", "65.32%", help="When model predicts churn, correctness %")

with col3:
    st.metric("Recall", "58.47%", help="% of actual churners caught")

with col4:
    st.metric("F1 Score", "0.6178", help="Balanced metric")

st.info("""
**How to interpret:**
- **Accuracy 82%:** Model gets 82 out of 100 predictions right
- **Precision 65%:** When we predict churn, it's correct 65% of the time (good for targeting offers)
- **Recall 58%:** We catch 58% of customers who will actually churn
- **F1:** Balanced view of precision and recall trade-off
""")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; padding: 20px;'>
    <p>🎓 Customer Churn Prediction ML Model</p>
    <p>Built with Logistic Regression | Data-Driven Business Insights</p>
    <p>© 2024 | Telco Customer Analytics</p>
</div>
""", unsafe_allow_html=True)