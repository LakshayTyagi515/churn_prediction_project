"""
Backend for Churn Prediction - Simplified Version
"""

import pickle
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Load the trained model
try:
    with open('churn_model.pkl', 'rb' , encoding = "UTF-8") as f:
        model = pickle.load(f)
    print("✓ Model loaded successfully")
except FileNotFoundError:
    print("❌ Error: churn_model.pkl not found. Run churn_prediction.py first.")
    model = None


def predict_churn(customer_data):
    """
    Predict if a customer will churn
    """
    
    if model is None:
        return {
            'will_churn': 'Error',
            'probability': 0,
            'error': 'Model not loaded. Run churn_prediction.py first.'
        }
    
    try:
        # Create DataFrame
        df = pd.DataFrame([customer_data])
        
        # Standardize 'No internet service' and 'No phone service' to 'No'
        replacements = {'No internet service': 'No', 'No phone service': 'No'}
        for col in df.columns:
            if col in df.columns and df[col].dtype == 'object':
                df[col] = df[col].replace(replacements)
        
        # Ensure numeric columns
        numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Create tenure_group
        if 'tenure' in df.columns:
            df['tenure_group'] = pd.cut(df['tenure'], 
                                        bins=[0, 12, 24, 48, np.inf],
                                        labels=['0-12 months', '13-24 months', 
                                               '25-48 months', '49+ months'],
                                        include_lowest=True)
        
        # Create avg_monthly_spend
        if 'TotalCharges' in df.columns and 'tenure' in df.columns:
            df['avg_monthly_spend'] = df.apply(
                lambda row: row['TotalCharges'] / row['tenure'] if row['tenure'] > 0 else row.get('MonthlyCharges', 0),
                axis=1
            )
        
        # Get categorical columns
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        # One-hot encode with drop_first=True (same as training)
        df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
        
        # Get expected feature count from model
        n_features = model.n_features_in_
        
        # Get feature names if available
        if hasattr(model, 'feature_names_in_'):
            expected_features = list(model.feature_names_in_)
            
            # Add missing features with 0
            for feature in expected_features:
                if feature not in df_encoded.columns:
                    df_encoded[feature] = 0
            
            # Select only expected features in correct order
            df_final = df_encoded[expected_features]
        else:
            # If feature names not available, just use first n features
            available_cols = df_encoded.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(available_cols) < n_features:
                # Pad with zeros
                while len(available_cols) < n_features:
                    new_col = f'missing_feature_{len(available_cols)}'
                    df_encoded[new_col] = 0
                    available_cols.append(new_col)
            
            df_final = df_encoded[available_cols[:n_features]]
        
        # Make prediction
        prediction = model.predict(df_final)[0]
        probability = model.predict_proba(df_final)[0][1]
        
        return {
            'will_churn': 'Yes' if prediction == 1 else 'No',
            'probability': round(probability * 100, 2),
            'error': None
        }
    
    except Exception as e:
        return {
            'will_churn': 'Error',
            'probability': 0,
            'error': f'Prediction error: {str(e)}'
        }


def get_churn_risk_level(probability):
    """Convert probability to risk level"""
    if probability < 25:
        return "🟢 LOW RISK"
    elif probability < 50:
        return "🟡 MEDIUM RISK"
    else:
        return "🔴 HIGH RISK"
