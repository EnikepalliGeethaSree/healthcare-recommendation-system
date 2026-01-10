"""
Script 5: SHAP Explainer for Model Interpretability
- Generate SHAP values for feature importance
- Handle missing values
- Create explanation utilities
"""

import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import pickle
import os

# -----------------------------
# Surrogate model from Spark features
# -----------------------------
def train_surrogate_model_from_spark():
    """
    Train a surrogate RandomForestClassifier in Python
    using the same features used in Spark RandomForest.
    Handles missing values and scaling.
    """
    print("\nPreparing surrogate model from Spark-trained features...")
    
    # Load processed data
    df = pd.read_parquet('data/processed/healthcare_cleaned.parquet')
    
    # Feature columns
    feature_cols = ['age', 'bmi', 'bp_systolic', 'glucose', 'cholesterol', 
                    'sleep_hours', 'alcohol_units']
    
    # Encode categorical features
    df['sex_num'] = df['sex'].map({'Male': 1, 'Female': 0})
    df['stress_num'] = df['stress_level'].map({'Low': 0, 'Medium': 1, 'High': 2})
    df['activity_num'] = df['activity_level'].map({'Sedentary':0,'Light':1,'Moderate':2,'Active':3})
    df['smoking_num'] = df['smoking'].map({'Never':0,'Former':1,'Current':2})
    
    feature_cols_all = feature_cols + ['sex_num', 'stress_num', 'activity_num', 'smoking_num']
    
    X = df[feature_cols_all].values
    y = df['disease_label'].values
    
    # Impute missing values
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    
    # Train RandomForest
    rf_model = RandomForestClassifier(n_estimators=150, max_depth=15, 
                                      random_state=42, n_jobs=-1)
    rf_model.fit(X_scaled, y)
    
    print(f"Surrogate model accuracy: {rf_model.score(X_scaled, y):.4f}")
    
    # Save models
    os.makedirs('models', exist_ok=True)
    with open('models/shap_surrogate_model.pkl','wb') as f:
        pickle.dump(rf_model, f)
    with open('models/shap_scaler.pkl','wb') as f:
        pickle.dump(scaler, f)
    with open('models/shap_imputer.pkl','wb') as f:
        pickle.dump(imputer, f)
    with open('models/feature_names.pkl','wb') as f:
        pickle.dump(feature_cols_all, f)
    
    print("✓ Surrogate model trained from Spark features and saved")
    
    return rf_model, scaler, X_scaled, feature_cols_all, imputer

# -----------------------------
# Generate SHAP values
# -----------------------------
def generate_shap_values():
    print("\nGenerating SHAP values...")
    
    rf_model, scaler, X_scaled, feature_names, imputer = train_surrogate_model_from_spark()
    
    # Create SHAP explainer
    explainer = shap.TreeExplainer(rf_model)
    
    # Subset for speed
    X_subset = X_scaled[:1000]
    
    # Apply imputer and scaler
    X_subset_imputed = imputer.transform(X_subset)
    X_subset_scaled = scaler.transform(X_subset_imputed)
    
    shap_values = explainer.shap_values(X_subset_scaled)
    
    # Save SHAP explainer
    with open('models/shap_explainer.pkl', 'wb') as f:
        pickle.dump(explainer, f)
    
    print("✓ SHAP explainer saved")
    
    # Summary plot
    print("\nGenerating SHAP summary plot...")
    plt.figure(figsize=(12,8))
    shap_values_plot = shap_values[0] if isinstance(shap_values, list) else shap_values
    shap.summary_plot(shap_values_plot, X_subset_scaled, 
                     feature_names=feature_names, show=False)
    
    plt.title('SHAP Feature Importance', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    os.makedirs('data/processed', exist_ok=True)
    plt.savefig('data/processed/shap_summary.png', dpi=300, bbox_inches='tight')
    print("✓ SHAP summary plot saved to data/processed/shap_summary.png")
    plt.close()
    
    return explainer

# -----------------------------
# Explain single prediction
# -----------------------------
def explain_single_prediction(patient_data):
    """Generate SHAP explanation for a single prediction"""
    
    with open('models/shap_surrogate_model.pkl','rb') as f:
        rf_model = pickle.load(f)
    with open('models/shap_scaler.pkl','rb') as f:
        scaler = pickle.load(f)
    with open('models/shap_imputer.pkl','rb') as f:
        imputer = pickle.load(f)
    with open('models/shap_explainer.pkl','rb') as f:
        explainer = pickle.load(f)
    with open('models/feature_names.pkl','rb') as f:
        feature_names = pickle.load(f)
    
    X_input = np.array(patient_data).reshape(1,-1)
    X_input_imputed = imputer.transform(X_input)
    X_scaled = scaler.transform(X_input_imputed)
    
    shap_values = explainer.shap_values(X_scaled)
    prediction = rf_model.predict(X_scaled)[0]
    prediction_proba = rf_model.predict_proba(X_scaled)[0]
    
    shap_vals = shap_values[int(prediction)] if isinstance(shap_values, list) else shap_values
    contributions = dict(zip(feature_names, shap_vals[0]))
    
    sorted_contributions = dict(sorted(contributions.items(),
                                       key=lambda x: abs(x[1]), reverse=True))
    
    return {
        'prediction': int(prediction),
        'probabilities': prediction_proba.tolist(),
        'contributions': sorted_contributions
    }

# -----------------------------
# Format contribution text
# -----------------------------
def format_contribution_text(contributions, top_n=5):
    feature_labels = {
        'age': 'Age',
        'bmi': 'BMI',
        'bp_systolic': 'Blood Pressure',
        'glucose': 'Glucose',
        'cholesterol': 'Cholesterol',
        'sleep_hours': 'Sleep Hours',
        'alcohol_units': 'Alcohol Consumption',
        'sex_num': 'Sex',
        'stress_num': 'Stress Level',
        'activity_num': 'Activity Level',
        'smoking_num': 'Smoking Status'
    }
    
    top_features = list(contributions.items())[:top_n]
    explanation_parts = []
    for feature, value in top_features:
        label = feature_labels.get(feature, feature)
        direction = "increases" if value > 0 else "decreases"
        explanation_parts.append(f"• {label} {direction} risk (impact: {abs(value):.3f})")
    return "\n".join(explanation_parts)

# -----------------------------
# Main execution
# -----------------------------
if __name__ == "__main__":
    try:
        explainer = generate_shap_values()
        print("\n✓ SHAP analysis complete!")
        
        # Test single prediction
        print("\nTesting single prediction explanation...")
        test_patient = [65, 32.5, 155, 180, 260, 5.5, 12, 1, 2, 0, 2]  # High-risk patient
        
        explanation = explain_single_prediction(test_patient)
        print(f"\nPredicted class: {explanation['prediction']}")
        print(f"Prediction probabilities: {explanation['probabilities'][:3]}")
        print("\nTop feature contributions:")
        print(format_contribution_text(explanation['contributions']))
        
    except Exception as e:
        print(f"Error in SHAP analysis: {e}")
        raise
