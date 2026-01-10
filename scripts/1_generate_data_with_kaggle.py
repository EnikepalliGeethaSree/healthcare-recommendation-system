"""
1_generate_data_REVISED.py - Evidence-Based Label Assignment
Implements clinical guideline-based diet and lifestyle recommendations
"""

import pandas as pd
import numpy as np
import os
import warnings

warnings.filterwarnings('ignore')
np.random.seed(42)

# ============================================================
# EVIDENCE-BASED RISK SCORING FUNCTIONS
# ============================================================

def calculate_metabolic_risk_score(age, bmi, glucose, bp, cholesterol, activity, smoking):
    """
    Calculate metabolic syndrome risk score (NCEP ATP III criteria)
    Returns: risk_score (0-100)
    """
    risk_score = 0
    
    # Age factor
    if age > 65:
        risk_score += 20
    elif age > 45:
        risk_score += 10
    
    # BMI (WHO obesity classification)
    if bmi >= 35:
        risk_score += 20
    elif bmi >= 30:
        risk_score += 15
    elif bmi >= 25:
        risk_score += 5
    
    # Glucose (ADA diabetes criteria)
    if glucose >= 200:
        risk_score += 20
    elif glucose >= 140:
        risk_score += 12
    elif glucose >= 100:
        risk_score += 5
    
    # Blood Pressure (JNC 8 guidelines)
    if bp >= 160:
        risk_score += 15
    elif bp >= 140:
        risk_score += 10
    elif bp >= 130:
        risk_score += 5
    
    # Cholesterol (ATP III)
    if cholesterol >= 280:
        risk_score += 15
    elif cholesterol >= 240:
        risk_score += 10
    elif cholesterol >= 200:
        risk_score += 5
    
    # Activity level
    activity_map = {'Sedentary': 15, 'Light': 8, 'Moderate': 0, 'Active': -5}
    risk_score += activity_map.get(activity, 0)
    
    # Smoking
    smoking_map = {'Current': 20, 'Former': 5, 'Never': 0}
    risk_score += smoking_map.get(smoking, 0)
    
    return min(risk_score, 100)


def calculate_cardiovascular_risk_score(age, sex, bp, cholesterol, smoking, diabetes_present):
    """
    Framingham-based cardiovascular risk score
    """
    risk_score = 0
    
    # Age and sex interaction
    if sex == 'Male':
        if age > 60:
            risk_score += 15
        elif age > 45:
            risk_score += 10
    else:
        if age > 65:
            risk_score += 15
        elif age > 55:
            risk_score += 10
    
    # Blood pressure
    if bp >= 160:
        risk_score += 20
    elif bp >= 140:
        risk_score += 12
    elif bp >= 130:
        risk_score += 6
    
    # Cholesterol
    if cholesterol >= 280:
        risk_score += 18
    elif cholesterol >= 240:
        risk_score += 12
    elif cholesterol >= 200:
        risk_score += 6
    
    # Smoking multiplier
    if smoking == 'Current':
        risk_score += 25
    elif smoking == 'Former':
        risk_score += 8
    
    # Diabetes enhancer
    if diabetes_present:
        risk_score += 15
    
    return min(risk_score, 100)


def calculate_lifestyle_burden_score(sleep_hours, stress_level, alcohol_units, activity_level):
    """
    Composite lifestyle burden score
    """
    burden_score = 0
    
    # Sleep quality (U-shaped risk)
    if sleep_hours < 5:
        burden_score += 20
    elif sleep_hours < 6:
        burden_score += 12
    elif sleep_hours > 9:
        burden_score += 8
    elif 7 <= sleep_hours <= 8:
        burden_score -= 5
    
    # Stress level
    stress_map = {'High': 20, 'Medium': 10, 'Low': 0}
    burden_score += stress_map.get(stress_level, 0)
    
    # Alcohol consumption
    if alcohol_units > 14:
        burden_score += 18
    elif alcohol_units > 7:
        burden_score += 10
    elif alcohol_units > 4:
        burden_score += 4
    
    # Physical activity
    activity_map = {'Sedentary': 20, 'Light': 10, 'Moderate': 0, 'Active': -8}
    burden_score += activity_map.get(activity_level, 0)
    
    return max(burden_score, 0)


# ============================================================
# EVIDENCE-BASED LABEL ASSIGNMENT
# ============================================================

def assign_diet_plan_evidence_based(age, sex, bmi, glucose, cholesterol, bp, disease, 
                                     activity, smoking, stress, sleep_hours, alcohol_units):
    """
    Evidence-based diet plan using multi-criteria decision analysis
    Based on: AHA, ADA, DASH, Mediterranean Diet clinical trials
    """
    
    # Calculate risk scores
    metabolic_risk = calculate_metabolic_risk_score(age, bmi, glucose, bp, cholesterol, activity, smoking)
    cardio_risk = calculate_cardiovascular_risk_score(age, sex, bp, cholesterol, smoking, 
                                                       disease == 'Diabetes_Type2')
    
    # Priority 1: Active disease conditions
    if disease == 'Diabetes_Type2' or glucose >= 200:
        if bmi >= 30:
            return 'Low_Carb'
        else:
            return 'Mediterranean'
    
    if disease == 'Heart_Disease' or cardio_risk >= 60:
        if cholesterol >= 240:
            return 'Heart_Healthy'
        else:
            return 'Mediterranean'
    
    if disease == 'Hypertension' or bp >= 140:
        return 'DASH'
    
    # Priority 2: Metabolic syndrome
    if metabolic_risk >= 50:
        if bmi >= 35:
            return 'Weight_Loss'
        elif glucose >= 100:
            return 'Low_Carb'
        else:
            return 'Mediterranean'
    
    # Priority 3: Obesity
    if disease == 'Obesity' or bmi >= 30:
        if activity == 'Sedentary':
            return 'Weight_Loss'
        else:
            return 'Low_Carb'
    
    # Priority 4: Inflammatory conditions
    if disease in ['Anxiety', 'SleepDisorder'] or stress == 'High':
        return 'Anti_Inflammatory'
    
    # Priority 5: Nutritional deficiency
    if disease == 'NutritionalDeficiency' or bmi < 18.5:
        return 'Plant_Based'
    
    # Priority 6: Cholesterol management
    if disease == 'Hyperlipidemia' or cholesterol >= 240:
        return 'Heart_Healthy'
    
    # Priority 7: Prediabetes
    if disease == 'Prediabetes' or (100 <= glucose < 126):
        return 'Low_Carb'
    
    # Default
    if age < 40 and bmi < 25 and metabolic_risk < 20:
        return 'Balanced'
    else:
        return 'Mediterranean'


def assign_lifestyle_plan_evidence_based(age, sex, bmi, bp, glucose, disease, 
                                         activity, smoking, stress, sleep_hours, 
                                         alcohol_units, cholesterol):
    """
    Evidence-based lifestyle intervention
    Based on: CDC guidelines, Tobacco Cessation Guidelines, CBT-I protocols
    """
    
    lifestyle_burden = calculate_lifestyle_burden_score(sleep_hours, stress, alcohol_units, activity)
    metabolic_risk = calculate_metabolic_risk_score(age, bmi, glucose, bp, cholesterol, activity, smoking)
    
    # Priority 1: Active smoking (highest modifiable risk)
    if smoking == 'Current':
        return 'Smoking_Cessation'
    
    # Priority 2: Severe sleep disorder
    if disease == 'SleepDisorder' or sleep_hours < 5:
        return 'Sleep_Improvement'
    
    # Priority 3: High stress / mental health
    if disease == 'Anxiety' or stress == 'High':
        if sleep_hours < 6:
            return 'Sleep_Improvement'
        else:
            return 'Stress_Management'
    
    # Priority 4: Alcohol misuse
    if alcohol_units > 14:
        return 'Alcohol_Reduction'
    
    # Priority 5: Sedentary behavior
    if activity == 'Sedentary':
        if disease in ['Obesity', 'Diabetes_Type2', 'Heart_Disease']:
            return 'Active_Movement'
        elif lifestyle_burden >= 40:
            return 'Active_Movement'
    
    # Priority 6: Chronic disease management
    if disease in ['Diabetes_Type2', 'Heart_Disease', 'Hypertension', 'MetabolicSyndrome']:
        if metabolic_risk >= 60:
            return 'Preventive_Care'
        else:
            return 'Active_Movement'
    
    # Priority 7: Multiple lifestyle risks
    if lifestyle_burden >= 35:
        if stress == 'High':
            return 'Stress_Management'
        elif sleep_hours < 6.5:
            return 'Sleep_Improvement'
        else:
            return 'Active_Movement'
    
    # Priority 8: Obesity management
    if disease == 'Obesity' or bmi >= 30:
        return 'Active_Movement'
    
    # Priority 9: Prediabetes / early intervention
    if disease in ['Prediabetes', 'Hyperlipidemia']:
        return 'Preventive_Care'
    
    # Default: Healthy individuals
    if age < 50 and bmi < 25 and metabolic_risk < 25:
        return 'Wellness_Focus'
    else:
        return 'Maintenance'


# ============================================================
# SYNTHETIC DATA GENERATION (SAME AS BEFORE)
# ============================================================

def generate_synthetic_records(n_samples):
    """Generate synthetic data with strong correlations"""
    
    print(f"\n📊 Generating {n_samples:,} synthetic records with evidence-based labels...")
    
    records = []
    
    # Disease profiles (SAME AS YOUR ORIGINAL)
    disease_profiles = {
        'Healthy': {
            'weight': 0.20, 'age': (18, 50, 35, 8), 'bmi': (18.5, 24.9, 22, 1.8),
            'bp': (90, 119, 105, 8), 'glucose': (70, 99, 85, 8), 
            'cholesterol': (120, 199, 165, 18), 'sleep': (7, 9, 7.5, 0.5),
            'stress': [0.70, 0.25, 0.05], 'activity': [0.05, 0.20, 0.45, 0.30],
            'smoking': [0.85, 0.12, 0.03], 'alcohol': (1.5, 1.2)
        },
        'Diabetes_Type2': {
            'weight': 0.12, 'age': (45, 80, 62, 10), 'bmi': (30, 45, 35, 4),
            'bp': (130, 180, 148, 12), 'glucose': (150, 280, 195, 30),
            'cholesterol': (200, 320, 248, 25), 'sleep': (4, 7, 5.5, 0.9),
            'stress': [0.15, 0.40, 0.45], 'activity': [0.60, 0.30, 0.08, 0.02],
            'smoking': [0.35, 0.35, 0.30], 'alcohol': (4.2, 2.5)
        },
        'Heart_Disease': {
            'weight': 0.10, 'age': (50, 85, 66, 9), 'bmi': (26, 40, 31, 4),
            'bp': (145, 200, 168, 15), 'glucose': (100, 180, 122, 20),
            'cholesterol': (230, 350, 278, 30), 'sleep': (4, 7, 5.9, 0.9),
            'stress': [0.10, 0.35, 0.55], 'activity': [0.65, 0.25, 0.08, 0.02],
            'smoking': [0.25, 0.30, 0.45], 'alcohol': (6.5, 3.2)
        },
        'Hypertension': {
            'weight': 0.15, 'age': (40, 75, 57, 10), 'bmi': (25, 38, 29, 4),
            'bp': (145, 190, 158, 12), 'glucose': (85, 140, 104, 15),
            'cholesterol': (180, 280, 218, 22), 'sleep': (5, 8, 6.6, 0.8),
            'stress': [0.20, 0.50, 0.30], 'activity': [0.45, 0.35, 0.15, 0.05],
            'smoking': [0.40, 0.35, 0.25], 'alcohol': (5.1, 2.8)
        },
        'Obesity': {
            'weight': 0.10, 'age': (25, 65, 44, 12), 'bmi': (32, 50, 37, 5),
            'bp': (110, 160, 128, 12), 'glucose': (90, 160, 113, 18),
            'cholesterol': (180, 300, 233, 26), 'sleep': (4, 7, 5.6, 1.0),
            'stress': [0.25, 0.45, 0.30], 'activity': [0.70, 0.25, 0.04, 0.01],
            'smoking': [0.50, 0.30, 0.20], 'alcohol': (4.0, 2.6)
        },
        'Prediabetes': {
            'weight': 0.10, 'age': (35, 70, 51, 10), 'bmi': (26, 35, 29, 3.5),
            'bp': (115, 145, 127, 10), 'glucose': (100, 125, 111, 7),
            'cholesterol': (190, 250, 213, 18), 'sleep': (5, 8, 6.6, 0.9),
            'stress': [0.30, 0.50, 0.20], 'activity': [0.40, 0.35, 0.20, 0.05],
            'smoking': [0.55, 0.25, 0.20], 'alcohol': (3.2, 2.1)
        },
        'Hyperlipidemia': {
            'weight': 0.08, 'age': (35, 75, 55, 12), 'bmi': (23, 35, 27, 4),
            'bp': (105, 150, 123, 12), 'glucose': (80, 130, 98, 12),
            'cholesterol': (250, 350, 288, 28), 'sleep': (5, 8, 6.9, 0.8),
            'stress': [0.35, 0.45, 0.20], 'activity': [0.35, 0.40, 0.20, 0.05],
            'smoking': [0.50, 0.30, 0.20], 'alcohol': (4.3, 2.7)
        },
        'MetabolicSyndrome': {
            'weight': 0.06, 'age': (40, 75, 59, 11), 'bmi': (32, 45, 36, 4.5),
            'bp': (140, 180, 153, 12), 'glucose': (115, 180, 138, 20),
            'cholesterol': (220, 320, 258, 26), 'sleep': (4, 7, 5.4, 0.9),
            'stress': [0.12, 0.38, 0.50], 'activity': [0.60, 0.30, 0.08, 0.02],
            'smoking': [0.35, 0.35, 0.30], 'alcohol': (5.4, 3.1)
        },
        'Anxiety': {
            'weight': 0.05, 'age': (20, 60, 37, 11), 'bmi': (18, 32, 23, 4),
            'bp': (100, 140, 116, 10), 'glucose': (75, 110, 88, 10),
            'cholesterol': (150, 230, 183, 20), 'sleep': (3, 6, 4.5, 1.0),
            'stress': [0.02, 0.18, 0.80], 'activity': [0.30, 0.40, 0.25, 0.05],
            'smoking': [0.50, 0.25, 0.25], 'alcohol': (3.5, 2.5)
        },
        'SleepDisorder': {
            'weight': 0.03, 'age': (25, 65, 45, 13), 'bmi': (22, 38, 29, 5),
            'bp': (105, 150, 125, 12), 'glucose': (80, 130, 100, 14),
            'cholesterol': (160, 250, 200, 22), 'sleep': (3, 5.5, 4.2, 0.8),
            'stress': [0.15, 0.40, 0.45], 'activity': [0.50, 0.35, 0.12, 0.03],
            'smoking': [0.50, 0.30, 0.20], 'alcohol': (4.1, 2.5)
        },
        'NutritionalDeficiency': {
            'weight': 0.01, 'age': (18, 70, 42, 15), 'bmi': (15, 22, 19, 2),
            'bp': (90, 120, 105, 8), 'glucose': (70, 95, 82, 8),
            'cholesterol': (120, 180, 150, 15), 'sleep': (5, 8, 6.5, 0.9),
            'stress': [0.40, 0.40, 0.20], 'activity': [0.30, 0.40, 0.25, 0.05],
            'smoking': [0.60, 0.25, 0.15], 'alcohol': (2.0, 1.5)
        }
    }
    
    diseases = list(disease_profiles.keys())
    weights = [disease_profiles[d]['weight'] for d in diseases]
    
    # Generate records
    for i in range(n_samples):
        if i % 50000 == 0 and i > 0:
            print(f"    Generated {i:,} / {n_samples:,} records...")
        
        disease = np.random.choice(diseases, p=weights)
        profile = disease_profiles[disease]
        
        # Generate features (SAME AS BEFORE)
        age = int(np.clip(np.random.normal(profile['age'][2], profile['age'][3]), 
                          profile['age'][0], profile['age'][1]))
        sex = np.random.choice(['Male', 'Female'])
        bmi = np.clip(np.random.normal(profile['bmi'][2], profile['bmi'][3]),
                      profile['bmi'][0], profile['bmi'][1])
        bp_systolic = int(np.clip(np.random.normal(profile['bp'][2], profile['bp'][3]),
                                   profile['bp'][0], profile['bp'][1]))
        glucose = np.clip(np.random.normal(profile['glucose'][2], profile['glucose'][3]),
                          profile['glucose'][0], profile['glucose'][1])
        cholesterol = np.clip(np.random.normal(profile['cholesterol'][2], profile['cholesterol'][3]),
                              profile['cholesterol'][0], profile['cholesterol'][1])
        sleep_hours = np.clip(np.random.normal(profile['sleep'][2], profile['sleep'][3]),
                              profile['sleep'][0], profile['sleep'][1])
        
        stress_level = np.random.choice(['Low', 'Medium', 'High'], p=profile['stress'])
        activity_level = np.random.choice(['Sedentary', 'Light', 'Moderate', 'Active'], 
                                         p=profile['activity'])
        diet_type = np.random.choice(['Western', 'Mediterranean', 'Asian', 'Vegetarian'],
                                     p=[0.4, 0.25, 0.2, 0.15])
        smoking = np.random.choice(['Never', 'Former', 'Current'], p=profile['smoking'])
        alcohol_units = np.clip(np.random.normal(profile['alcohol'][0], profile['alcohol'][1]), 0, 30)
        
        # NEW: Evidence-based label assignment
        diet_plan = assign_diet_plan_evidence_based(
            age, sex, bmi, glucose, cholesterol, bp_systolic, disease,
            activity_level, smoking, stress_level, sleep_hours, alcohol_units
        )
        
        lifestyle_plan = assign_lifestyle_plan_evidence_based(
            age, sex, bmi, bp_systolic, glucose, disease,
            activity_level, smoking, stress_level, sleep_hours, 
            alcohol_units, cholesterol
        )
        
        records.append({
            'age': age, 'sex': sex, 'bmi': round(bmi, 1),
            'bp_systolic': bp_systolic, 'glucose': round(glucose, 1),
            'cholesterol': round(cholesterol, 1), 'sleep_hours': round(sleep_hours, 1),
            'stress_level': stress_level, 'activity_level': activity_level,
            'diet_type': diet_type, 'smoking': smoking,
            'alcohol_units': round(alcohol_units, 1),
            'disease': disease, 'diet_plan': diet_plan, 'lifestyle_plan': lifestyle_plan
        })
    
    print(f"    Generated {n_samples:,} / {n_samples:,} records... ✅ COMPLETE")
    return pd.DataFrame(records)


# ============================================================
# MAIN EXECUTION (SAME AS BEFORE)
# ============================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("🥼 REVISED HEALTHCARE DATASET GENERATION")
    print("Evidence-Based Diet & Lifestyle Label Assignment")
    print("="*70)
    
    os.makedirs('data/raw', exist_ok=True)
    
    # Generate 500K records with NEW label assignment
    df_final = generate_synthetic_records(500000)
    
    # Shuffle
    df_final = df_final.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Save
    output_path = 'data/raw/healthcare_data.csv'
    df_final.to_csv(output_path, index=False)
    
    # Print summary
    print("\n" + "="*70)
    print("✅ DATASET GENERATION COMPLETE")
    print("="*70)
    print(f"📁 Saved to: {output_path}")
    print(f"📊 Total rows: {len(df_final):,}")
    
    print("\n📈 Disease distribution:")
    disease_counts = df_final['disease'].value_counts()
    for disease, count in disease_counts.items():
        pct = (count / len(df_final)) * 100
        print(f"  {disease:25s}: {count:6,} ({pct:5.2f}%)")
    
    print("\n📈 Diet plan distribution:")
    diet_counts = df_final['diet_plan'].value_counts()
    for diet, count in diet_counts.items():
        pct = (count / len(df_final)) * 100
        print(f"  {diet:25s}: {count:6,} ({pct:5.2f}%)")
    
    print("\n📈 Lifestyle plan distribution:")
    lifestyle_counts = df_final['lifestyle_plan'].value_counts()
    for lifestyle, count in lifestyle_counts.items():
        pct = (count / len(df_final)) * 100
        print(f"  {lifestyle:25s}: {count:6,} ({pct:5.2f}%)")
    
    print("\n✅ Dataset ready for training!")
    print("\n▶️ Next step: python scripts/2_preprocess_data.py")