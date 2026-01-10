"""
train_comparative_models.py - Comprehensive ML Model Comparison Study
=====================================================================

This script implements a rigorous comparative analysis of six machine learning
algorithms for multi-output healthcare recommendation, suitable for IEEE publication.

Models Compared:
1. Logistic Regression (baseline linear model)
2. Support Vector Machine (SVM) with RBF kernel
3. XGBoost (gradient boosting)
4. LightGBM (efficient gradient boosting)
5. Neural Network (Multi-Layer Perceptron)
6. Random Forest (ensemble baseline)

Each model is trained for three prediction tasks:
- Disease Risk Classification (11 classes)
- Diet Plan Recommendation (8 classes)
- Lifestyle Intervention Suggestion (8 classes)

Performance metrics: Accuracy, Precision, Recall, F1-Score, Training Time
"""

import pandas as pd
import numpy as np
import time
import json
import pickle
import os
import warnings
from pathlib import Path

# Machine Learning Libraries
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, classification_report, confusion_matrix)

# Model Imports
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================

class Config:
    """Configuration parameters for the comparative study"""
    
    # Data paths
    DATA_PATH = 'data/processed/healthcare_cleaned.parquet'
    MODELS_DIR = 'models/comparative'
    RESULTS_FILE = 'models/comparative_results.json'
    
    # Train-test split
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    
    # Feature columns
    NUMERIC_FEATURES = ['age', 'bmi', 'bp_systolic', 'glucose', 'cholesterol', 
                       'sleep_hours', 'alcohol_units']
    CATEGORICAL_FEATURES = ['sex', 'stress_level', 'activity_level', 
                           'diet_type', 'smoking']
    
    # Target columns
    TARGETS = {
        'disease': 'disease',
        'diet': 'diet_plan',
        'lifestyle': 'lifestyle_plan'
    }
    
    # Cross-validation folds
    CV_FOLDS = 5

# ============================================================
# DATA LOADING AND PREPROCESSING
# ============================================================

class DataProcessor:
    """Handles data loading, cleaning, and preprocessing"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='mean')
        self.label_encoders = {}
        
    def load_and_prepare_data(self, data_path):
        """Load and preprocess the healthcare dataset"""
        
        print("="*70)
        print("DATA LOADING AND PREPROCESSING")
        print("="*70)
        
        # Load data
        print(f"\n[1/6] Loading data from {data_path}...")
        df = pd.read_parquet(data_path)
        print(f"✓ Loaded {len(df):,} records with {len(df.columns)} columns")
        
        # Select required columns
        print("\n[2/6] Selecting features and targets...")
        required_cols = (Config.NUMERIC_FEATURES + Config.CATEGORICAL_FEATURES + 
                        list(Config.TARGETS.values()))
        
        # Check for missing columns
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing columns in dataset: {missing_cols}")
        
        df = df[required_cols].copy()
        print(f"✓ Selected {len(required_cols)} columns")
        
        # Handle missing values in numeric features
        print("\n[3/6] Handling missing values...")
        null_counts = df[Config.NUMERIC_FEATURES].isnull().sum()
        if null_counts.sum() > 0:
            print(f"  Found {null_counts.sum()} missing values")
            df[Config.NUMERIC_FEATURES] = self.imputer.fit_transform(
                df[Config.NUMERIC_FEATURES]
            )
            print("  ✓ Imputed missing numeric values")
        else:
            print("  ✓ No missing values found")
        
        # Encode categorical features
        print("\n[4/6] Encoding categorical features...")
        for col in Config.CATEGORICAL_FEATURES:
            le = LabelEncoder()
            df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
            self.label_encoders[col] = le
        print(f"✓ Encoded {len(Config.CATEGORICAL_FEATURES)} categorical features")
        
        # Encode target variables
        print("\n[5/6] Encoding target variables...")
        for target_name, target_col in Config.TARGETS.items():
            le = LabelEncoder()
            df[f'{target_col}_encoded'] = le.fit_transform(df[target_col].astype(str))
            self.label_encoders[f'{target_name}_target'] = le
            print(f"  ✓ {target_name}: {len(le.classes_)} classes")
        
        # Prepare feature matrix
        print("\n[6/6] Preparing feature matrix...")
        feature_cols = (Config.NUMERIC_FEATURES + 
                       [f'{col}_encoded' for col in Config.CATEGORICAL_FEATURES])
        X = df[feature_cols].values
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        print(f"✓ Feature matrix shape: {X_scaled.shape}")
        
        # Prepare targets
        y_disease = df[f"{Config.TARGETS['disease']}_encoded"].values
        y_diet = df[f"{Config.TARGETS['diet']}_encoded"].values
        y_lifestyle = df[f"{Config.TARGETS['lifestyle']}_encoded"].values
        
        return X_scaled, y_disease, y_diet, y_lifestyle, df
    
    def save_preprocessors(self, output_dir):
        """Save preprocessing objects"""
        os.makedirs(output_dir, exist_ok=True)
        
        with open(f'{output_dir}/scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        with open(f'{output_dir}/imputer.pkl', 'wb') as f:
            pickle.dump(self.imputer, f)
        with open(f'{output_dir}/label_encoders.pkl', 'wb') as f:
            pickle.dump(self.label_encoders, f)
        
        print("\n✓ Preprocessors saved")

# ============================================================
# MODEL DEFINITIONS
# ============================================================

class ModelFactory:
    """Factory for creating and configuring ML models"""
    
    @staticmethod
    def get_models():
        """Return dictionary of all models to compare"""
        
        models = {
            'LogisticRegression': LogisticRegression(
                max_iter=1000,
                random_state=Config.RANDOM_STATE,
                n_jobs=-1,
                solver='saga',
                multi_class='multinomial'
            ),
            
            'SVM': SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                random_state=Config.RANDOM_STATE,
                probability=True,
                max_iter=1000
            ),
            
            'XGBoost': XGBClassifier(
                n_estimators=100,
                max_depth=10,
                learning_rate=0.1,
                random_state=Config.RANDOM_STATE,
                n_jobs=-1,
                eval_metric='mlogloss'
            ),
            
            'LightGBM': LGBMClassifier(
                n_estimators=100,
                max_depth=10,
                learning_rate=0.1,
                random_state=Config.RANDOM_STATE,
                n_jobs=-1,
                verbose=-1
            ),
            
            'NeuralNetwork': MLPClassifier(
                hidden_layer_sizes=(128, 64, 32),
                activation='relu',
                solver='adam',
                max_iter=500,
                random_state=Config.RANDOM_STATE,
                early_stopping=True,
                validation_fraction=0.1
            ),
            
            'RandomForest': RandomForestClassifier(
                n_estimators=120,
                max_depth=15,
                min_samples_split=5,
                random_state=Config.RANDOM_STATE,
                n_jobs=-1
            )
        }
        
        return models

# ============================================================
# MODEL TRAINER
# ============================================================

class ModelTrainer:
    """Handles model training and evaluation"""
    
    def __init__(self):
        self.results = {
            'disease': {},
            'diet': {},
            'lifestyle': {},
            'metadata': {
                'train_samples': 0,
                'test_samples': 0,
                'total_samples': 0,
                'features': len(Config.NUMERIC_FEATURES) + len(Config.CATEGORICAL_FEATURES),
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
        }
    
    def train_and_evaluate(self, model_name, model, X_train, X_test, y_train, y_test, target_name):
        """Train a model and compute all metrics"""
        
        print(f"\n  Training {model_name}...")
        
        # Training
        start_time = time.time()
        try:
            model.fit(X_train, y_train)
            training_time = time.time() - start_time
            
            # Predictions
            y_pred = model.predict(X_test)
            
            # Metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            
            # Cross-validation score
            cv_scores = cross_val_score(model, X_train, y_train, 
                                       cv=min(Config.CV_FOLDS, 3), 
                                       scoring='accuracy', n_jobs=-1)
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()
            
            metrics = {
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1),
                'cv_accuracy_mean': float(cv_mean),
                'cv_accuracy_std': float(cv_std),
                'training_time_seconds': float(training_time),
                'num_classes': int(len(np.unique(y_train)))
            }
            
            print(f"    ✓ Accuracy: {accuracy:.4f} | F1: {f1:.4f} | Time: {training_time:.2f}s")
            
            return model, metrics
            
        except Exception as e:
            print(f"    ✗ Error training {model_name}: {str(e)}")
            return None, None
    
    def train_all_models(self, X_train, X_test, y_dict_train, y_dict_test):
        """Train all models for all targets"""
        
        print("\n" + "="*70)
        print("MODEL TRAINING - COMPARATIVE STUDY")
        print("="*70)
        
        models_dict = ModelFactory.get_models()
        
        # Store sample counts
        self.results['metadata']['train_samples'] = len(X_train)
        self.results['metadata']['test_samples'] = len(X_test)
        self.results['metadata']['total_samples'] = len(X_train) + len(X_test)
        
        # Train for each target
        for target_name in ['disease', 'diet', 'lifestyle']:
            print(f"\n{'='*70}")
            print(f"TARGET: {target_name.upper()}")
            print(f"{'='*70}")
            
            y_train = y_dict_train[target_name]
            y_test = y_dict_test[target_name]
            
            self.results[target_name]['num_classes'] = int(len(np.unique(y_train)))
            self.results[target_name]['models'] = {}
            
            # Train each model
            for model_name, model in models_dict.items():
                trained_model, metrics = self.train_and_evaluate(
                    model_name, model, X_train, X_test, y_train, y_test, target_name
                )
                
                if metrics:
                    self.results[target_name]['models'][model_name] = metrics
                    
                    # Save trained model
                    model_path = f"{Config.MODELS_DIR}/{target_name}_{model_name.lower()}.pkl"
                    os.makedirs(Config.MODELS_DIR, exist_ok=True)
                    with open(model_path, 'wb') as f:
                        pickle.dump(trained_model, f)
        
        return self.results
    
    def save_results(self, filepath):
        """Save results to JSON file"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=4)
        print(f"\n✓ Results saved to {filepath}")
    
    def print_summary(self):
        """Print comprehensive results summary"""
        
        print("\n" + "="*70)
        print("COMPARATIVE STUDY RESULTS SUMMARY")
        print("="*70)
        
        print(f"\nDataset Information:")
        print(f"  Training samples: {self.results['metadata']['train_samples']:,}")
        print(f"  Test samples: {self.results['metadata']['test_samples']:,}")
        print(f"  Total samples: {self.results['metadata']['total_samples']:,}")
        print(f"  Number of features: {self.results['metadata']['features']}")
        
        for target_name in ['disease', 'diet', 'lifestyle']:
            print(f"\n{'='*70}")
            print(f"{target_name.upper()} PREDICTION")
            print(f"{'='*70}")
            print(f"Number of classes: {self.results[target_name]['num_classes']}")
            
            # Create comparison table
            models_data = self.results[target_name]['models']
            
            print(f"\n{'Model':<20} {'Accuracy':<12} {'F1-Score':<12} {'Precision':<12} {'Recall':<12} {'Time(s)':<10}")
            print("-"*88)
            
            # Sort by accuracy
            sorted_models = sorted(models_data.items(), 
                                  key=lambda x: x[1]['accuracy'], 
                                  reverse=True)
            
            for model_name, metrics in sorted_models:
                print(f"{model_name:<20} "
                      f"{metrics['accuracy']:<12.4f} "
                      f"{metrics['f1_score']:<12.4f} "
                      f"{metrics['precision']:<12.4f} "
                      f"{metrics['recall']:<12.4f} "
                      f"{metrics['training_time_seconds']:<10.2f}")
            
            # Best model
            best_model = sorted_models[0]
            print(f"\n🏆 Best Model: {best_model[0]} (Accuracy: {best_model[1]['accuracy']:.4f})")
        
        # Overall best models
        print(f"\n{'='*70}")
        print("OVERALL BEST MODELS BY TARGET")
        print(f"{'='*70}")
        
        for target_name in ['disease', 'diet', 'lifestyle']:
            models_data = self.results[target_name]['models']
            best = max(models_data.items(), key=lambda x: x[1]['accuracy'])
            print(f"{target_name.capitalize():<15}: {best[0]:<20} (Acc: {best[1]['accuracy']:.4f})")

# ============================================================
# MAIN EXECUTION
# ============================================================

def main():
    """Main execution pipeline"""
    
    print("="*70)
    print("COMPREHENSIVE MACHINE LEARNING COMPARATIVE STUDY")
    print("Healthcare Recommendation System - Multi-Output Classification")
    print("="*70)
    print("\nThis study compares 6 ML algorithms across 3 prediction tasks:")
    print("  • Disease Risk Classification")
    print("  • Diet Plan Recommendation")
    print("  • Lifestyle Intervention Suggestion")
    print("\nModels: LogisticRegression, SVM, XGBoost, LightGBM, NeuralNetwork, RandomForest")
    print("="*70)
    
    try:
        # 1. Load and preprocess data
        processor = DataProcessor()
        X, y_disease, y_diet, y_lifestyle, df = processor.load_and_prepare_data(
            Config.DATA_PATH
        )
        
        # 2. Train-test split
        print("\n" + "="*70)
        print("TRAIN-TEST SPLIT")
        print("="*70)
        print(f"\nSplitting data: {int((1-Config.TEST_SIZE)*100)}% train, {int(Config.TEST_SIZE*100)}% test")
        
        # Split for disease
        X_train, X_test, y_disease_train, y_disease_test = train_test_split(
            X, y_disease, test_size=Config.TEST_SIZE, 
            random_state=Config.RANDOM_STATE, stratify=y_disease
        )
        
        # Get corresponding indices for other targets
        train_indices = np.arange(len(X))
        test_indices = train_indices[-len(y_disease_test):]
        train_indices = train_indices[:-len(y_disease_test)]
        
        y_dict_train = {
            'disease': y_disease_train,
            'diet': y_diet[train_indices],
            'lifestyle': y_lifestyle[train_indices]
        }
        
        y_dict_test = {
            'disease': y_disease_test,
            'diet': y_diet[test_indices],
            'lifestyle': y_lifestyle[test_indices]
        }
        
        print(f"✓ Training set: {len(X_train):,} samples")
        print(f"✓ Test set: {len(X_test):,} samples")
        
        # 3. Train all models
        trainer = ModelTrainer()
        results = trainer.train_all_models(X_train, X_test, y_dict_train, y_dict_test)
        
        # 4. Save results
        trainer.save_results(Config.RESULTS_FILE)
        
        # 5. Save preprocessors
        processor.save_preprocessors(Config.MODELS_DIR)
        
        # 6. Print summary
        trainer.print_summary()
        
        print("\n" + "="*70)
        print("✓ COMPARATIVE STUDY COMPLETED SUCCESSFULLY")
        print("="*70)
        print(f"\n📁 Models saved in: {Config.MODELS_DIR}/")
        print(f"📊 Results saved in: {Config.RESULTS_FILE}")
        print("\nNext steps:")
        print("  1. Review comparative_results.json for detailed metrics")
        print("  2. Use best-performing models for deployment")
        print("  3. Incorporate findings in your IEEE paper")
        
    except FileNotFoundError as e:
        print(f"\n❌ ERROR: Data file not found")
        print(f"   {str(e)}")
        print("\n   Please ensure the preprocessed data exists:")
        print(f"   {Config.DATA_PATH}")
        print("\n   Run preprocessing scripts first:")
        print("   1. python scripts/1_generate_data_with_kaggle.py")
        print("   2. python scripts/2_preprocess_data.py")
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        print("\n   Troubleshooting:")
        print("   1. Check if all required libraries are installed")
        print("   2. Verify data file exists and is readable")
        print("   3. Ensure sufficient memory is available")
        print("   4. Check Python version (3.8+ recommended)")

if __name__ == "__main__":
    main()