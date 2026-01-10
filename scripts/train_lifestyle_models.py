# train_lifestyle_models.py

import pandas as pd
import numpy as np
import json
import os
import pickle
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# MODELS
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# ===============================
# CONFIG
# ===============================
DATA_PATH = "data/processed/healthcare_cleaned.parquet"
NUMERIC = ['age', 'bmi', 'bp_systolic', 'glucose', 'cholesterol', 'sleep_hours', 'alcohol_units']
CATEGORICAL = ['sex', 'stress_level', 'activity_level', 'diet_type', 'smoking']

TARGET = "lifestyle_plan"
SAVE_DIR = "models/lifestyle_models"
RESULTS_FILE = "models/lifestyle_results.json"

os.makedirs(SAVE_DIR, exist_ok=True)


# ===============================
# LOAD DATA
# ===============================
df = pd.read_parquet(DATA_PATH)

label_encoder = LabelEncoder()
df['y'] = label_encoder.fit_transform(df[TARGET])
pickle.dump(label_encoder, open(f"{SAVE_DIR}/lifestyle_label_encoder.pkl", "wb"))

X = df[NUMERIC + CATEGORICAL]
y = df['y']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# ===============================
# PREPROCESSOR
# ===============================
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), NUMERIC),
    ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL)
])


# ===============================
# MODELS
# ===============================
models = {
    "LogisticRegression": LogisticRegression(max_iter=1000, n_jobs=-1),
    "SVM": SVC(kernel="rbf", C=1.0, gamma="scale"),
    "XGBoost": XGBClassifier(
        n_estimators=150, max_depth=10, learning_rate=0.1, eval_metric="mlogloss"
    ),
    "LightGBM": LGBMClassifier(
        n_estimators=150, max_depth=12, learning_rate=0.1
    ),
    "NeuralNetwork": MLPClassifier(
        hidden_layer_sizes=(256,128,64), max_iter=300, early_stopping=True
    ),
    "RandomForest": RandomForestClassifier(
        n_estimators=200, max_depth=18, n_jobs=-1
    )
}


# ===============================
# TRAIN & EVALUATE
# ===============================
results = {}

for name, model in models.items():
    print(f"\nTraining {name}...")

    pipe = Pipeline([
        ("preprocess", preprocessor),
        ("model", model)
    ])

    start = time.time()
    pipe.fit(X_train, y_train)
    t = round(time.time() - start, 2)

    y_pred = pipe.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")

    results[name] = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "training_time": t
    }

    pickle.dump(pipe, open(f"{SAVE_DIR}/{name}.pkl", "wb"))

    print(f"➡ {name} | ACC={accuracy:.4f} | F1={f1:.4f} | Time={t}s")


json.dump(results, open(RESULTS_FILE, "w"), indent=4)
print(f"\nResults saved to: {RESULTS_FILE}")
