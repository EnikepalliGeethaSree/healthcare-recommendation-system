# Personalized Healthcare Recommendation System

### Disease Prediction | Diet Recommendation | Lifestyle Advice

An end-to-end AI-powered healthcare system that predicts diseases, recommends personalized diet plans, and suggests lifestyle interventions using machine learning, big data processing, and explainable AI.

---

## Overview

This project presents a scalable healthcare recommendation system that integrates:

* Disease Risk Prediction
* Diet Plan Recommendation
* Lifestyle Intervention Suggestions

All from a single patient input profile.

The system leverages PySpark for big data processing, advanced ML models, and SHAP explainability to provide transparent and interpretable predictions.

---

## Key Features

* Multi-output prediction (Disease, Diet, Lifestyle)
* Big Data Processing using PySpark
* Real-time data streaming simulation
* Explainable AI using SHAP
* Interactive Streamlit Dashboard
* Comparative study of multiple ML models
* Health Risk Index (HRI) scoring system

---

## Tech Stack

| Category         | Technologies                                |
| ---------------- | ------------------------------------------- |
| Big Data         | PySpark (Spark MLlib, Structured Streaming) |
| Machine Learning | Scikit-learn, XGBoost, LightGBM             |
| Explainability   | SHAP                                        |
| Frontend         | Streamlit + Plotly                          |
| Data Processing  | Pandas, NumPy                               |
| Storage          | Parquet (Snappy Compression)                |
| Language         | Python                                      |

---

## System Architecture

### Architecture Diagram

```
                ┌────────────────────────────┐
                │   Synthetic Data Generator │
                │   (Pandas, NumPy)          │
                └────────────┬───────────────┘
                             │
                             ▼
                ┌────────────────────────────┐
                │   PySpark Preprocessing    │
                │ (Imputation, Encoding,     │
                │  Scaling, Feature Vector)  │
                └────────────┬───────────────┘
                             │
                             ▼
                ┌────────────────────────────┐
                │   Patient Clustering       │
                │      (KMeans)              │
                └────────────┬───────────────┘
                             │
                             ▼
                ┌────────────────────────────┐
                │   Model Training           │
                │ (Random Forest - Spark)    │
                └────────────┬───────────────┘
                             │
        ┌────────────────────┼────────────────────┐
        ▼                    ▼                    ▼
┌───────────────┐   ┌────────────────┐   ┌──────────────────┐
│ Disease Model │   │ Diet Model     │   │ Lifestyle Model  │
└──────┬────────┘   └──────┬─────────┘   └──────┬───────────┘
       │                   │                    │
       └──────────────┬────┴────┬───────────────┘
                      ▼         ▼
                ┌────────────────────────────┐
                │   SHAP Explainability      │
                └────────────┬───────────────┘
                             │
                             ▼
                ┌────────────────────────────┐
                │  Streamlit Dashboard       │
                │ (User Input + Prediction)  │
                └────────────┬───────────────┘
                             │
                             ▼
                ┌────────────────────────────┐
                │ Real-Time Streaming        │
                │ (Spark Structured Stream)  │
                └────────────────────────────┘
```

---

## Machine Learning Models

### Models Used:

* Logistic Regression
* Support Vector Machine (SVM)
* Random Forest
* XGBoost
* LightGBM
* Neural Network (MLP)

### Best Performers:

* Disease Prediction: XGBoost
* Diet Recommendation: LightGBM
* Lifestyle Recommendation: XGBoost

---

## Performance Highlights

* Disease Prediction Accuracy: 95.67%
* Diet Recommendation Accuracy: 97.90%
* Lifestyle Recommendation Accuracy: 98.60%

Note: High accuracy is due to rule-based synthetic data generation. Real-world validation is recommended.

---

## Health Risk Index (HRI)

A composite score calculated as:

HRI = 0.5 × Disease Risk + 0.3 × Lifestyle Risk + 0.2 × Diet Risk

Used to classify patients into:

* Low Risk
* Medium Risk
* High Risk

---

## Streamlit Dashboard

Features include:

* Data Exploration (EDA)
* Patient Input Panel
* Model Performance Visualization
* SHAP Explainability Viewer

---

## Real-Time Streaming

* Simulates live patient data ingestion
* Uses Spark Structured Streaming
* Processes incoming data at regular intervals

---

## Project Structure

```
healthcare-recommendation-system/

├── data/
├── scripts/
│   ├── data_generation.py
│   ├── preprocessing_pyspark.py
│   ├── clustering.py
│   ├── model_training.py
│   ├── streaming.py
│
├── models/
│   ├── train_comparative_models.py
│   ├── train_diet_models.py
│   ├── train_lifestyle_models.py
│
├── app.py
├── requirements.txt
└── README.md
```

---

## How to Run

### Clone Repository

```
git clone https://github.com/EnikepalliGeethaSree/healthcare-recommendation-system.git
cd healthcare-recommendation-system
```

### Install Dependencies

```
pip install -r requirements.txt
```

### Run Application

```
streamlit run app.py
```

---

## Future Improvements

* Integration with real EHR datasets (FHIR APIs)
* Cloud deployment (AWS / Azure)
* Federated learning for privacy
* Continuous model retraining
* NLP-based symptom input

---

## Limitations

* Dataset is synthetically generated
* Models learn rule-based patterns
* Requires validation on real-world clinical data

---

## Author

Geetha Enikepalli
B.Tech – Artificial Intelligence and Data Science
Amrita Vishwa Vidyapeetham, Bengaluru

---

## Conclusion

This project demonstrates a scalable, interpretable, and production-ready healthcare AI system, showcasing strong skills in:

* Machine Learning
* Big Data Engineering
* Explainable AI
* Full-stack Python Development
