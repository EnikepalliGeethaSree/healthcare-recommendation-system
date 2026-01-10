# Healthcare Recommendation System

## Abstract
This project presents a comprehensive healthcare recommendation system
that integrates disease prediction, lifestyle guidance, and diet
recommendations using machine learning models. The system also includes
model explainability using SHAP and comparative model analysis.

## Key Features
- Disease prediction using ML classifiers
- Lifestyle and diet recommendation models
- Data preprocessing and clustering
- Model explainability using SHAP
- Comparative analysis of multiple models
- Streaming simulation for real-time inference
- IEEE paper-ready visualizations

## Dataset
Data is collected from Kaggle and processed into structured formats for
model training and evaluation.

## Project Structure
- data/ : Raw, processed, and Kaggle datasets
- models/ : Trained models, scalers, SHAP explainers, and results
- scripts/ : Modular pipeline scripts for data generation, training, and analysis
- figures_ieee_paper/ : Figures used for academic publication
- app.py : Application entry point

## How to Run
```bash
pip install -r requirements.txt
python app.py
