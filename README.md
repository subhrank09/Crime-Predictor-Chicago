🕵️ Chicago Crime Predictor: Spatial-Temporal Forecasting
📌 Project Overview
This project leverages Machine Learning to predict the category of crime likely to occur in Chicago based on location and time. By integrating historical data from Kaggle with live updates from the Chicago Data Portal (SODA API), the system provides a robust analytical tool for understanding urban safety patterns.

Key Features
Hybrid Data Pipeline: Combines 20+ years of historical records with real-time API ingestion.

Feature Engineering: Extracts temporal insights (rush hour, night-time, seasonality) and spatial bins.

Advanced Modeling: Implements a Random Forest Classifier to handle non-linear urban crime trends.

Interactive EDA: Visualizes crime "hotspots" and hourly trends using Seaborn and Folium.

📂 Project Structure
Plaintext

crime-predictor/
├── data/
│   ├── raw/                # Original API & Kaggle datasets
│   └── processed/          # Cleaned features ready for ML
├── src/
│   ├── data_ingestion.py   # Chicago SODA API implementation
│   ├── preprocessing.py    # Cleaning & Lat/Lon binning
│   ├── feature_engineering.py # Temporal feature extraction
│   └── train_model.py      # Random Forest training & evaluation
├── models/                 # Serialized .pkl model files
├── notebooks/              # Jupyter walkthroughs for EDA & Testing
└── requirements.txt        # Project dependencies
🚀 Getting Started
1. Prerequisites
Python 3.8+

A Chicago Data Portal App Token (Recommended for API ingestion).

2. Installation
Bash

git clone https://github.com/your-username/crime-predictor.git
cd crime-predictor
pip install -r requirements.txt
3. Usage
Ingest Data: Run python src/data_ingestion.py to fetch latest records.

Process: Run python src/preprocessing.py to clean raw files.

Train: Run python src/train_model.py to generate the .pkl model.

Visualize: Open notebooks/01_eda_and_visualization.ipynb in VS Code to see results.

📊 Results & Performance
Model: Random Forest Classifier

Accuracy: ~XX% (Replace with your actual score)

Top Features: Hour of day, Latitude/Longitude, and Location Description (e.g., Street vs. Apartment).

Note for Recruiters: This project demonstrates a full-stack Data Science lifecycle—from raw API handling and ETL pipelines to model deployment.

🛠 Tech Stack
Languages: Python

Libraries: Pandas, Scikit-Learn, Joblib, Sodapy, Folium

Tools: VS Code, Jupyter, Git
