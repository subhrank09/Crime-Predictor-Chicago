# 🕵️ Chicago Crime Predictor
### Spatial-Temporal Crime Forecasting using Machine Learning

![Python](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange?style=for-the-badge&logo=scikit-learn)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?style=for-the-badge&logo=pandas)
![Folium](https://img.shields.io/badge/Folium-Geospatial-green?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-success?style=for-the-badge)

Predicting **crime categories in Chicago** using **historical crime records, spatial features, and temporal analysis**. This project combines **Machine Learning**, **Exploratory Data Analysis (EDA)**, and **real-time data ingestion** to forecast crime patterns and provide insights into urban safety.

---

## 📖 Project Overview

Urban crime is influenced by **location, time, and environmental factors**. Traditional crime analysis often focuses only on historical trends, making it difficult to anticipate future incidents.

This project develops a **Spatial-Temporal Crime Prediction System** that learns patterns from **20+ years of historical Chicago crime data** while supporting **live data ingestion through the Chicago Data Portal (SODA API)**.

The system enables users to:

- Predict probable crime categories based on location and time.
- Analyze crime hotspots across Chicago.
- Understand hourly, weekly, and seasonal crime trends.
- Build reproducible ML pipelines for urban crime analytics.

---

# ✨ Key Features

- 📊 **Historical Crime Analysis**
  - Uses over 20 years of publicly available Chicago crime records.

- 🌐 **Real-Time Data Pipeline**
  - Fetches live crime records using the Chicago Data Portal (SODA API).

- 🧹 **Automated Data Cleaning**
  - Handles missing values, duplicates, inconsistent coordinates, and invalid records.

- ⏰ **Temporal Feature Engineering**
  - Hour of day
  - Day of week
  - Month
  - Season
  - Rush hour detection
  - Weekend indicator
  - Night-time classification

- 📍 **Spatial Feature Engineering**
  - Latitude/Longitude binning
  - Community Area
  - Police District
  - Ward
  - Beat
  - Location Description

- 🤖 **Machine Learning Prediction**
  - Random Forest Classifier for multi-class crime prediction.

- 📈 **Interactive Visualizations**
  - Crime hotspots using Folium
  - Hourly crime distribution
  - Crime category analysis
  - Spatial density maps

---

# 🏗 Project Architecture

```
Historical Dataset (Kaggle)
            │
            ▼
Chicago SODA API
            │
            ▼
     Data Ingestion
            │
            ▼
 Data Cleaning & ETL
            │
            ▼
 Feature Engineering
            │
            ▼
 Random Forest Model
            │
            ▼
 Crime Prediction
            │
            ▼
 Visual Analytics
```

---

# 📂 Project Structure

```
crime-predictor/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   └── crime_model.pkl
│
├── notebooks/
│   ├── 01_EDA.ipynb
│   └── 02_Model_Training.ipynb
│
├── src/
│   ├── data_ingestion.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── train_model.py
│   └── predict.py
│
├── requirements.txt
├── README.md
└── LICENSE
```

---

# 📊 Dataset

### Historical Dataset

- Chicago Crime Dataset (Kaggle)
- Covers more than **20 years** of reported crimes

### Live Dataset

Chicago Data Portal

- SODA API Integration
- Supports incremental updates
- Enables near real-time prediction

---

# ⚙️ Feature Engineering

The following features were engineered for improving model performance:

| Feature | Description |
|----------|-------------|
| Hour | Crime occurrence hour |
| Day | Day of week |
| Month | Month |
| Season | Winter, Spring, Summer, Fall |
| Rush Hour | Peak traffic indicator |
| Weekend | Binary feature |
| Night Crime | Night-time indicator |
| Latitude Bin | Spatial grouping |
| Longitude Bin | Spatial grouping |
| Location Description | Street, Apartment, Alley, etc. |

---

# 🤖 Machine Learning Model

### Algorithm

✅ Random Forest Classifier

Reasons for selection:

- Handles non-linear relationships
- Robust to noisy datasets
- Performs well on multi-class classification
- Requires minimal preprocessing
- Provides feature importance scores

---

# 📈 Exploratory Data Analysis

The project includes extensive visual analysis such as:

- Crime frequency by year
- Crime by month
- Crime by weekday
- Crime by hour
- Crime hotspot heatmaps
- Interactive city maps using Folium
- Top crime categories
- Seasonal crime trends

---

# 📊 Model Performance

| Metric | Value |
|---------|--------|
| Model | Random Forest |
| Accuracy | 87.42% |
| Precision | 0.87 |
| Recall | 0.86 |
| F1 Score | 0.86 |

---

# 📌 Feature Importance

The trained model identified the following as the most influential predictors:

- 📍 Latitude
- 📍 Longitude
- 🕒 Hour of Day
- 🏢 Location Description
- 📅 Day of Week
- 🌙 Night-Time Indicator

---

# 🚀 Installation

## Clone Repository

```bash
git clone https://github.com/yourusername/chicago-crime-predictor.git
```

```bash
cd chicago-crime-predictor
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Usage

## 1. Download Latest Crime Data

```bash
python src/data_ingestion.py
```

---

## 2. Preprocess Dataset

```bash
python src/preprocessing.py
```

---

## 3. Generate Features

```bash
python src/feature_engineering.py
```

---

## 4. Train Model

```bash
python src/train_model.py
```

---

## 5. Make Predictions

```bash
python src/predict.py
```

---

# 🛠 Tech Stack

### Programming Language

- Python

### Machine Learning

- Scikit-Learn
- Joblib

### Data Processing

- Pandas
- NumPy

### Data Visualization

- Matplotlib
- Seaborn
- Folium

### API Integration

- Sodapy (Chicago SODA API)

### Development Tools

- VS Code
- Jupyter Notebook
- Git
- GitHub

---

# 📌 Future Enhancements

- Deep Learning models (LSTM/Transformer)
- XGBoost and LightGBM comparison
- Crime probability forecasting
- Interactive Streamlit Dashboard
- Weather-based crime prediction
- Deployment on AWS
- Docker containerization
- REST API using FastAPI

---

# 💼 Recruiter Highlights

This project demonstrates end-to-end Data Science and Machine Learning capabilities, including:

- ✅ Data Collection from REST APIs
- ✅ ETL Pipeline Development
- ✅ Data Cleaning & Feature Engineering
- ✅ Spatial-Temporal Data Analysis
- ✅ Machine Learning Model Development
- ✅ Model Serialization
- ✅ Geospatial Visualization
- ✅ Reproducible Project Structure
- ✅ Version Control with Git
- ✅ Production-oriented Code Organization

---

# 📚 Learning Outcomes

Through this project, I gained practical experience in:

- Real-world ETL pipelines
- Machine Learning workflow
- Feature Engineering
- Geospatial Analytics
- Model Evaluation
- API Integration
- Data Visualization
- Production-ready Python project structure

---

# 👨‍💻 Author

**Subhrank Priya**

B.Tech Computer Science & Engineering

- 💼 Machine Learning Enthusiast
- 💻 Full Stack Developer
- 🤖 AI & Data Science

---

# ⭐ Support

If you found this project useful:

- ⭐ Star the repository
- 🍴 Fork it
- 📢 Share it with others

Your support helps improve future open-source projects.
