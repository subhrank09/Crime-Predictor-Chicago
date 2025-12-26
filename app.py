import streamlit as st
import pandas as pd
import joblib
import os
import gdown
from pathlib import Path
import datetime

# --- CONFIGURATION ---
st.set_page_config(page_title="Chicago Crime Predictor", page_icon="🕵️", layout="wide")

# --- MODEL DOWNLOAD & LOAD ---
MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "crime_model_v1.pkl"
# Your specific Google Drive File ID
GOOGLE_DRIVE_ID = "12UiIKe90v5Le72hxhBN0DLKPoCN042No"

@st.cache_resource
def load_model():
    # 1. Ensure the models directory exists
    if not MODEL_DIR.exists():
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    # 2. Download from Google Drive if the file isn't there
    if not MODEL_PATH.exists():
        with st.spinner("Downloading trained model from Google Drive (approx. 190MB)... Please wait."):
            url = f'https://drive.google.com/uc?id={GOOGLE_DRIVE_ID}'
            gdown.download(url, str(MODEL_PATH), quiet=False)
    
    # 3. Load the model
    return joblib.load(MODEL_PATH)

# Initialize the model
try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# --- UI DESIGN ---
st.title("🕵️ Chicago Crime Prediction Portal")
st.markdown("""
This tool uses a **Random Forest Classifier** to predict the most likely category of crime 
based on temporal and spatial inputs. 
""")

st.sidebar.header("Input Parameters")

# --- USER INPUTS ---
with st.sidebar:
    input_date = st.date_input("Select Date", datetime.date.today())
    input_time = st.time_input("Select Time", datetime.time(12, 0))
    
    # Coordinates for Chicago (Approximate ranges)
    lat = st.number_input("Latitude", value=41.8781, format="%.4f")
    lon = st.number_input("Longitude", value=-87.6298, format="%.4f")

# --- FEATURE PROCESSING ---
hour = input_time.hour
day_of_week = input_date.weekday()
month = input_date.month
is_weekend = 1 if day_of_week >= 5 else 0
lat_bin = round(lat, 3)
lon_bin = round(lon, 3)

# Create a DataFrame for prediction
input_df = pd.DataFrame([[hour, day_of_week, month, is_weekend, lat_bin, lon_bin]], 
                        columns=['hour', 'day_of_week', 'month', 'is_weekend', 'lat_bin', 'lon_bin'])

# --- PREDICTION ---
if st.button("Predict Crime Category"):
    prediction = model.predict(input_df)[0]
    probabilities = model.predict_proba(input_df)
    
    # Display Result
    st.subheader(f"Predicted Category: :red[{prediction}]")
    
    # Visualizing Probability
    st.write("### Prediction Confidence")
    prob_df = pd.DataFrame(probabilities, columns=model.classes_).T
    prob_df.columns = ['Confidence']
    st.bar_chart(prob_df)

# --- MAP VISUALIZATION ---
st.write("### Selected Incident Location")
map_data = pd.DataFrame({'lat': [lat], 'lon': [lon]})
st.map(map_data)
