import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
import datetime

# --- CONFIGURATION ---
st.set_page_config(page_title="Chicago Crime Predictor", page_icon="🕵️", layout="wide")

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    model_path = Path("models/crime_model_v1.pkl")
    return joblib.load(model_path)

model = load_model()

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
# We need to transform user input into the same format used during training
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