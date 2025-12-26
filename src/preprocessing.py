import pandas as pd
from pathlib import Path

def preprocess_chicago_data(input_path, output_path):
    # 1. Load the data
    print("Loading data...")
    df = pd.read_json(input_path)
    
    # 2. Drop rows with missing crucial info (Lat/Long/Date)
    df = df.dropna(subset=['latitude', 'longitude', 'date', 'primary_type'])
    
    # 3. Convert Date column to datetime objects
    df['date'] = pd.to_datetime(df['date'])
    
    # 4. Feature Engineering: Time
    print("Engineering time features...")
    df['hour'] = df['date'].dt.hour
    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    
    # 5. Feature Engineering: Spatial (Rounding for better grouping)
    # Rounding to 3 decimal places captures a block-level area (~110 meters)
    df['lat_bin'] = df['latitude'].astype(float).round(3)
    df['lon_bin'] = df['longitude'].astype(float).round(3)
    
    # 6. Simplify Crime Categories (Optional but recommended)
    # Many crimes are rare; grouping them improves model accuracy
    top_crimes = df['primary_type'].value_counts().nlargest(10).index
    df['crime_cat'] = df['primary_type'].apply(lambda x: x if x in top_crimes else 'OTHER')
    
    # 7. Select relevant features for the model
    features = ['hour', 'day_of_week', 'month', 'is_weekend', 'lat_bin', 'lon_bin', 'crime_cat']
    final_df = df[features]
    
    # 8. Save the cleaned data
    final_df.to_csv(output_path, index=False)
    print(f"Preprocessing complete! Saved to {output_path}")

if __name__ == "__main__":
    # Point to your global data folder
    RAW_DATA = Path(__file__).parent.parent / "data" / "raw" / "api_recent.json"
    PROCESSED_DATA = Path(__file__).parent.parent / "data" / "processed" / "chicago_cleaned.csv"
    
    # Ensure processed folder exists
    PROCESSED_DATA.parent.mkdir(parents=True, exist_ok=True)
    
    preprocess_chicago_data(RAW_DATA, PROCESSED_DATA)