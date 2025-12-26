import pandas as pd
from pathlib import Path

def engineer_features():
    processed_path = Path(__file__).parent.parent / "data" / "processed" / "chicago_cleaned.csv"
    if not processed_path.exists():
        print("Cleaned data not found. Please run preprocessing.py first.")
        return

    df = pd.read_csv(processed_path)

    # 1. Temporal Features
    df['is_night'] = df['hour'].apply(lambda x: 1 if x >= 22 or x <= 4 else 0)
    df['is_rush_hour'] = df['hour'].apply(lambda x: 1 if (7 <= x <= 9) or (16 <= x <= 18) else 0)
    
    # 2. Assign Seasons
    def get_season(month):
        if month in [12, 1, 2]: return 'Winter'
        if month in [3, 4, 5]: return 'Spring'
        if month in [6, 7, 8]: return 'Summer'
        return 'Fall'
    
    df['season'] = df['month'].apply(get_season)
    # One-hot encode seasons
    df = pd.get_dummies(df, columns=['season'], prefix='s')

    # 3. Save as the final training set
    final_path = Path(__file__).parent.parent / "data" / "processed" / "chicago_final_features.csv"
    df.to_csv(final_path, index=False)
    print(f"Feature engineering complete! Final dataset saved to: {final_path}")

if __name__ == "__main__":
    engineer_features()