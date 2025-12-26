import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

def train_crime_predictor():
    # 1. Load the processed data
    DATA_PATH = Path(__file__).parent.parent / "data" / "processed" / "chicago_cleaned.csv"
    df = pd.read_csv(DATA_PATH)

    # 2. Define Features (X) and Target (y)
    # We want to predict 'crime_cat'
    X = df.drop('crime_cat', axis=1)
    y = df['crime_cat']

    # 3. Train-Test Split (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. Initialize and Train Random Forest
    print("Training the Random Forest model (this may take a minute)...")
    # n_estimators=100 means 100 decision trees are voting
    model = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    # 5. Evaluate the Model
    y_pred = model.predict(X_test)
    print("\n--- Model Performance Report ---")
    print(f"Overall Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print("\nDetailed Metrics:")
    print(classification_report(y_test, y_pred))

    # 6. Save the trained model to the 'models/' folder
    MODEL_DIR = Path(__file__).parent.parent / "models"
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_DIR / "crime_model_v1.pkl")
    print(f"\nModel saved to {MODEL_DIR}/crime_model_v1.pkl")

if __name__ == "__main__":
    train_crime_predictor()