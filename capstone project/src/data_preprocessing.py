import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder

GOOGLE_SHEET_CSV_URL = "https://docs.google.com/spreadsheets/d/1y_UlIjIqQvBfugews-9RstOPkZD4BVfIeOaX2aapKZc/export?format=csv"

def load_and_preprocess():
    # -----------------------------------------
    # Load LIVE data from Google Forms
    # -----------------------------------------
    df = pd.read_csv(GOOGLE_SHEET_CSV_URL)

    # Save snapshot (optional but professional)
    df.to_csv("../data/latest_form_data.csv", index=False)

    # -----------------------------------------
    # Ensure Age is numeric (VERY IMPORTANT)
    # -----------------------------------------
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce")

    # -----------------------------------------
    # Derive Mental Health Impact (Rule-Based)
    # -----------------------------------------
    def mental_health(row):
        if row["StressLevel"] == "High" and row["SleepHours"] < 6:
            return "Poor"
        elif row["StressLevel"] == "Medium":
            return "Neutral"
        else:
            return "Good"

    df["MentalHealthImpact"] = df.apply(mental_health, axis=1)

    # -----------------------------------------
    # Encode ONLY non-age categorical columns
    # -----------------------------------------
    le = LabelEncoder()

    categorical_cols = [
        "Gender",
        "Primary_AI_Use",
        "Reliance_On_AI_For_Learning",
        "StressLevel"  # required for ML
    ]

    for col in categorical_cols:
        df[col] = le.fit_transform(df[col])

    return df
