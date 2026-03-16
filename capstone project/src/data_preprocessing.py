import pandas as pd
from sklearn.preprocessing import LabelEncoder

GOOGLE_SHEET_CSV_URL = "https://docs.google.com/spreadsheets/d/1y_UlIjIqQvBfugews-9RstOPkZD4BVfIeOaX2aapKZc/export?format=csv"

def load_and_preprocess():

    # -----------------------------------------
    # Load LIVE data from Google Forms
    # -----------------------------------------
    df = pd.read_csv(GOOGLE_SHEET_CSV_URL)

    # -----------------------------------------
    # Clean Age column
    # -----------------------------------------
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
    df = df.dropna(subset=["Age"])
    df["Age"] = df["Age"].astype(int)

    # -----------------------------------------
    # Derive Mental Health Impact
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
    # Encode categorical columns
    # -----------------------------------------
    le = LabelEncoder()

    categorical_cols = [
        "Gender",
        "Primary_AI_Use",
        "Reliance_On_AI_For_Learning",
        "StressLevel"
    ]

    for col in categorical_cols:
        df[col] = le.fit_transform(df[col].astype(str))

    # -----------------------------------------
    # Remove remaining missing values
    # -----------------------------------------
    df = df.dropna()

    return df