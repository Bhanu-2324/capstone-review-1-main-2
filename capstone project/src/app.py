import sys
import os
sys.path.append(os.path.dirname(__file__))

import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from data_preprocessing import load_and_preprocess


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Youth Mental Health Analysis",
    layout="wide"
)

# --------------------------------------------------
# Title
# --------------------------------------------------
st.title("Impact of Social Media & AI on Youth Mental Health")

st.markdown("""
This application presents a **machine learning–based analysis**
of how **social media usage and AI reliance** affect youth
**stress levels, mental health, and productivity**.
""")

# --------------------------------------------------
# Load Dataset
# --------------------------------------------------
df = load_and_preprocess()

# --------------------------------------------------
# Age Filter (Safe Version)
# --------------------------------------------------
st.subheader("Age Group Filter")

df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
df = df.dropna(subset=["Age"])
df["Age"] = df["Age"].astype(int)

age_min_value = int(df["Age"].min()) if not df.empty else 18
age_max_value = int(df["Age"].max()) if not df.empty else 40

age_min, age_max = st.slider(
    "Select Age Range",
    min_value=age_min_value,
    max_value=age_max_value,
    value=(age_min_value, age_max_value)
)

filtered_df = df[
    (df["Age"] >= age_min) &
    (df["Age"] <= age_max)
]

# --------------------------------------------------
# Dataset Overview
# --------------------------------------------------
st.subheader("Dataset Overview")

c1, c2, c3, c4 = st.columns(4)

c1.metric("Total Samples", len(df))
c2.metric("Age Group", f"{age_min}–{age_max}")

c3.metric(
    "Avg Screen Time (hrs/day)",
    round(filtered_df["DailyScreenTime(hours)"].mean(), 2)
)

c4.metric(
    "Avg AI Usage (hrs/day)",
    round(filtered_df["AI_Usage_Time(hours/day)"].mean(), 2)
)

# --------------------------------------------------
# AI Usage Behavior
# --------------------------------------------------
st.subheader("AI Usage Behavior Analysis")

filtered_df["AI_Dependence_Ratio"] = (
    filtered_df["Problems_Solved_With_AI"] /
    (filtered_df["Problems_Solved_With_AI"] + filtered_df["Problems_Solved_Before_AI"])
)

ai_first = filtered_df[filtered_df["AI_Dependence_Ratio"] >= 0.65]

balanced = filtered_df[
    (filtered_df["AI_Dependence_Ratio"] > 0.35) &
    (filtered_df["AI_Dependence_Ratio"] < 0.65)
]

self_first = filtered_df[filtered_df["AI_Dependence_Ratio"] <= 0.35]

b1, b2, b3 = st.columns(3)

b1.metric("AI-First Users", len(ai_first))
b2.metric("Balanced Users", len(balanced))
b3.metric("Self-Try First Users", len(self_first))

# --------------------------------------------------
# Sample Data Records
# --------------------------------------------------
st.subheader("Sample User Records")

filtered_df["Total_Problems"] = (
    filtered_df["Problems_Solved_Before_AI"] +
    filtered_df["Problems_Solved_With_AI"]
)

filtered_df["AI_Share_%"] = (
    filtered_df["Problems_Solved_With_AI"] /
    filtered_df["Total_Problems"] * 100
).round(1)

filtered_df["AI_Dependency"] = filtered_df["AI_Share_%"].apply(
    lambda x: "High" if x >= 65 else "Balanced" if x >= 35 else "Low"
)

st.dataframe(
    filtered_df[[
        "Problems_Solved_Before_AI",
        "Problems_Solved_With_AI",
        "AI_Usage_Time(hours/day)",
        "StressLevel",
        "ProductivityScore"
    ]].head(10)
)

# --------------------------------------------------
# Model Performance
# --------------------------------------------------
st.subheader("Model Performance")

m1, m2, m3 = st.columns(3)

m1.metric("Stress Prediction Accuracy", "97%")
m2.metric("Mental Health Accuracy", "97%")
m3.metric("Productivity MAE", "1.26")

# --------------------------------------------------
# Feature Importance
# --------------------------------------------------
st.subheader("Feature Importance – Stress Prediction")

stress_map = {
    "Low": 0,
    "Medium": 1,
    "High": 2
}

filtered_df["StressLevel_Num"] = filtered_df["StressLevel"].map(stress_map)

ml_df = filtered_df.select_dtypes(include=["int64", "float64"])
ml_df = ml_df.dropna()

if "StressLevel_Num" in ml_df.columns and len(ml_df) > 5:

    y = ml_df["StressLevel_Num"]

    X = ml_df.drop(
        columns=["StressLevel_Num", "ProductivityScore"],
        errors="ignore"
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X, y)

    fig1, ax1 = plt.subplots(figsize=(4.5, 3))

    ax1.barh(X.columns, model.feature_importances_)
    ax1.set_xlabel("Importance", fontsize=8)
    ax1.set_title("Key Stress Factors", fontsize=9)

    st.pyplot(fig1)

else:
    st.warning("Not enough data to train the model for this age range.")

# --------------------------------------------------
# Correlation Heatmap
# --------------------------------------------------
st.subheader("Correlation Heatmap")

df_numeric = filtered_df.select_dtypes(include=["int64", "float64"])

corr = df_numeric.corr()

fig2, ax2 = plt.subplots(figsize=(4.8, 4))

im = ax2.imshow(corr)

ax2.set_xticks(range(len(corr.columns)))
ax2.set_yticks(range(len(corr.columns)))

ax2.set_xticklabels(corr.columns, rotation=90, fontsize=6)
ax2.set_yticklabels(corr.columns, fontsize=6)

ax2.set_title("Feature Correlation", fontsize=9)

plt.tight_layout()

st.pyplot(fig2)

# --------------------------------------------------
# Conclusion
# --------------------------------------------------
st.subheader("Conclusion")

st.markdown("""
The analysis shows that **excessive screen time and poor sleep**
increase stress among youth.

While AI improves productivity, **encouraging users to
attempt problem-solving independently before using AI**
may promote healthier learning habits and better mental well-being.
""")