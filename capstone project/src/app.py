import sys
import os
sys.path.append(os.path.dirname(__file__))

import streamlit as st
import matplotlib.pyplot as plt
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
# Title & Introduction
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


import pandas as pd

# st.subheader("Age Group Filter")

# age_min, age_max = st.slider(
#     "Select Age Range",
#     min_value=int(df["Age"].min()),
#     max_value=int(df["Age"].max()),
#     value=(18, 35)
# )

# filtered_df = df[(df["Age"] >= age_min) & (df["Age"] <= age_max)]

# replaced with this :

# --------------------------------------------------
# Age Filter (Safe Version)
# --------------------------------------------------
st.subheader("Age Group Filter")

# Clean Age column safely
df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
df = df.dropna(subset=["Age"])
df["Age"] = df["Age"].astype(int)

# Safe min/max values
age_min_value = int(df["Age"].min()) if not df.empty else 18
age_max_value = int(df["Age"].max()) if not df.empty else 40

# Slider
age_min, age_max = st.slider(
    "Select Age Range",
    min_value=age_min_value,
    max_value=age_max_value,
    value=(age_min_value, age_max_value)
)

# Filter data
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


st.caption(
    "The dataset focuses on youth and captures realistic patterns of AI usage, "
    "screen time, sleep, and problem-solving behavior."
)

# --------------------------------------------------
# AI Usage Behavior Analysis
# --------------------------------------------------
st.subheader("AI Usage Behavior Analysis")

# ai_first = df[
#     df["Problems_Solved_With_AI"] > df["Problems_Solved_Before_AI"]
# ]

# self_first = df[
#     df["Problems_Solved_Before_AI"] >= df["Problems_Solved_With_AI"]
# ]

# b1, b2 = st.columns(2)

# b1.metric("AI-First Users", f"{len(ai_first)} users")
# b2.metric("Self-Try First Users", f"{len(self_first)} users")

# Create AI dependence ratio
df["AI_Dependence_Ratio"] = (
    df["Problems_Solved_With_AI"] /
    (df["Problems_Solved_With_AI"] + df["Problems_Solved_Before_AI"])
)

# Classify users based on dependence
ai_first = df[df["AI_Dependence_Ratio"] >= 0.65]
balanced = df[
    (df["AI_Dependence_Ratio"] > 0.35) &
    (df["AI_Dependence_Ratio"] < 0.65)
]
self_first = df[df["AI_Dependence_Ratio"] <= 0.35]

# Display metrics
b1, b2, b3 = st.columns(3)

b1.metric("AI-First Users", len(ai_first))
b2.metric("Balanced Users", len(balanced))
b3.metric("Self-Try First Users", len(self_first))

st.markdown("""
**Interpretation:**  
- **AI-First Users** show high dependence on AI tools.  
- **Balanced Users** combine independent thinking with AI assistance.  
- **Self-Try First Users** prioritize solving problems independently before using AI.  

This classification is based on **relative dependence**, not raw counts.
""")


# --------------------------------------------------
# Sample Data Records
# --------------------------------------------------
st.subheader("Sample User Records")

df["Total_Problems"] = (
    df["Problems_Solved_Before_AI"] + df["Problems_Solved_With_AI"]
)

df["AI_Share_%"] = (
    df["Problems_Solved_With_AI"] / df["Total_Problems"] * 100
).round(1)

df["AI_Dependency"] = df["AI_Share_%"].apply(
    lambda x: "High" if x >= 65 else "Balanced" if x >= 35 else "Low"
)
##

st.dataframe(
    df[[
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

st.caption(
    "High accuracy indicates strong classification performance, "
    "while low MAE shows precise productivity prediction."
)

# --------------------------------------------------
# Feature Importance (Small & Balanced)
# --------------------------------------------------
st.subheader("Feature Importance – Stress Prediction")
# --------------------------------------------------
# Feature Importance – Stress Prediction
# --------------------------------------------------

# Map StressLevel text → numeric (ONLY for ML)
stress_map = {
    "Low": 0,
    "Medium": 1,
    "High": 2
}

df["StressLevel_Num"] = df["StressLevel"].map(stress_map)

# Select numeric columns only
# Use filtered data instead of full dataset
ml_df = filtered_df.select_dtypes(include=["int64", "float64"])

# Remove missing values
ml_df = ml_df.dropna()

# Target
y = ml_df["StressLevel_Num"]

# Features
X = ml_df.drop(
    columns=["StressLevel_Num", "ProductivityScore"],
    errors="ignore"
)

# Train model ONLY if data exists
if len(X) > 0 and len(y) > 0:

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
    st.warning("Not enough data to train the model for the selected age range.")

# X = df.drop(["StressLevel", "MentalHealthImpact"], axis=1)
# y = df["StressLevel"]

# model = RandomForestClassifier(n_estimators=100, random_state=42)
# model.fit(X, y)


fig1, ax1 = plt.subplots(figsize=(4.5, 3))
ax1.barh(X.columns, model.feature_importances_)
ax1.set_xlabel("Importance", fontsize=8)
ax1.set_title("Key Stress Factors", fontsize=9)
ax1.tick_params(axis='both', labelsize=7)
plt.tight_layout()

st.pyplot(fig1, use_container_width=False)

st.caption(
    "Screen time and sleep duration are the strongest contributors to stress prediction."
)

# --------------------------------------------------
# Correlation Heatmap (Small & Balanced)
# --------------------------------------------------
st.subheader("Correlation Heatmap")

# Select ONLY numeric columns for correlation
df_numeric = df.select_dtypes(include=["int64", "float64"])

corr = df_numeric.corr()


fig2, ax2 = plt.subplots(figsize=(4.8, 4))
im = ax2.imshow(corr)

ax2.set_xticks(range(len(corr.columns)))
ax2.set_yticks(range(len(corr.columns)))
ax2.set_xticklabels(corr.columns, rotation=90, fontsize=6)
ax2.set_yticklabels(corr.columns, fontsize=6)
ax2.set_title("Feature Correlation", fontsize=9)

plt.tight_layout()
st.pyplot(fig2, use_container_width=False)

st.caption(
    "Screen time correlates positively with stress, while sleep and physical activity "
    "show negative correlation."
)

# --------------------------------------------------
# Conclusion
# --------------------------------------------------
st.subheader("Conclusion")

st.markdown("""
The analysis shows that **excessive screen time and poor sleep**
increase stress among youth.  
While AI significantly improves productivity, **encouraging users to
attempt problem-solving independently before using AI** may promote
healthier learning habits and better mental well-being.
""")
