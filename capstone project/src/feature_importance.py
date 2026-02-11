import sys
import os
sys.path.append(os.path.dirname(__file__))

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from data_preprocessing import load_and_preprocess

# Load data
df = load_and_preprocess()

# Features & target
X = df.drop(["StressLevel", "MentalHealthImpact"], axis=1)
y = df["StressLevel"]

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Get feature importance
importances = model.feature_importances_
features = X.columns

# Plot
plt.figure()
plt.barh(features, importances)
plt.xlabel("Importance Score")
plt.title("Feature Importance for Stress Prediction")
plt.tight_layout()
plt.show()
