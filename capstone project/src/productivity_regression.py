print("🚀 Productivity Regression Script Started")

import sys
import os
sys.path.append(os.path.dirname(__file__))

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from data_preprocessing import load_and_preprocess

# Load data
df = load_and_preprocess()
print("📊 Dataset shape:", df.shape)

# Features & Target
X = df.drop(["ProductivityScore", "StressLevel", "MentalHealthImpact"], axis=1)
y = df["ProductivityScore"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction
pred = model.predict(X_test)

# Result
print("✅ Productivity MAE:", mean_absolute_error(y_test, pred))
print("🎉 Productivity Regression Completed")
