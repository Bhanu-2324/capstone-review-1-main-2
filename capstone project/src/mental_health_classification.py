print("🚀 Mental Health Classification Script Started")

import sys
import os
sys.path.append(os.path.dirname(__file__))

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from data_preprocessing import load_and_preprocess

# Load data
df = load_and_preprocess()
print("📊 Dataset shape:", df.shape)

# Features & Target
X = df.drop(["MentalHealthImpact", "StressLevel"], axis=1)
y = df["MentalHealthImpact"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = DecisionTreeClassifier(max_depth=5, random_state=42)
model.fit(X_train, y_train)

# Prediction
pred = model.predict(X_test)

# Result
print("✅ Mental Health Impact Accuracy:", accuracy_score(y_test, pred))
print("🎉 Mental Health Classification Completed")
