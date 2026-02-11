print("🚀 Script started")

import sys
import os
sys.path.append(os.path.dirname(__file__))

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from data_preprocessing import load_and_preprocess

print("📥 Loading dataset...")

df = load_and_preprocess()

print("📊 Data loaded. Shape:", df.shape)

X = df.drop(["StressLevel", "MentalHealthImpact"], axis=1)
y = df["StressLevel"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("🤖 Training model...")

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

pred = model.predict(X_test)

print("✅ Stress Level Accuracy:", accuracy_score(y_test, pred))
print("🎉 Script finished successfully")
