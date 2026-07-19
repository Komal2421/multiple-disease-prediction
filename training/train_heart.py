"""
Train multiple classifiers on the Heart Disease dataset.
Compares: Logistic Regression, Random Forest, SVM, Decision Tree, KNN
Saves the best model and scaler as .pkl files for the Flask app.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# ─── Load & Prepare Data ──────────────────────────────────────────────────────
df = pd.read_csv("datasets/heart.csv")

X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ─── Define Classifiers ────────────────────────────────────────────────────────
classifiers = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM":                 SVC(kernel="rbf", random_state=42),
    "Decision Tree":       DecisionTreeClassifier(random_state=42),
    "KNN":                 KNeighborsClassifier(n_neighbors=5),
}

# ─── Train, Evaluate & Compare ────────────────────────────────────────────────
print("\n" + "="*60)
print("  HEART DISEASE — Multi-Classifier Accuracy Comparison")
print("="*60)

results = {}
trained_models = {}

for name, clf in classifiers.items():
    clf.fit(X_train_scaled, y_train)
    y_pred = clf.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc
    trained_models[name] = clf
    print(f"  {name:<25} Accuracy: {acc*100:.2f}%")

# ─── Pick Best Model ──────────────────────────────────────────────────────────
best_name = max(results, key=results.get)
best_model = trained_models[best_name]
best_acc   = results[best_name]

print("-"*60)
print(f"  ✅ Best Model : {best_name}")
print(f"  ✅ Accuracy   : {best_acc*100:.2f}%")
print("="*60)

# Detailed report for the winner
y_pred_best = best_model.predict(X_test_scaled)
print(f"\nClassification Report — {best_name}:")
print(classification_report(y_test, y_pred_best, target_names=["No Disease", "Heart Disease"]))

# ─── Save Best Model & Scaler ─────────────────────────────────────────────────
joblib.dump(best_model, "models/heart_model.pkl")
joblib.dump(scaler,     "models/heart_scaler.pkl")
joblib.dump(best_name,  "models/heart_best_model_name.pkl")

print(f"Saved '{best_name}' as models/heart_model.pkl")
print("Saved scaler as models/heart_scaler.pkl")