import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib


df = pd.read_csv("datasets/diabetes.csv")


X = df.drop("Outcome", axis=1)
y = df["Outcome"]


feature_names = X.columns


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)


y_pred = model.predict(X_test_scaled)

print("Accuracy:", accuracy_score(y_test, y_pred))


joblib.dump(model, "models/diabetes_model.pkl")
joblib.dump(scaler, "models/diabetes_scaler.pkl")
joblib.dump(feature_names, "models/diabetes_features.pkl")

print("Saved successfully!")