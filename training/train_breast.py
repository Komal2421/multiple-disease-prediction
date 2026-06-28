import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score




df = pd.read_csv("datasets/breast_cancer.csv")
print(df.columns)



print("First Five Rows")
print(df.head())

print("\nDataset Shape")
print(df.shape)

print("\nColumn Names")
print(df.columns)

print("\nDataset Information")
print(df.info())

print("\nStatistical Summary")
print(df.describe())

print("\nMissing Values")
print(df.isnull().sum())


df["diagnosis"] = df["diagnosis"].map({"M":1, "B":0})



if "id" in df.columns:
    df = df.drop("id", axis=1)

if "Unnamed: 32" in df.columns:
    df = df.drop("Unnamed: 32", axis=1)



plt.figure(figsize=(6,4))

df["diagnosis"].value_counts().plot(kind="bar")

plt.title("Breast Cancer Distribution")

plt.xlabel("Diagnosis")

plt.ylabel("Count")

plt.show()


# Radius Mean Distribution
plt.figure(figsize=(7,5))

plt.hist(df["radius_mean"], bins=20)

plt.title("Radius Mean Distribution")

plt.xlabel("Radius Mean")

plt.ylabel("Frequency")

plt.show()



plt.figure(figsize=(10,8))

plt.imshow(df.corr(), cmap="coolwarm")

plt.colorbar()

plt.xticks(range(len(df.columns)), df.columns, rotation=90)

plt.yticks(range(len(df.columns)), df.columns)

plt.title("Correlation Matrix")

plt.show()




X = df.drop("diagnosis", axis=1)

y = df["diagnosis"]



X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)



scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)




model = LogisticRegression(max_iter=5000)

model.fit(X_train, y_train)




y_pred = model.predict(X_test)



accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy :", accuracy)




joblib.dump(model, "models/breast_model.pkl")

joblib.dump(scaler, "models/breast_scaler.pkl")

print("\nBreast Cancer Model Saved Successfully")