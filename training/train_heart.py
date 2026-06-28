import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score




df = pd.read_csv("datasets/heart.csv")




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




# Target Distribution
plt.figure(figsize=(6,4))
df["target"].value_counts().plot(kind="bar")

plt.title("Heart Disease Distribution")
plt.xlabel("Target")
plt.ylabel("Count")

plt.show()


# Age Distribution
plt.figure(figsize=(7,5))

plt.hist(df["age"], bins=15)

plt.title("Age Distribution")

plt.xlabel("Age")

plt.ylabel("Frequency")

plt.show()


# Cholesterol Distribution
plt.figure(figsize=(7,5))

plt.hist(df["chol"], bins=20)

plt.title("Cholesterol Distribution")

plt.xlabel("Cholesterol")

plt.ylabel("Frequency")

plt.show()


# Correlation Matrix
plt.figure(figsize=(10,8))

plt.imshow(df.corr(), cmap="coolwarm")

plt.colorbar()

plt.xticks(range(len(df.columns)), df.columns, rotation=90)

plt.yticks(range(len(df.columns)), df.columns)

plt.title("Correlation Matrix")

plt.show()




X = df.drop("target", axis=1)

y = df["target"]




X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)




scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)



model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)




y_pred = model.predict(X_test)




accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy :", accuracy)



joblib.dump(model, "models/heart_model.pkl")

joblib.dump(scaler, "models/heart_scaler.pkl")

print("\nHeart Disease Model Saved Successfully")