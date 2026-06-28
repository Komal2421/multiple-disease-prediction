import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score



df = pd.read_csv("datasets/parkinsons.csv")
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



# Parkinson's Distribution
plt.figure(figsize=(6,4))
df["status"].value_counts().plot(kind="bar")

plt.title("Parkinson's Disease Distribution")
plt.xlabel("Status")
plt.ylabel("Count")

plt.show()


# Age-independent feature example
plt.figure(figsize=(7,5))

plt.hist(df["MDVP:Fo(Hz)"], bins=20)

plt.title("Distribution of MDVP:Fo(Hz)")
plt.xlabel("MDVP:Fo(Hz)")
plt.ylabel("Frequency")

plt.show()


# Correlation Matrix
plt.figure(figsize=(10,8))

plt.imshow(df.corr(numeric_only=True), cmap="coolwarm")

plt.colorbar()

plt.xticks(range(len(df.select_dtypes(include='number').columns)),
           df.select_dtypes(include='number').columns,
           rotation=90)

plt.yticks(range(len(df.select_dtypes(include='number').columns)),
           df.select_dtypes(include='number').columns)

plt.title("Correlation Matrix")

plt.show()

X = df.drop(["name", "status"], axis=1)


y = df["status"]




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



joblib.dump(model, "models/parkinsons_model.pkl")

joblib.dump(scaler, "models/parkinsons_scaler.pkl")

print("\nParkinson's Model Saved Successfully")