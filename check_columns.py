import pandas as pd

print("Parkinson's Dataset Columns")
df = pd.read_csv("datasets/parkinsons.csv")
print(df.columns)

print("\nBreast Cancer Dataset Columns")
df = pd.read_csv("datasets/breast_cancer.csv")
print(df.columns)