import os
import sys
import pandas as pd

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from training.utils import train_evaluate_compare

def main():
    data_path = os.path.join(project_root, "datasets", "breast_cancer.csv")
    print(f"Loading dataset from: {data_path}")
    df = pd.read_csv(data_path)
    
    df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})
    
    feature_cols = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean']
    X = df[feature_cols]
    y = df["diagnosis"]
    
    train_evaluate_compare(
        X=X,
        y=y,
        disease_name="Breast Cancer",
        model_save_path=os.path.join(project_root, "models", "breast_model.pkl"),
        scaler_save_path=os.path.join(project_root, "models", "breast_scaler.pkl")
    )

if __name__ == "__main__":
    main()