import os
import sys
import pandas as pd

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from training.utils import train_evaluate_compare

def main():
    data_path = os.path.join(project_root, "datasets", "heart.csv")
    print(f"Loading dataset from: {data_path}")
    df = pd.read_csv(data_path)
    
    X = df.drop("target", axis=1)
    y = df["target"]
    
    train_evaluate_compare(
        X=X,
        y=y,
        disease_name="Heart Disease",
        model_save_path=os.path.join(project_root, "models", "heart_model.pkl"),
        scaler_save_path=os.path.join(project_root, "models", "heart_scaler.pkl")
    )

if __name__ == "__main__":
    main()