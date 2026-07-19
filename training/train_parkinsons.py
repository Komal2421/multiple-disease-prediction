import os
import sys
import pandas as pd

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from training.utils import train_evaluate_compare

def main():
    data_path = os.path.join(project_root, "datasets", "parkinsons.csv")
    print(f"Loading dataset from: {data_path}")
    df = pd.read_csv(data_path)
    
    feature_cols = ['MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)', 'MDVP:Shimmer']
    X = df[feature_cols]
    y = df["status"]
    
    train_evaluate_compare(
        X=X,
        y=y,
        disease_name="Parkinson's Disease",
        model_save_path=os.path.join(project_root, "models", "parkinsons_model.pkl"),
        scaler_save_path=os.path.join(project_root, "models", "parkinsons_scaler.pkl")
    )

if __name__ == "__main__":
    main()