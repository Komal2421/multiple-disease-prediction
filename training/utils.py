import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

def train_evaluate_compare(
    X, 
    y, 
    disease_name, 
    model_save_path, 
    scaler_save_path
):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    scaler_scaled = StandardScaler()
    X_train_scaled = scaler_scaled.fit_transform(X_train)
    X_test_scaled = scaler_scaled.transform(X_test)
    
    scaler_identity = StandardScaler(with_mean=False, with_std=False)
    X_train_unscaled = scaler_identity.fit_transform(X_train)
    X_test_unscaled = scaler_identity.transform(X_test)
    
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    svm_model = SVC(probability=True, random_state=42)
    
    if disease_name.lower() == "diabetes":
        print("\nPerforming Hyperparameter Tuning for Diabetes Models...")
        from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
        cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        rf_param_dist = {
            'n_estimators': [50, 100, 200, 300],
            'max_depth': [None, 5, 10, 15, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None],
            'bootstrap': [True, False]
        }
        print("Tuning Random Forest...")
        rf_search = RandomizedSearchCV(
            estimator=RandomForestClassifier(random_state=42),
            param_distributions=rf_param_dist,
            n_iter=30,
            cv=cv_strategy,
            scoring='accuracy',
            n_jobs=-1,
            random_state=42
        )
        rf_search.fit(X_train_unscaled, y_train)
        rf_model = rf_search.best_estimator_
        print(f"Best Random Forest Parameters: {rf_search.best_params_}")
        
        svm_param_dist = {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
            'kernel': ['linear', 'rbf', 'poly'],
            'degree': [2, 3, 4]
        }
        print("Tuning SVM...")
        svm_search = RandomizedSearchCV(
            estimator=SVC(probability=True, random_state=42),
            param_distributions=svm_param_dist,
            n_iter=30,
            cv=cv_strategy,
            scoring='accuracy',
            n_jobs=-1,
            random_state=42
        )
        svm_search.fit(X_train_scaled, y_train)
        svm_model = svm_search.best_estimator_
        print(f"Best SVM Parameters: {svm_search.best_params_}")
        
    models_config = {
        "Logistic Regression": {
            "model": LogisticRegression(max_iter=5000, random_state=42),
            "requires_scaling": True
        },
        "Decision Tree": {
            "model": DecisionTreeClassifier(random_state=42),
            "requires_scaling": False
        },
        "Random Forest": {
            "model": rf_model,
            "requires_scaling": False
        },
        "Support Vector Machine (SVM)": {
            "model": svm_model,
            "requires_scaling": True
        },
        "K-Nearest Neighbors (KNN)": {
            "model": KNeighborsClassifier(n_neighbors=5),
            "requires_scaling": True
        }
    }
    
    if XGBOOST_AVAILABLE:
        models_config["XGBoost"] = {
            "model": XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42),
            "requires_scaling": False
        }
    else:
        print("XGBoost is not installed on this system. Skipping XGBoost gracefully...")
        
    results = []
    trained_models = {}
    
    print("\n" + "="*80)
    print(f" Training & Comparing Models for: {disease_name.upper()} ")
    print("="*80)
    
    for name, config in models_config.items():
        model = config["model"]
        requires_scaling = config["requires_scaling"]
        
        if requires_scaling:
            X_tr, X_te = X_train_scaled, X_test_scaled
        else:
            X_tr, X_te = X_train_unscaled, X_test_unscaled
            
        try:
            model.fit(X_tr, y_train)
            y_pred = model.predict(X_te)
            
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X_te)
                if proba.shape[1] > 1:
                    roc_auc = roc_auc_score(y_test, proba[:, 1])
                else:
                    roc_auc = 0.0
            else:
                roc_auc = 0.0
                
            acc = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            cm = confusion_matrix(y_test, y_pred)
            
            results.append({
                "algorithm": name,
                "accuracy": acc,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "roc_auc": roc_auc,
                "confusion_matrix": cm,
                "requires_scaling": requires_scaling
            })
            trained_models[name] = model
            
            print(f"[{name}] Training completed.")
            
        except Exception as e:
            print(f"Error training {name}: {e}")
            
    print("\n" + "-"*80)
    print(f" PERFORMANCE COMPARISON: {disease_name.upper()} ")
    print("-"*80)
    print(f"{'Algorithm':<30} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'ROC-AUC':<10}")
    print("-"*80)
    for r in results:
        print(f"{r['algorithm']:<30} | {r['accuracy']*100:-8.2f}% | {r['precision']*100:-8.2f}% | {r['recall']*100:-8.2f}% | {r['f1_score']*100:-8.2f}% | {r['roc_auc']*100:-8.2f}%")
    print("-"*80)
    
    print("\nConfusion Matrices:")
    for r in results:
        cm = r['confusion_matrix']
        print(f"\n{r['algorithm']}:")
        print(f"  TN: {cm[0,0]:<5} FP: {cm[0,1]:<5}")
        print(f"  FN: {cm[1,0]:<5} TP: {cm[1,1]:<5}")
        
    sorted_results = sorted(results, key=lambda x: (x["accuracy"], x["f1_score"]), reverse=True)
    
    if not sorted_results:
        print("Error: No models were successfully trained.")
        return
        
    best_result = sorted_results[0]
    best_name = best_result["algorithm"]
    best_model = trained_models[best_name]
    best_requires_scaling = best_result["requires_scaling"]
    
    print("\n" + "="*80)
    print(f" >>> BEST MODEL SELECTED FOR {disease_name.upper()}: {best_name} <<<")
    print(f"    Accuracy: {best_result['accuracy']*100:.2f}% (F1-Score: {best_result['f1_score']*100:.2f}%)")
    print("="*80)
    
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    joblib.dump(best_model, model_save_path)
    print(f"Saved best model to: {model_save_path}")
    
    if best_requires_scaling:
        joblib.dump(scaler_scaled, scaler_save_path)
        print(f"Saved standard scaler to: {scaler_save_path}")
    else:
        joblib.dump(scaler_identity, scaler_save_path)
        print(f"Saved pass-through scaler (no-op) to: {scaler_save_path}")
        
    try:
        results_df = pd.DataFrame([{
            "Algorithm": r["algorithm"],
            "Accuracy": r["accuracy"],
            "Precision": r["precision"],
            "Recall": r["recall"],
            "F1-Score": r["f1_score"],
            "ROC-AUC": r["roc_auc"],
            "Confusion-Matrix": r["confusion_matrix"].tolist()
        } for r in results])
        
        disease_clean = disease_name.lower().replace(" ", "_").replace("'", "")
        project_root = os.path.dirname(os.path.dirname(model_save_path))
        eval_dir = os.path.join(project_root, "evaluation")
        os.makedirs(eval_dir, exist_ok=True)
        eval_csv_path = os.path.join(eval_dir, f"{disease_clean}_evaluation.csv")
        eval_json_path = os.path.join(eval_dir, f"{disease_clean}_evaluation.json")
        
        results_df.to_csv(eval_csv_path, index=False)
        results_df.to_json(eval_json_path, orient="records", indent=4)
        print(f"Saved evaluation results to:\n - {eval_csv_path}\n - {eval_json_path}")
    except Exception as e:
        print(f"Warning: Could not save evaluation results files: {e}")
        
    print(f"Successfully finalized training for {disease_name}!")
