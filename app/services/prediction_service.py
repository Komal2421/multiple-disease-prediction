import os
import joblib
import numpy as np
from app.utils.helpers import safe_float

# Base directory relative to project root
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Load machine learning models and scalers
diabetes_model = joblib.load(os.path.join(MODELS_DIR, "diabetes_model.pkl"))
heart_model = joblib.load(os.path.join(MODELS_DIR, "heart_model.pkl"))
parkinsons_model = joblib.load(os.path.join(MODELS_DIR, "parkinsons_model.pkl"))
breast_model = joblib.load(os.path.join(MODELS_DIR, "breast_model.pkl"))

diabetes_scaler = joblib.load(os.path.join(MODELS_DIR, "diabetes_scaler.pkl"))
heart_scaler = joblib.load(os.path.join(MODELS_DIR, "heart_scaler.pkl"))
parkinsons_scaler = joblib.load(os.path.join(MODELS_DIR, "parkinsons_scaler.pkl"))
breast_scaler = joblib.load(os.path.join(MODELS_DIR, "breast_scaler.pkl"))

def predict_disease(disease, form_data):
    """
    Executes ML prediction for the requested disease using submitted form data.
    Returns a dictionary containing prediction results, status, confidence, risk level, display name, and feature dict.
    """
    result = ""
    status = "safe"
    feature_dict = {}
    p_disease = 0.0

    if disease == "diabetes":
        feature_dict = {
            "Pregnancies": safe_float(form_data.get("Pregnancies")),
            "Glucose": safe_float(form_data.get("Glucose")),
            "Blood Pressure": safe_float(form_data.get("BloodPressure")),
            "Skin Thickness": safe_float(form_data.get("SkinThickness")),
            "Insulin": safe_float(form_data.get("Insulin")),
            "BMI": safe_float(form_data.get("BMI")),
            "Diabetes Pedigree Function": safe_float(form_data.get("DiabetesPedigreeFunction")),
            "Age": safe_float(form_data.get("Age"))
        }
        features = np.array([list(feature_dict.values())])
        features_scaled = diabetes_scaler.transform(features)
        
        prediction = diabetes_model.predict(features_scaled)
        proba = diabetes_model.predict_proba(features_scaled)[0]
        p_disease = proba[1]
        
        if prediction[0] == 1:
            result = "Diabetes Detected"
            status = "danger"
        else:
            result = "No Diabetes"
            status = "safe"

    elif disease == "heart":
        feature_dict = {
            "Age": safe_float(form_data.get("age")),
            "Sex (1=Male, 0=Female)": safe_float(form_data.get("sex")),
            "Chest Pain Type (0-3)": safe_float(form_data.get("cp")),
            "Resting Blood Pressure": safe_float(form_data.get("trestbps")),
            "Cholesterol": safe_float(form_data.get("chol")),
            "Fasting Blood Sugar (1=True, 0=False)": safe_float(form_data.get("fbs")),
            "Resting ECG (0-2)": safe_float(form_data.get("restecg")),
            "Max Heart Rate": safe_float(form_data.get("thalach")),
            "Exercise Angina (1=Yes, 0=No)": safe_float(form_data.get("exang")),
            "Old Peak": safe_float(form_data.get("oldpeak")),
            "Slope (0-2)": safe_float(form_data.get("slope")),
            "Major Vessels (0-4)": safe_float(form_data.get("ca")),
            "Thalassemia (0-3)": safe_float(form_data.get("thal"))
        }
        features = np.array([list(feature_dict.values())])
        features_scaled = heart_scaler.transform(features)
        
        prediction = heart_model.predict(features_scaled)
        proba = heart_model.predict_proba(features_scaled)[0]
        p_disease = proba[1]
        
        if prediction[0] == 1:
            result = "Heart Disease Detected"
            status = "danger"
        else:
            result = "No Heart Disease"
            status = "safe"

    elif disease == "parkinsons":
        feature_dict = {
            "MDVP:Fo(Hz)": safe_float(form_data.get("mdvp_fo")),
            "MDVP:Fhi(Hz)": safe_float(form_data.get("mdvp_fhi")),
            "MDVP:Flo(Hz)": safe_float(form_data.get("mdvp_flo")),
            "MDVP:Jitter(%)": safe_float(form_data.get("mdvp_jitter")),
            "MDVP:Shimmer": safe_float(form_data.get("mdvp_shimmer"))
        }
        features = np.array([list(feature_dict.values())])
        features_scaled = parkinsons_scaler.transform(features)
        
        prediction = parkinsons_model.predict(features_scaled)
        proba = parkinsons_model.predict_proba(features_scaled)[0]
        p_disease = proba[1]
        
        if prediction[0] == 1:
            result = "Parkinson's Detected"
            status = "danger"
        else:
            result = "No Parkinson's"
            status = "safe"

    elif disease == "breast":
        feature_dict = {
            "Radius Mean": safe_float(form_data.get("radius_mean")),
            "Texture Mean": safe_float(form_data.get("texture_mean")),
            "Perimeter Mean": safe_float(form_data.get("perimeter_mean")),
            "Area Mean": safe_float(form_data.get("area_mean")),
            "Smoothness Mean": safe_float(form_data.get("smoothness_mean"))
        }
        features = np.array([list(feature_dict.values())])
        features_scaled = breast_scaler.transform(features)
        
        prediction = breast_model.predict(features_scaled)
        proba = breast_model.predict_proba(features_scaled)[0]
        p_disease = proba[1]
        
        if prediction[0] == 1:
            result = "Breast Cancer Detected"
            status = "danger"
        else:
            result = "No Cancer"
            status = "safe"

    confidence_val = p_disease if status == "danger" else (1.0 - p_disease)
    confidence_pct = round(confidence_val * 100, 1)
    
    if p_disease >= 0.70:
        risk_level = "High"
    elif p_disease >= 0.35:
        risk_level = "Moderate"
    else:
        risk_level = "Low"
        
    disease_display = {
        "diabetes": "Diabetes",
        "heart": "Heart Disease",
        "parkinsons": "Parkinson's Disease",
        "breast": "Breast Cancer"
    }.get(disease, disease.capitalize())

    return {
        "prediction": result,
        "status": status,
        "confidence": confidence_pct,
        "risk_level": risk_level,
        "disease_display": disease_display,
        "feature_dict": feature_dict
    }
