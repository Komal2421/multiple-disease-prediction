from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# ---------------- MODELS ----------------
diabetes_model = joblib.load("models/diabetes_model.pkl")
heart_model = joblib.load("models/heart_model.pkl")
parkinsons_model = joblib.load("models/parkinsons_model.pkl")
breast_model = joblib.load("models/breast_model.pkl")

# ---------------- SCALERS ----------------
diabetes_scaler = joblib.load("models/diabetes_scaler.pkl")
heart_scaler = joblib.load("models/heart_scaler.pkl")
parkinsons_scaler = joblib.load("models/parkinsons_scaler.pkl")
breast_scaler = joblib.load("models/breast_scaler.pkl")


# ✅ SAFE CONVERSION FUNCTION (IMPORTANT FIX)
def safe_float(value):
    if value is None or value == "":
        return 0.0
    return float(value)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    disease = request.form["disease"]
    result = ""
    status = "safe"

    # ---------------- DIABETES ----------------
    if disease == "diabetes":

        features = np.array([[
            safe_float(request.form["Pregnancies"]),
            safe_float(request.form["Glucose"]),
            safe_float(request.form["BloodPressure"]),
            safe_float(request.form["SkinThickness"]),
            safe_float(request.form["Insulin"]),
            safe_float(request.form["BMI"]),
            safe_float(request.form["DiabetesPedigreeFunction"]),
            safe_float(request.form["Age"])
        ]])

        features = diabetes_scaler.transform(features)
        prediction = diabetes_model.predict(features)

        if prediction[0] == 1:
            result = "Diabetes Detected"
            status = "danger"
        else:
            result = "No Diabetes"
            status = "safe"

    # ---------------- HEART ----------------
    elif disease == "heart":

        features = np.array([[
            safe_float(request.form["age"]),
            safe_float(request.form["sex"]),
            safe_float(request.form["cp"]),
            safe_float(request.form["trestbps"]),
            safe_float(request.form["chol"]),
            safe_float(request.form["fbs"]),
            safe_float(request.form["restecg"]),
            safe_float(request.form["thalach"]),
            safe_float(request.form["exang"]),
            safe_float(request.form["oldpeak"]),
            safe_float(request.form["slope"]),
            safe_float(request.form["ca"]),
            safe_float(request.form["thal"])
        ]])

        features = heart_scaler.transform(features)
        prediction = heart_model.predict(features)

        if prediction[0] == 1:
            result = "Heart Disease Detected"
            status = "danger"
        else:
            result = "No Heart Disease"
            status = "safe"

    # ---------------- PARKINSON'S ----------------
    elif disease == "parkinsons":

        features = np.array([[
            safe_float(request.form["mdvp_fo"]),
            safe_float(request.form["mdvp_fhi"]),
            safe_float(request.form["mdvp_flo"]),
            safe_float(request.form["mdvp_jitter"]),
            safe_float(request.form["mdvp_shimmer"])
        ]])

        features = parkinsons_scaler.transform(features)
        prediction = parkinsons_model.predict(features)

        if prediction[0] == 1:
            result = "Parkinson's Detected"
            status = "danger"
        else:
            result = "No Parkinson's"
            status = "safe"

    # ---------------- BREAST CANCER ----------------
    elif disease == "breast":

        features = np.array([[
            safe_float(request.form["radius_mean"]),
            safe_float(request.form["texture_mean"]),
            safe_float(request.form["perimeter_mean"]),
            safe_float(request.form["area_mean"]),
            safe_float(request.form["smoothness_mean"])
        ]])

        features = breast_scaler.transform(features)
        prediction = breast_model.predict(features)

        if prediction[0] == 1:
            result = "Breast Cancer Detected"
            status = "danger"
        else:
            result = "No Cancer"
            status = "safe"

    return render_template("index.html", prediction=result, status=status)


if __name__ == "__main__":
    app.run(debug=True)