# MedPredict AI Architecture Documentation

## Overview
MedPredict AI is a production-grade Flask web application that provides multi-disease risk assessment (Diabetes, Heart Disease, Parkinson's Disease, and Breast Cancer) using machine learning classifiers, complemented by personalized AI clinical explanations powered by Google Gemini GenAI SDK.

---

## Directory Structure

```
MedPredict-AI/
│
├── app/                        # Application Package
│   ├── __init__.py             # Flask App Factory and App Initialization
│   ├── routes.py               # Application Endpoint Handlers & Session Guards
│   ├── services/               # Core Business & Integration Services
│   │   ├── __init__.py
│   │   ├── auth_service.py     # SQLite Database Operations & User Password Hashing
│   │   ├── prediction_service.py # Scikit-Learn Model Loading, Feature Scaling & Inference
│   │   └── gemini_service.py   # Google Gemini GenAI SDK Integration & Model Fallbacks
│   ├── utils/                  # Helper Utilities
│   │   ├── __init__.py
│   │   └── helpers.py          # Input Data Formatting & Sanitization
│   ├── static/                 # Static Assets (CSS, JS, Images)
│   │   ├── healthcare_bg.png
│   │   ├── script.js
│   │   └── style.css
│   └── templates/              # Jinja2 HTML Templates
│       ├── index.html
│       ├── login.html
│       └── register.html
│
├── models/                     # Trained Scikit-Learn Model Binaries (.pkl)
├── datasets/                   # Medical Training Datasets (.csv)
├── training/                   # Model Training & Evaluation Pipeline Scripts
├── evaluation/                 # Model Evaluation Reports (JSON/CSV)
├── tests/                      # Automated Unit Test Suite
│   ├── __init__.py
│   └── test_app.py
├── docs/                       # Project Technical Documentation
│   └── architecture.md
├── run.py                      # Primary Entry Point for Local Development
├── app.py                      # Backward-Compatibility Shim (delegates to app)
├── requirements.txt            # Python Dependencies Specification
├── Dockerfile                  # Container Build Instructions
├── .dockerignore               # Docker Build Ignore Patterns
└── README.md                   # Project Overview & Usage Instructions
```

---

## Layer Responsibilities

### 1. Presentation Layer (`app/templates/`, `app/static/`)
- Handles user interface rendering using Jinja2 templates and modern CSS styling.
- Asynchronous API calls via client-side JavaScript (`script.js`) for fetching Gemini AI recommendations without page reloads.

### 2. Controller & Routing Layer (`app/routes.py`)
- Defines application endpoints: `/`, `/login`, `/register`, `/logout`, `/predict`, `/get_explanation`.
- Enforces session security via `@app.before_request`.
- Delegates business tasks to service components.

### 3. Service Layer (`app/services/`)
- `auth_service.py`: Encapsulates user database access (`users.db`) using Werkzeug password hashing.
- `prediction_service.py`: Manages Scikit-Learn model loading (`joblib`), input scaling, prediction score computation, and risk classification.
- `gemini_service.py`: Integrates Google Gemini GenAI SDK with multi-model fallback retry logic for high availability.

### 4. Machine Learning & Artifacts (`models/`, `training/`)
- Pre-trained models stored in `models/`.
- Modular offline training pipeline in `training/utils.py` and dataset-specific training scripts.

---

## Running & Deployment

### Local Development
```bash
python run.py
```

### Docker Container Deployment
```bash
docker build -t medpredict-ai .
docker run -p 5000:5000 medpredict-ai
```

### Render Deployment
Production environments like Render execute Gunicorn via:
```bash
gunicorn run:app
```
or
```bash
gunicorn app:app
```
Both invocation methods are fully supported.
