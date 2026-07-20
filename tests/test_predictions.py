import os
import sys
import uuid
import unittest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app import app
from app.services.prediction_service import predict_disease
from app.services.auth_service import register_user, authenticate_user, validate_password_policy, is_brute_force_locked

class ComprehensiveAuthAndPredictionTestCase(unittest.TestCase):
    def setUp(self):
        app.config['TESTING'] = True
        app.config['WTF_CSRF_ENABLED'] = False
        self.client = app.test_client()

    def test_password_policy_validation(self):
        """Verify password complexity requirements."""
        # Weak passwords
        self.assertFalse(validate_password_policy("short", "short")[0])
        self.assertFalse(validate_password_policy("nouppercase1!", "nouppercase1!")[0])
        self.assertFalse(validate_password_policy("NOLOWERCASE1!", "NOLOWERCASE1!")[0])
        self.assertFalse(validate_password_policy("NoNumber!", "NoNumber!")[0])
        self.assertFalse(validate_password_policy("NoSpecial1", "NoSpecial1")[0])
        
        # Mismatched passwords
        self.assertFalse(validate_password_policy("StrongPass1!", "MismatchedPass1!")[0])
        
        # Valid strong password
        self.assertTrue(validate_password_policy("StrongPass1!", "StrongPass1!")[0])

    def test_auth_registration_and_login(self):
        """Verify secure registration and authentication flow."""
        test_username = f"user_{uuid.uuid4().hex[:8]}"
        strong_password = "SecurePassword123!"
        
        # Mismatched password registration fails
        success, err = register_user(test_username, strong_password, "WrongConfirm123!")
        self.assertFalse(success)
        
        # Valid registration succeeds
        success, err = register_user(test_username, strong_password, strong_password)
        self.assertTrue(success)
        
        # Duplicate registration fails
        success, err = register_user(test_username, strong_password, strong_password)
        self.assertFalse(success)
        self.assertIn("already exists", err)
        
        # Authentication with valid credentials succeeds
        user, err = authenticate_user(test_username, strong_password, ip_address="127.0.0.1")
        self.assertIsNotNone(user)
        self.assertIsNone(err)
        
        # Authentication with invalid password fails with generic message
        user, err = authenticate_user(test_username, "WrongPassword1!", ip_address="127.0.0.1")
        self.assertIsNone(user)
        self.assertEqual(err, "Invalid username or password.")

    def test_brute_force_lockout(self):
        """Verify brute force lockout after repeated failed attempts."""
        test_user = f"lockuser_{uuid.uuid4().hex[:8]}"
        test_ip = f"192.168.1.{uuid.uuid4().int % 200 + 10}"
        
        # Trigger 5 failed login attempts
        for _ in range(5):
            authenticate_user(test_user, "WrongPass1!", ip_address=test_ip)
            
        # 6th attempt should be locked out
        user, err = authenticate_user(test_user, "WrongPass1!", ip_address=test_ip)
        self.assertIsNone(user)
        self.assertIn("Too many failed login attempts", err)

    def test_diabetes_prediction(self):
        form_data = {
            "Pregnancies": "6",
            "Glucose": "148",
            "BloodPressure": "72",
            "SkinThickness": "35",
            "Insulin": "0",
            "BMI": "33.6",
            "DiabetesPedigreeFunction": "0.627",
            "Age": "50"
        }
        res = predict_disease("diabetes", form_data)
        self.assertIn(res["status"], ["safe", "danger"])

    def test_heart_prediction(self):
        form_data = {
            "age": "63", "sex": "1", "cp": "3", "trestbps": "145",
            "chol": "233", "fbs": "1", "restecg": "0", "thalach": "150",
            "exang": "0", "oldpeak": "2.3", "slope": "0", "ca": "0", "thal": "1"
        }
        res = predict_disease("heart", form_data)
        self.assertIn(res["status"], ["safe", "danger"])

    def test_parkinsons_prediction(self):
        form_data = {
            "mdvp_fo": "119.992", "mdvp_fhi": "157.302", "mdvp_flo": "74.997",
            "mdvp_jitter": "0.00784", "mdvp_shimmer": "0.04374"
        }
        res = predict_disease("parkinsons", form_data)
        self.assertIn(res["status"], ["safe", "danger"])

    def test_breast_prediction(self):
        form_data = {
            "radius_mean": "17.99", "texture_mean": "10.38", "perimeter_mean": "122.8",
            "area_mean": "1001", "smoothness_mean": "0.1184"
        }
        res = predict_disease("breast", form_data)
        self.assertIn(res["status"], ["safe", "danger"])

if __name__ == '__main__':
    unittest.main()
