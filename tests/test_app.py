import os
import sys
import unittest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app import app
from app.utils.helpers import safe_float

class MedPredictSecurityTestCase(unittest.TestCase):
    def setUp(self):
        app.config['TESTING'] = True
        app.config['WTF_CSRF_ENABLED'] = False
        self.client = app.test_client()

    def test_app_exists(self):
        """Verify the Flask application instance is correctly created."""
        self.assertIsNotNone(app)

    def test_login_page_renders(self):
        """Verify the login page renders successfully with HTTP 200."""
        response = self.client.get('/login')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Login', response.data)

    def test_register_page_renders(self):
        """Verify the register page renders successfully with HTTP 200."""
        response = self.client.get('/register')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Create Account', response.data)

    def test_unauthenticated_redirect(self):
        """Verify unauthenticated users attempting to access home page are redirected to login."""
        response = self.client.get('/')
        self.assertEqual(response.status_code, 302)
        self.assertTrue(response.headers['Location'].endswith('/login'))

    def test_safe_float_helper(self):
        """Verify helper utility for safe float conversion."""
        self.assertEqual(safe_float("12.5"), 12.5)
        self.assertEqual(safe_float(""), 0.0)
        self.assertEqual(safe_float(None), 0.0)

if __name__ == '__main__':
    unittest.main()
