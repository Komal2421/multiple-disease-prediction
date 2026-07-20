import os
from datetime import timedelta
from flask import Flask
from flask_wtf.csrf import CSRFProtect
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from dotenv import load_dotenv

from app.services.auth_service import init_db
from app.routes import register_routes

load_dotenv(override=True)

csrf = CSRFProtect()
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"],
    storage_uri="memory://"
)

def create_app():
    """
    Application Factory for MedPredict AI Flask Application.
    Configures session security, CSRF protection, rate limiting, and database schema.
    """
    app = Flask(__name__)
    app.secret_key = os.environ.get("FLASK_SECRET_KEY", "medpredict_super_secret_key_123!")
    
    # Security Configurations for Session Cookies
    app.config["SESSION_COOKIE_HTTPONLY"] = True
    app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
    app.config["PERMANENT_SESSION_LIFETIME"] = timedelta(hours=2)
    
    # Secure Cookie flag enabled in Production environment
    is_prod = os.environ.get("FLASK_ENV", "development").lower() == "production"
    app.config["SESSION_COOKIE_SECURE"] = is_prod

    # Initialize Extensions
    csrf.init_app(app)
    limiter.init_app(app)
    
    # Initialize SQLite database schema
    init_db()
    
    # Register application routes
    register_routes(app, limiter)
    
    return app

# Instantiate app for direct module imports (e.g. gunicorn app:app)
app = create_app()
