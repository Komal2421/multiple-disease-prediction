"""
Backward-compatibility entry point for MedPredict AI application.
Delegates application instance import to the new modular 'app' package.
"""

from app import app

if __name__ == "__main__":
    app.run(debug=True)