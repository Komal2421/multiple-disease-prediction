"""
Utility helper functions for MedPredict AI application.
"""

def safe_float(value):
    """
    Safely converts a form input value to a float.
    Returns 0.0 if value is None or empty string.
    """
    if value is None or value == "":
        return 0.0
    return float(value)
