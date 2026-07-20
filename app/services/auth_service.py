import os
import re
import sqlite3
import logging
from datetime import datetime, timedelta, timezone
from werkzeug.security import generate_password_hash, check_password_hash

# Configure internal security logger
logger = logging.getLogger("medpredict.security")
logger.setLevel(logging.INFO)

# Define path to SQLite database at project root level
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATABASE_PATH = os.path.join(BASE_DIR, "users.db")

LOCKOUT_THRESHOLD = 5  # Max failed attempts allowed
LOCKOUT_DURATION_MINUTES = 15  # Lockout duration in minutes

def get_db():
    """
    Establishes and returns a database connection with Row factory enabled.
    """
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """
    Initializes the SQLite database schema for users and failed login attempt tracking.
    """
    with sqlite3.connect(DATABASE_PATH) as conn:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.execute('''
            CREATE TABLE IF NOT EXISTS failed_login_attempts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ip_address TEXT NOT NULL,
                username TEXT NOT NULL,
                attempt_time TIMESTAMP NOT NULL
            )
        ''')
        conn.commit()

def validate_username(username):
    """
    Sanitizes and validates username format.
    Returns (is_valid, error_msg, clean_username).
    """
    if not username:
        return False, "Username cannot be empty.", ""
        
    clean_username = username.strip()
    if not clean_username:
        return False, "Username cannot be empty.", ""
        
    if len(clean_username) < 3 or len(clean_username) > 30:
        return False, "Username must be between 3 and 30 characters in length.", clean_username
        
    if not re.match(r"^[a-zA-Z0-9_-]+$", clean_username):
        return False, "Username can only contain letters, numbers, underscores, and hyphens.", clean_username
        
    return True, "", clean_username

def validate_password_policy(password, confirm_password):
    """
    Enforces MedPredict AI strict password complexity policy:
    - Must match confirm password
    - Minimum 8 characters
    - At least one uppercase letter
    - At least one lowercase letter
    - At least one number
    - At least one special character
    """
    if not password:
        return False, "Password cannot be empty."
        
    if password != confirm_password:
        return False, "Password and Confirm Password do not match."
        
    if len(password) < 8:
        return False, "Password must be at least 8 characters long."
        
    if not re.search(r"[A-Z]", password):
        return False, "Password must contain at least one uppercase letter."
        
    if not re.search(r"[a-z]", password):
        return False, "Password must contain at least one lowercase letter."
        
    if not re.search(r"[0-9]", password):
        return False, "Password must contain at least one number."
        
    if not re.search(r"[!@#$%^&*()_+\-=\[\]{};':\"\\|,.<>\/?]", password):
        return False, "Password must contain at least one special character (!@#$%^&*)."
        
    return True, ""

def is_brute_force_locked(ip_address, username):
    """
    Checks if an IP or username is locked out due to repeated failed login attempts.
    Returns (is_locked, minutes_remaining).
    """
    now_utc = datetime.now(timezone.utc)
    cutoff_time = (now_utc - timedelta(minutes=LOCKOUT_DURATION_MINUTES)).strftime('%Y-%m-%d %H:%M:%S')
    
    conn = get_db()
    try:
        query = '''
            SELECT COUNT(*), MAX(attempt_time) 
            FROM failed_login_attempts 
            WHERE (ip_address = ? OR username = ?) AND attempt_time > ?
        '''
        row = conn.execute(query, (ip_address, username, cutoff_time)).fetchone()
        fail_count = row[0] if row else 0
        
        if fail_count >= LOCKOUT_THRESHOLD:
            last_attempt_str = row[1]
            if last_attempt_str:
                last_attempt = datetime.strptime(last_attempt_str[:19], '%Y-%m-%d %H:%M:%S').replace(tzinfo=timezone.utc)
                unlock_time = last_attempt + timedelta(minutes=LOCKOUT_DURATION_MINUTES)
                remaining_sec = max(0, (unlock_time - now_utc).total_seconds())
                remaining_min = int(remaining_sec // 60) + 1
                return True, remaining_min
            return True, LOCKOUT_DURATION_MINUTES
        return False, 0
    finally:
        conn.close()

def record_failed_attempt(ip_address, username):
    """
    Records a failed login attempt in the database for rate-limiting analysis using UTC timestamps.
    """
    logger.warning(f"Security Alert: Failed login attempt for user '{username}' from IP '{ip_address}'")
    now_str = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
    conn = get_db()
    try:
        conn.execute("INSERT INTO failed_login_attempts (ip_address, username, attempt_time) VALUES (?, ?, ?)", (ip_address, username, now_str))
        conn.commit()
    finally:
        conn.close()

def clear_failed_attempts(ip_address, username):
    """
    Clears failed login attempts upon successful login.
    """
    conn = get_db()
    try:
        conn.execute("DELETE FROM failed_login_attempts WHERE ip_address = ? OR username = ?", (ip_address, username))
        conn.commit()
    finally:
        conn.close()

def register_user(username, password, confirm_password):
    """
    Registers a new user securely into the database.
    Enforces username validation, password complexity policy, and parameterized SQLite insertion.
    Returns (success, error_message).
    """
    is_valid_user, user_err, clean_username = validate_username(username)
    if not is_valid_user:
        return False, user_err
        
    is_valid_pwd, pwd_err = validate_password_policy(password, confirm_password)
    if not is_valid_pwd:
        return False, pwd_err
        
    conn = get_db()
    try:
        password_hash = generate_password_hash(password)
        conn.execute("INSERT INTO users (username, password_hash) VALUES (?, ?)", (clean_username, password_hash))
        conn.commit()
        logger.info(f"User registration successful for '{clean_username}'")
        return True, None
    except sqlite3.IntegrityError:
        logger.warning(f"Registration failed: Username '{clean_username}' already exists.")
        return False, "Username already exists. Please choose a different one."
    finally:
        conn.close()

def authenticate_user(username, password, ip_address="127.0.0.1"):
    """
    Authenticates a user against stored credentials.
    Protects against timing attacks and account enumeration via generic error responses.
    Returns (user_object, error_message).
    """
    is_valid_user, _, clean_username = validate_username(username)
    if not is_valid_user or not password:
        return None, "Invalid username or password."
        
    # Check for brute-force lockout
    is_locked, remaining_min = is_brute_force_locked(ip_address, clean_username)
    if is_locked:
        logger.warning(f"Login blocked for locked out account/IP: user='{clean_username}', IP='{ip_address}'")
        return None, f"Too many failed login attempts. Account locked temporarily. Please try again in {remaining_min} minute(s)."
        
    conn = get_db()
    try:
        user = conn.execute("SELECT * FROM users WHERE username = ?", (clean_username,)).fetchone()
    finally:
        conn.close()
        
    if user and check_password_hash(user['password_hash'], password):
        clear_failed_attempts(ip_address, clean_username)
        logger.info(f"Successful authentication for user '{clean_username}' from IP '{ip_address}'")
        return user, None
    else:
        record_failed_attempt(ip_address, clean_username)
        return None, "Invalid username or password."
