import sys
import traceback
from flask import render_template, request, redirect, url_for, session, flash
from app.services.auth_service import register_user, authenticate_user
from app.services.prediction_service import predict_disease
from app.services.gemini_service import generate_ai_explanation

def register_routes(app, limiter=None):
    """
    Registers HTTP routes, rate limiters, and request security hooks onto the Flask application instance.
    """
    
    @app.before_request
    def require_login():
        allowed_endpoints = ['login', 'register', 'static']
        if request.endpoint and request.endpoint not in allowed_endpoints:
            if 'user_id' not in session:
                return redirect(url_for('login'))

    @app.route("/")
    def home():
        return render_template("index.html", username=session.get('username'))

    # Helper decorator for optional rate limiting
    def limit(rate_spec):
        if limiter:
            return limiter.limit(rate_spec)
        return lambda f: f

    @app.route("/register", methods=["GET", "POST"])
    @limit("5 per minute")
    def register():
        if 'user_id' in session:
            return redirect(url_for('home'))
            
        if request.method == "POST":
            username = request.form.get("username", "")
            password = request.form.get("password", "")
            confirm_password = request.form.get("confirm_password", "")
            
            success, err_msg = register_user(username, password, confirm_password)
            if success:
                flash("Registration successful! Please login.", "success")
                return redirect(url_for('login'))
            else:
                flash(err_msg, "danger")
                # Remain on registration page to display inline errors
                return render_template("register.html")
                
        return render_template("register.html")

    @app.route("/login", methods=["GET", "POST"])
    @limit("5 per minute")
    def login():
        if 'user_id' in session:
            return redirect(url_for('home'))
            
        if request.method == "POST":
            username = request.form.get("username", "")
            password = request.form.get("password", "")
            client_ip = request.remote_addr or "127.0.0.1"
            
            user, err_msg = authenticate_user(username, password, ip_address=client_ip)
            if user:
                # Regenerate session ID to prevent Session Fixation attacks
                session.clear()
                session.permanent = True
                session['user_id'] = user['id']
                session['username'] = user['username']
                flash(f"Welcome back, {user['username']}!", "success")
                return redirect(url_for('home'))
            else:
                flash(err_msg, "danger")
                return render_template("login.html")
                
        return render_template("login.html")

    @app.route("/logout")
    def logout():
        session.clear()
        flash("You have been logged out.", "success")
        return redirect(url_for('login'))

    @app.route("/predict", methods=["POST"])
    def predict():
        disease = request.form["disease"]
        
        eval_result = predict_disease(disease, request.form)
        
        return render_template(
            "index.html",
            prediction=eval_result["prediction"],
            status=eval_result["status"],
            confidence=eval_result["confidence"],
            risk_level=eval_result["risk_level"],
            explanation="PENDING",
            disease=eval_result["disease_display"],
            input_data=eval_result["feature_dict"],
            username=session.get('username')
        )

    @app.route("/get_explanation", methods=["POST"])
    def get_explanation():
        if 'user_id' not in session:
            return {"explanation": "Personalized health insights are temporarily unavailable. Your assessment has been completed successfully."}, 401
            
        try:
            data = request.get_json() or {}
            disease = data.get("disease", "")
            prediction = data.get("prediction", "")
            risk_level = data.get("risk_level", "")
            confidence = data.get("confidence", 0.0)
            features = data.get("features", {})
            
            explanation = generate_ai_explanation(disease, prediction, risk_level, confidence, features)
            return {"explanation": explanation}
        except Exception as e:
            print(f"Error in get_explanation endpoint: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            return {"explanation": "Personalized health insights are temporarily unavailable. Your assessment has been completed successfully."}
