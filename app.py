from flask import Flask, render_template, request, redirect, url_for, session
import joblib
import os
import numpy as np
import time
from supabase import create_client, Client
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get base directory for absolute paths (crucial for Vercel)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "default-secret-key")

# Supabase configuration
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")

# Initialize supabase client safely
supabase = None
if supabase_url and supabase_key:
    try:
        supabase = create_client(supabase_url, supabase_key)
    except Exception as e:
        print(f"Error initializing Supabase: {e}")
else:
    print("Warning: SUPABASE_URL or SUPABASE_KEY missing from environment variables.")

# Load models using absolute paths
model = None
vectorizer = None

try:
    model_path = os.path.join(BASE_DIR, "best_model.pkl")
    vectorizer_path = os.path.join(BASE_DIR, "tfidf_vectorizer.pkl")
    
    if os.path.exists(model_path) and os.path.exists(vectorizer_path):
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
    else:
        print(f"Model files not found at: {model_path} or {vectorizer_path}")
except Exception as e:
    print(f"Error loading models: {e}")

@app.route("/")
def intro():
    # If user is logged in, you might want to show their email or a dashboard link
    return render_template("intro.html", user=session.get("user"))

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")
        try:
            # Tell Supabase where to redirect after email confirmation
            redirect_url = request.host_url.rstrip('/') + "/login"
            res = supabase.auth.sign_up({
                "email": email, 
                "password": password,
                "options": {"email_redirect_to": redirect_url}
            })
            return render_template("signup.html", success="Verification email sent! Please check your inbox.")
        except Exception as e:
            return render_template("signup.html", error=str(e))
    return render_template("signup.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")
        try:
            res = supabase.auth.sign_in_with_password({"email": email, "password": password})
            session["user"] = {
                "id": res.user.id,
                "email": res.user.email
            }
            # Redirect to the main analyze page after successful login
            return redirect(url_for("index"))
        except Exception as e:
            # Provide more specific error feedback
            error_msg = str(e)
            if "Invalid login credentials" in error_msg:
                error_msg = "Invalid email or password. Please try again."
            return render_template("login.html", error=error_msg)
    return render_template("login.html")

@app.route("/logout")
def logout():
    supabase.auth.sign_out()
    session.pop("user", None)
    return redirect(url_for("intro"))

@app.route("/analyze", methods=["GET", "POST"])
def index():
    # Enforce login
    if "user" not in session:
        return redirect(url_for("login"))

    sentiment_result = None
    confidence = None
    sentiment_color = None

    if request.method == "POST":
        review_text = request.form["review"]
        if not review_text.strip():
            sentiment_result = "Please enter some text"
        elif model is None or vectorizer is None:
            sentiment_result = "Model Error: AI engine is not ready."
            sentiment_color = "gray"
        else:
            time.sleep(0.5)
            try:
                transformed_text = vectorizer.transform([review_text])
                prediction = model.predict(transformed_text)[0]
                probabilities = model.predict_proba(transformed_text)[0]
                confidence = max(probabilities) * 100
                
                sentiment_map = {
                    'positive': ('Positive', '#22c55e'),
                    'negative': ('Negative', '#ef4444'),
                    'neutral': ('Neutral', '#6b7280')
                }
                
                if prediction in sentiment_map:
                    sentiment_result, sentiment_color = sentiment_map[prediction]
                else:
                    sentiment_result = "Unknown"
                    sentiment_color = "gray"
            except Exception as e:
                print(f"Prediction error: {e}")
                sentiment_result = "Analysis failed. Please try again."
                sentiment_color = "gray"

    return render_template("index.html", 
                         sentiment=sentiment_result,
                         confidence=confidence,
                         color=sentiment_color,
                         user=session.get("user"))

if __name__ == "__main__":
    app.run(debug=True)
