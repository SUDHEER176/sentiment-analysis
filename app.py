from flask import Flask, render_template, request, redirect, url_for, session
import joblib
import os
import numpy as np
import time
from supabase import create_client, Client
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "default-secret-key")

# Supabase configuration
url: str = os.getenv("SUPABASE_URL")
key: str = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(url, key)

# Load models using joblib
try:
    model = joblib.load("best_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
except Exception as e:
    print(f"Error loading models: {e}")
    model = None
    vectorizer = None

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
        else:
            time.sleep(0.5)
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

    return render_template("index.html", 
                         sentiment=sentiment_result,
                         confidence=confidence,
                         color=sentiment_color,
                         user=session.get("user"))

if __name__ == "__main__":
    app.run(debug=True)
