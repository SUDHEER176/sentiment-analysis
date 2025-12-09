from flask import Flask, render_template, request
import joblib
import os
import numpy as np
import time

# Get the correct paths for Vercel deployment
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
template_dir = os.path.join(base_dir, 'templates')
static_dir = os.path.join(base_dir, 'static')

app = Flask(__name__, template_folder=template_dir, static_folder=static_dir)

# Load models using joblib (more robust than pickle)
model_path = os.path.join(base_dir, "best_model.pkl")
vectorizer_path = os.path.join(base_dir, "tfidf_vectorizer.pkl")

try:
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
except Exception as e:
    print(f"Error loading models: {e}")
    print(f"Looking for model at: {model_path}")
    print(f"Looking for vectorizer at: {vectorizer_path}")
    model = None
    vectorizer = None

@app.route("/", methods=["GET", "POST"])
def index():
    sentiment_result = None
    confidence = None
    sentiment_color = None

    if request.method == "POST":
        review_text = request.form["review"]
        if not review_text.strip():
            sentiment_result = "Please enter some text"
        else:
            # Add slight delay to simulate processing (shows loading spinner)
            time.sleep(0.5)
            
            transformed_text = vectorizer.transform([review_text])
            prediction = model.predict(transformed_text)[0]
            
            # Get prediction probabilities for confidence/accuracy
            probabilities = model.predict_proba(transformed_text)[0]
            confidence = max(probabilities) * 100  # Convert to percentage
            
            # Map predictions to sentiment and color
            sentiment_map = {
                'positive': ('Positive', 'green'),
                'negative': ('Negative', 'red'),
                'neutral': ('Neutral', 'gray')
            }
            
            if prediction in sentiment_map:
                sentiment_result, sentiment_color = sentiment_map[prediction]
            else:
                sentiment_result = "Unknown"
                sentiment_color = "gray"

    return render_template("index.html", 
                         sentiment=sentiment_result,
                         confidence=confidence,
                         color=sentiment_color)
