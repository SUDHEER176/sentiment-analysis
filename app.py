from flask import Flask, render_template, request
import joblib
import os
import numpy as np
import time

# Get the correct base directory - works for both local and Vercel
if os.path.exists(os.path.join(os.getcwd(), 'best_model.pkl')):
    BASE_DIR = os.getcwd()
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Configure Flask with proper paths
app = Flask(__name__, 
            template_folder=os.path.join(BASE_DIR, 'templates'),
            static_folder=os.path.join(BASE_DIR, 'static'))

# Load models using absolute paths
model = None
vectorizer = None

try:
    model_path = os.path.join(BASE_DIR, "best_model.pkl")
    vectorizer_path = os.path.join(BASE_DIR, "tfidf_vectorizer.pkl")
    
    print(f"Looking for models in: {BASE_DIR}")
    print(f"Model path: {model_path}, exists: {os.path.exists(model_path)}")
    print(f"Vectorizer path: {vectorizer_path}, exists: {os.path.exists(vectorizer_path)}")
    
    if os.path.exists(model_path) and os.path.exists(vectorizer_path):
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        print("Models loaded successfully!")
    else:
        print(f"Model files not found at: {model_path} or {vectorizer_path}")
except Exception as e:
    print(f"Error loading models: {e}")
    import traceback
    traceback.print_exc()

@app.route("/", methods=["GET", "POST"])
def index():

    sentiment_result = None
    confidence = None
    sentiment_color = "#6b7280"  # Default gray

    if request.method == "POST":
        review_text = request.form["review"]
        if not review_text.strip():
            sentiment_result = "Please enter some text"
        elif model is None or vectorizer is None:
            sentiment_result = "Model Error: AI engine is not ready."
            sentiment_color = "#6b7280"
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
                    sentiment_color = "#6b7280"
            except Exception as e:
                print(f"Prediction error: {e}")
                sentiment_result = "Analysis failed. Please try again."
                sentiment_color = "#6b7280"

    return render_template("index.html", 
                         sentiment=sentiment_result,
                         confidence=confidence,
                         color=sentiment_color)

if __name__ == "__main__":
    app.run(debug=True)
