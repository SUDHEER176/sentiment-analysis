# Sentiment Analysis App

A web-based sentiment analysis application that uses machine learning to classify product reviews as positive, negative, or neutral.

## Features

- **AI-Powered Analysis**: Uses a trained machine learning model to analyze sentiment with high accuracy
- **Real-Time Processing**: Get instant sentiment classification for your text input
- **Confidence Scores**: View confidence percentages for each prediction
- **User-Friendly Interface**: Clean, responsive web interface for easy interaction
- **Color-Coded Results**: Visual indicators (green for positive, red for negative, gray for neutral)

## Project Structure

```
sentiment_app/
├── app.py                 # Flask application server
├── templates/
│   └── index.html        # Main web interface
├── static/
│   └── style.css         # Styling for the web application
├── best_model.pkl        # Trained sentiment classification model
├── tfidf_vectorizer.pkl  # TF-IDF vectorizer for text preprocessing
├── gen ai.ipynb          # Jupyter notebook with model development
└── README.md             # This file
```

## Requirements

- Python 3.x
- Flask
- joblib
- numpy
- scikit-learn (for model compatibility)

## Installation

1. **Clone or download the project**:
   ```bash
   cd sentiment_app
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python -m venv .venv
   .venv\Scripts\Activate.ps1  # On Windows
   source .venv/bin/activate   # On macOS/Linux
   ```

3. **Install dependencies**:
   ```bash
   pip install flask joblib numpy scikit-learn
   ```

## Usage

1. **Start the Flask application**:
   ```bash
   python app.py
   ```

2. **Open your browser** and navigate to:
   ```
   http://localhost:5000
   ```

3. **Enter a product review** in the text area and click "Analyze" to see the sentiment classification

## How It Works

- The application loads pre-trained models (`best_model.pkl` and `tfidf_vectorizer.pkl`)
- User input text is vectorized using the TF-IDF vectorizer
- The vectorized text is fed into the trained machine learning model
- The model predicts the sentiment (positive, negative, or neutral) with a confidence score
- Results are displayed in the web interface with color-coded indicators

## Model Details

- **Algorithm**: Binary/Multi-class Classification (Logistic Regression or similar)
- **Vectorization**: TF-IDF (Term Frequency-Inverse Document Frequency)
- **Classes**: Positive, Negative, Neutral
- **Input**: Product review text
- **Output**: Sentiment classification + Confidence percentage

## Development

The `gen ai.ipynb` Jupyter notebook contains:
- Data exploration and preprocessing
- Model training and evaluation
- Feature engineering with TF-IDF
- Model comparison and selection

## File Descriptions

| File | Purpose |
|------|---------|
| `app.py` | Main Flask application handling routes and sentiment prediction |
| `index.html` | HTML template for the web interface |
| `style.css` | CSS styling for responsive design and animations |
| `best_model.pkl` | Serialized trained ML model |
| `tfidf_vectorizer.pkl` | Serialized TF-IDF vectorizer for text transformation |
| `gen ai.ipynb` | Jupyter notebook documenting model development process |

## License

This project is open source and available under the MIT License.

## Author

Created by SUDHEER176

---

**For questions or improvements, feel free to contribute or create an issue!**
