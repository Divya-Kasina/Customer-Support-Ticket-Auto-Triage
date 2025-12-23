# =====================================================
# Support Ticket Classification
# ML Training + Model Persistence + REST API
# =====================================================

import pandas as pd
import re
import time
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report
)

from flask import Flask, request, jsonify


# -----------------------------------------------------
# 1. Load Dataset
# -----------------------------------------------------
DATA_PATH = "IT_Service_Ticket_Classification.csv"
df = pd.read_csv(DATA_PATH)


# -----------------------------------------------------
# 2. Text Preprocessing
# -----------------------------------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


df["text"] = df["Document"].astype(str).apply(clean_text)

X = df["text"]
y = df["Topic_group"]


# -----------------------------------------------------
# 3. Train-Test Split
# -----------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# -----------------------------------------------------
# 4. Build ML Pipeline
# -----------------------------------------------------
model = Pipeline([
    ("tfidf", TfidfVectorizer(
        stop_words="english",
        max_features=8000,
        ngram_range=(1, 2)
    )),
    ("clf", LogisticRegression(
        max_iter=3000,
        class_weight="balanced"
    ))
])


# -----------------------------------------------------
# 5. Train Model
# -----------------------------------------------------
model.fit(X_train, y_train)


# -----------------------------------------------------
# 6. Evaluate Model
# -----------------------------------------------------
start_time = time.time()
y_pred = model.predict(X_test)
end_time = time.time()

latency = (end_time - start_time) / len(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision, recall, f1, _ = precision_recall_fscore_support(
    y_test, y_pred, average="macro"
)

print("\n=== Model Evaluation ===")
print("Accuracy:", accuracy)
print("Precision (macro):", precision)
print("Recall (macro):", recall)
print("F1-score (macro):", f1)
print("Average inference latency (seconds):", latency)

print("\nDetailed Classification Report:\n")
print(classification_report(y_test, y_pred))


# -----------------------------------------------------
# 7. Save Trained Model
# -----------------------------------------------------
MODEL_PATH = "ticket_classifier.pkl"
joblib.dump(model, MODEL_PATH)
print(f"\nModel saved to {MODEL_PATH}")


# =====================================================
# 8. REST API (Flask)
# =====================================================
app = Flask(__name__)
loaded_model = joblib.load(MODEL_PATH)


# ---------- Home ----------
@app.route("/", methods=["GET"])
def home():
    return "Support Ticket Classification API is running"


# ---------- Browser Test Page ----------
@app.route("/ui", methods=["GET"])
def ui():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Support Ticket Classifier</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                background-color: #f4f6f8;
                padding: 40px;
            }
            .container {
                max-width: 600px;
                background: white;
                padding: 30px;
                border-radius: 8px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                margin: auto;
            }
            h2 {
                text-align: center;
            }
            label {
                font-weight: bold;
            }
            input, textarea {
                width: 100%;
                padding: 10px;
                margin-top: 5px;
                margin-bottom: 15px;
                border-radius: 4px;
                border: 1px solid #ccc;
            }
            button {
                width: 100%;
                padding: 12px;
                background-color: #007bff;
                color: white;
                border: none;
                border-radius: 4px;
                font-size: 16px;
                cursor: pointer;
            }
            button:hover {
                background-color: #0056b3;
            }
            .result {
                margin-top: 20px;
                padding: 15px;
                background: #e9f5ff;
                border-radius: 5px;
                display: none;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h2>Support Ticket Classification</h2>

            <label>Subject</label>
            <input type="text" id="subject" placeholder="Enter ticket subject">

            <label>Description</label>
            <textarea id="description" rows="5" placeholder="Enter ticket description"></textarea>

            <button onclick="predict()">Classify Ticket</button>

            <div class="result" id="result"></div>
        </div>

        <script>
            function predict() {
                const subject = document.getElementById("subject").value;
                const description = document.getElementById("description").value;

                fetch("/predict", {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json"
                    },
                    body: JSON.stringify({
                        subject: subject,
                        description: description
                    })
                })
                .then(response => response.json())
                .then(data => {
                    const resultDiv = document.getElementById("result");
                    resultDiv.style.display = "block";
                    resultDiv.innerHTML = `
                        <b>Predicted Category:</b> ${data.predicted_category}<br>
                        <b>Latency:</b> ${data.latency_seconds.toFixed(4)} seconds
                    `;
                })
                .catch(error => {
                    alert("Error connecting to API");
                    console.error(error);
                });
            }
        </script>
    </body>
    </html>
    """

# ---------- Browser Prediction ----------
@app.route("/predict_form", methods=["POST"])
def predict_form():
    subject = request.form.get("subject", "")
    description = request.form.get("description", "")

    text = clean_text(subject + " " + description)
    prediction = loaded_model.predict([text])[0]

    return f"""
    <h3>Predicted Category: {prediction}</h3>
    <a href="/test">Go Back</a>
    """


# ---------- JSON API Prediction ----------
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    if not data:
        return jsonify({"error": "No JSON payload provided"}), 400

    subject = data.get("subject", "")
    description = data.get("description", "")

    if not subject and not description:
        return jsonify({"error": "Subject or description required"}), 400

    text = clean_text(subject + " " + description)

    start = time.time()
    prediction = loaded_model.predict([text])[0]
    inference_latency = time.time() - start

    return jsonify({
        "predicted_category": prediction,
        "latency_seconds": inference_latency
    })


if __name__ == "__main__":
    print("\n===================================")
    print("🚀 Frontend UI available at:")
    print("👉 http://127.0.0.1:5000/ui")
    print("===================================\n")

    app.run(debug=True)
