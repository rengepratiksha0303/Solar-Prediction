import os
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

app = Flask(__name__)

MODEL_FILE = "model.pkl"
DATA_FILE = "wind_solar_energy_dataset.csv"

# ------------------------------
# Train & Save Model (Run Once)
# ------------------------------
def train_and_save_model():
    df = pd.read_csv(DATA_FILE)
    df = df.dropna()

    target = "Solar"  # change to 'Wind' for wind prediction
    features = df.columns.drop(target)

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    with open(MODEL_FILE, "wb") as f:
        pickle.dump(model, f)

    print("Model trained and saved to", MODEL_FILE)

if not os.path.exists(MODEL_FILE):
    train_and_save_model()

# ------------------------------
# Load Model
# ------------------------------
with open(MODEL_FILE, "rb") as f:
    model = pickle.load(f)

# ------------------------------
# Home Page
# ------------------------------
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

# ------------------------------
# Predict API (JSON)
# ------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)

    # Convert to 2D array
    features = np.array(data["features"]).reshape(1, -1)
    pred = model.predict(features)

    return jsonify({
        "prediction": float(pred[0])
    })

# ------------------------------
# Run App
# ------------------------------
if __name__ == "__main__":
    app.run(debug=True)
