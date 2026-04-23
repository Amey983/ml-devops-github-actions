import os
import pickle
from pathlib import Path

from flask import Flask, jsonify, request


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "trained_model.pkl"


def ensure_model_exists():
    if MODEL_PATH.exists():
        return

    from model.train_model import train_and_save_model

    train_and_save_model()


def load_model():
    ensure_model_exists()
    with MODEL_PATH.open("rb") as model_file:
        return pickle.load(model_file)


model = load_model()
app = Flask(__name__)


@app.get("/")
def health_check():
    return jsonify(
        {
            "status": "ok",
            "message": "ML prediction API is running.",
            "model": "diabetes_regression",
        }
    )


@app.post("/predict")
def predict():
    payload = request.get_json(silent=True)
    if not payload:
        return jsonify({"error": "Request body must be valid JSON."}), 400

    features = payload.get("features")
    if not isinstance(features, list) or len(features) != 10:
        return jsonify(
            {
                "error": "Field 'features' must be a list containing exactly 10 numeric values."
            }
        ), 400

    try:
        values = [[float(value) for value in features]]
    except (TypeError, ValueError):
        return jsonify({"error": "All feature values must be numeric."}), 400

    prediction = model.predict(values)[0]
    return jsonify({"prediction": round(float(prediction), 4)})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
