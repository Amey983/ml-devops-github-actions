import pickle
from pathlib import Path

import pytest


MODEL_PATH = Path(__file__).resolve().parents[1] / "model" / "trained_model.pkl"


@pytest.fixture
def client():
    from app import app, ensure_model_exists

    ensure_model_exists()
    app.config["TESTING"] = True
    with app.test_client() as test_client:
        yield test_client


def test_model_file_exists():
    from app import ensure_model_exists

    ensure_model_exists()
    assert MODEL_PATH.exists()


def test_model_can_predict():
    with MODEL_PATH.open("rb") as model_file:
        model = pickle.load(model_file)

    prediction = model.predict([[0.05, -0.04, 0.02, 0.01, -0.03, -0.02, 0.04, -0.01, 0.03, 0.02]])
    assert isinstance(float(prediction[0]), float)


def test_health_endpoint(client):
    response = client.get("/")
    body = response.get_json()

    assert response.status_code == 200
    assert body["status"] == "ok"


def test_predict_endpoint(client):
    payload = {
        "features": [0.05, -0.04, 0.02, 0.01, -0.03, -0.02, 0.04, -0.01, 0.03, 0.02]
    }
    response = client.post("/predict", json=payload)
    body = response.get_json()

    assert response.status_code == 200
    assert "prediction" in body


def test_predict_endpoint_rejects_invalid_payload(client):
    response = client.post("/predict", json={"features": [1, 2]})

    assert response.status_code == 400
