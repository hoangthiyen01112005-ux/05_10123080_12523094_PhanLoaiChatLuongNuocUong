import sys
from pathlib import Path

from fastapi.testclient import TestClient
from pymongo.errors import PyMongoError


# Cho phép test import các file trong app/backend/src
BACKEND_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(BACKEND_SRC))


import main


client = TestClient(
    main.app,
    raise_server_exceptions=False,
)


def test_health(monkeypatch):
    monkeypatch.setattr(
        main,
        "check_database_connection",
        lambda: True,
    )

    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "ok"
    assert data["service"] == "backend"
    assert data["database_connected"] is True
    assert "request_id" in data


def test_request_id_from_header(monkeypatch):
    monkeypatch.setattr(
        main,
        "check_database_connection",
        lambda: True,
    )

    response = client.get(
        "/health",
        headers={
            "X-Request-ID": "backend-test-123",
        },
    )

    assert response.status_code == 200
    assert response.json()["request_id"] == "backend-test-123"
    assert response.headers["X-Request-ID"] == "backend-test-123"


def test_history(monkeypatch):
    fake_history = [
        {
            "_id": "123",
            "request_id": "request-001",
            "features": {
                "ph": 7.0,
            },
            "prediction": 1,
            "probability": 0.85,
            "model_version": "1.0.0",
            "created_at": "2026-09-26T10:00:00+00:00",
        }
    ]

    monkeypatch.setattr(
        main,
        "get_prediction_history",
        lambda limit=20: fake_history,
    )

    response = client.get("/api/history?limit=20")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "ok"
    assert data["count"] == 1
    assert len(data["data"]) == 1
    assert data["data"][0]["request_id"] == "request-001"
    assert data["data"][0]["prediction"] == 1
    assert "request_id" in data


def test_database_error(monkeypatch):
    def raise_database_error(limit=20):
        raise PyMongoError("Database unavailable")

    monkeypatch.setattr(
        main,
        "get_prediction_history",
        raise_database_error,
    )

    response = client.get(
        "/api/history",
        headers={
            "X-Request-ID": "db-error-test-123",
        },
    )

    assert response.status_code == 503

    data = response.json()

    assert data["status"] == "error"
    assert data["message"] == "Database service unavailable"
    assert data["request_id"] == "db-error-test-123"

    assert response.headers["X-Request-ID"] == "db-error-test-123"

@main.app.get("/test-system-error")
def trigger_system_error():
    raise RuntimeError("Test system error")


def test_system_error_handler():
    response = client.get(
        "/test-system-error",
        headers={
            "X-Request-ID": "system-error-test-123",
        },
    )

    assert response.status_code == 500

    data = response.json()

    assert data["status"] == "error"
    assert data["message"] == "Internal server error"
    assert data["request_id"] == "system-error-test-123"

    assert response.headers["X-Request-ID"] == "system-error-test-123"

def test_cors_preflight():
    response = client.options(
        "/health",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "GET",
        },
    )

    assert response.status_code == 200
    assert "access-control-allow-origin" in response.headers
    assert "access-control-allow-methods" in response.headers

def test_predict_and_save_history(monkeypatch):
    captured_ai_request = {}
    captured_history = {}

    class FakeResponse:
        status_code = 200

        def json(self):
            return {
                "prediction": 1,
                "label": "Uống được (Potable)",
                "probability": 0.85,
                "model_used": "Random Forest",
            }

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

        async def post(self, url, json=None, headers=None):
            captured_ai_request["url"] = url
            captured_ai_request["json"] = json
            captured_ai_request["headers"] = headers

            return FakeResponse()

    def fake_save_prediction_history(
        request_id,
        features,
        prediction,
        probability=None,
        model_version=None,
    ):
        captured_history["request_id"] = request_id
        captured_history["features"] = features
        captured_history["prediction"] = prediction
        captured_history["probability"] = probability
        captured_history["model_version"] = model_version

        return "fake-history-id"

    monkeypatch.setattr(
        main.httpx,
        "AsyncClient",
        FakeAsyncClient,
    )

    monkeypatch.setattr(
        main,
        "save_prediction_history",
        fake_save_prediction_history,
    )

    payload = {
        "ph": 7.0,
        "Hardness": 204.89,
        "Solids": 20791.32,
        "Chloramines": 7.30,
        "Sulfate": 368.51,
        "Conductivity": 564.30,
        "Organic_carbon": 10.37,
        "Trihalomethanes": 86.99,
        "Turbidity": 2.96,
        "model_type": "rf",
    }

    response = client.post(
        "/api/predict",
        json=payload,
        headers={
            "X-Request-ID": "predict-test-123",
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["prediction"] == 1
    assert data["probability"] == 0.85
    assert data["model_used"] == "Random Forest"
    assert data["request_id"] == "predict-test-123"

    assert captured_ai_request["url"] == (
        f"{main.AI_SERVICE_URL}/predict"
    )

    assert (
        captured_ai_request["headers"]["X-Request-ID"]
        == "predict-test-123"
    )

    assert captured_ai_request["json"]["model_type"] == "rf"

    assert captured_history["request_id"] == "predict-test-123"
    assert captured_history["prediction"] == 1
    assert captured_history["probability"] == 0.85
    assert captured_history["model_version"] is None

    assert "model_type" not in captured_history["features"]
    assert captured_history["features"]["ph"] == 7.0

def test_model_info_proxy(monkeypatch):
    captured_request = {}

    class FakeResponse:
        status_code = 200

        def json(self):
            return {
                "status": "ok",
                "request_id": "model-info-test-123",
                "model": {
                    "name": "test-model",
                    "version": "1.0.0",
                },
            }

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

        async def get(self, url, headers=None):
            captured_request["url"] = url
            captured_request["headers"] = headers

            return FakeResponse()

    monkeypatch.setattr(
        main.httpx,
        "AsyncClient",
        FakeAsyncClient,
    )

    response = client.get(
        "/api/model-info",
        headers={
            "X-Request-ID": "model-info-test-123",
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "ok"
    assert data["request_id"] == "model-info-test-123"
    assert data["model"]["name"] == "test-model"
    assert data["model"]["version"] == "1.0.0"

    assert captured_request["url"] == (
        f"{main.AI_SERVICE_URL}/model-info"
    )

    assert (
        captured_request["headers"]["X-Request-ID"]
        == "model-info-test-123"
    )

    assert response.headers["X-Request-ID"] == "model-info-test-123"