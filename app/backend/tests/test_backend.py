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