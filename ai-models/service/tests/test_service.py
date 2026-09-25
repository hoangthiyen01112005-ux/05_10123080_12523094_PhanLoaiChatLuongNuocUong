from fastapi.testclient import TestClient

from main import app


client = TestClient(app)


VALID_PAYLOAD = {
    "ph": 7.0,
    "Hardness": 200.0,
    "Solids": 20000.0,
    "Chloramines": 7.0,
    "Sulfate": 330.0,
    "Conductivity": 420.0,
    "Organic_carbon": 14.0,
    "Trihalomethanes": 65.0,
    "Turbidity": 4.0,
}


def test_health():
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "ok"
    assert data["service"] == "ai-service"
    assert data["model_loaded"] is False
    assert "request_id" in data


def test_request_id_from_header():
    response = client.get(
        "/health",
        headers={
            "X-Request-ID": "test-request-123",
        },
    )

    assert response.status_code == 200
    assert response.json()["request_id"] == "test-request-123"
    assert response.headers["X-Request-ID"] == "test-request-123"


def test_validate_valid_input():
    response = client.post(
        "/validate",
        json=VALID_PAYLOAD,
    )

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "valid"
    assert "request_id" in data
    assert data["data"]["ph"] == 7.0
    assert data["data"]["Turbidity"] == 4.0


def test_validate_invalid_range():
    payload = VALID_PAYLOAD.copy()
    payload["ph"] = 20.0

    response = client.post(
        "/validate",
        json=payload,
    )

    assert response.status_code == 422

    data = response.json()

    assert data["status"] == "error"
    assert data["message"] == "Invalid input data"
    assert "request_id" in data
    assert len(data["errors"]) > 0


def test_validate_missing_field():
    payload = VALID_PAYLOAD.copy()
    payload.pop("Turbidity")

    response = client.post(
        "/validate",
        json=payload,
    )

    assert response.status_code == 422

    data = response.json()

    assert data["status"] == "error"
    assert data["message"] == "Invalid input data"
    assert "request_id" in data