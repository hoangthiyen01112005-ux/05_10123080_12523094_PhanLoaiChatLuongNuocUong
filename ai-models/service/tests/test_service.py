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
    assert data["model_loaded"] is True
    assert set(data["available_models"]) == {"lr", "svm", "rf", "knn"}
    assert data["model_count"] == 4
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

def test_model_info_unavailable():
    response = client.get("/model-info")

    assert response.status_code == 503

    data = response.json()

    assert data["detail"]["message"] == "Model metadata is not available"
    assert "request_id" in data["detail"]

@app.get("/test-system-error")
def trigger_system_error():
    raise RuntimeError("Test system error")


def test_system_error_handler():
    test_client = TestClient(
        app,
        raise_server_exceptions=False,
    )

    response = test_client.get(
        "/test-system-error",
        headers={
            "X-Request-ID": "test-error-500",
        },
    )

    assert response.status_code == 500

    data = response.json()

    assert data["status"] == "error"
    assert data["message"] == "Internal server error"
    assert data["request_id"] == "test-error-500"

    assert response.headers["X-Request-ID"] == "test-error-500"

def test_predict_all_models():
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
    }

    for model_type in ["lr", "svm", "rf", "knn"]:
        response = client.post(
            "/predict",
            json={
                **payload,
                "model_type": model_type,
            },
            headers={
                "X-Request-ID": f"predict-{model_type}-123",
            },
        )

        assert response.status_code == 200

        data = response.json()

        assert data["prediction"] in [0, 1]
        assert data["model_type"] == model_type
        assert data["request_id"] == f"predict-{model_type}-123"

        assert "label" in data
        assert "probability" in data
        assert "model_used" in data

        if data["probability"] is not None:
            assert 0.0 <= data["probability"] <= 1.0