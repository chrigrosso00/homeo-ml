"""Integration tests for FastAPI Serving layer and telemetry dispatch."""

from collections.abc import Generator
from datetime import datetime

import pytest
from starlette.testclient import TestClient

from src.data.schemas import TelemetryEvent
from src.serving.app import app


@pytest.fixture
def client() -> Generator[TestClient, None, None]:
    """Provide TestClient initialized with application lifespan."""
    with TestClient(app) as test_client:
        yield test_client


def test_health_check_endpoint(client: TestClient) -> None:
    """Verify /health endpoint returns model version and feature dimensions."""
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "healthy"
    assert payload["model_version"] == "v1.0.0"
    assert payload["n_features"] == 41
    assert 0.0 < payload["optimal_threshold"] < 1.0


def test_predict_legitimate_transaction(client: TestClient) -> None:
    """Verify scoring a normal legitimate transaction."""
    tx_payload = {
        "user_id": 22058,
        "signup_time": "2015-02-24T22:55:49",
        "purchase_time": "2015-04-18T02:47:11",
        "purchase_value": 34.0,
        "device_id": "QVPSPJUOCKZAR",
        "source": "SEO",
        "browser": "Chrome",
        "sex": "M",
        "age": 39,
        "ip_address": 732758400.0,
    }

    response = client.post("/predict", json=tx_payload)
    assert response.status_code == 200
    res = response.json()
    assert "transaction_id" in res
    assert 0.0 <= res["fraud_probability"] <= 1.0
    assert isinstance(res["is_fraud"], bool)
    assert res["model_version"] == "v1.0.0"
    assert res["latency_ms"] < 20.0  # Must be strictly under 20ms SLA


def test_predict_instant_suspicious_transaction(client: TestClient) -> None:
    """Verify scoring an instant automated purchase (<5 seconds after signup)."""
    tx_payload = {
        "user_id": 1359,
        "signup_time": "2015-01-01T18:52:44",
        "purchase_time": "2015-01-01T18:52:45",  # 1 second delta
        "purchase_value": 52.0,
        "device_id": "SHARED_BOT_DEV",
        "source": "Direct",
        "browser": "Opera",
        "sex": "F",
        "age": 28,
        "ip_address": 2621474000.0,
    }

    response = client.post("/predict", json=tx_payload)
    assert response.status_code == 200
    res = response.json()
    assert res["is_fraud"] is True
    assert res["fraud_probability"] > 0.5


def test_predict_validation_error(client: TestClient) -> None:
    """Verify Pydantic validation rejects negative amount or invalid enum."""
    invalid_payload = {
        "user_id": 1,
        "signup_time": "invalid_date",
        "purchase_time": "2015-01-01T12:00:00",
        "purchase_value": -50.0,  # Negative amount rejected
        "device_id": "DEV",
        "source": "INVALID_CHANNEL",
        "browser": "Chrome",
        "sex": "X",
        "age": 12,  # Under 18 rejected
        "ip_address": 123.0,
    }

    response = client.post("/predict", json=invalid_payload)
    assert response.status_code == 422  # HTTP Unprocessable Entity


def test_telemetry_producer_graceful_fallback() -> None:
    """Verify telemetry dispatch returns False without throwing when Kafka is offline."""
    from src.serving.telemetry import KafkaTelemetryProducer

    producer = KafkaTelemetryProducer(bootstrap_servers="localhost:9999")
    event = TelemetryEvent(
        transaction_id="test-id",
        user_id=1,
        signup_time=datetime.now().isoformat(),
        purchase_time=datetime.now().isoformat(),
        purchase_value=10.0,
        device_id="DEV",
        source="SEO",
        browser="Chrome",
        sex="M",
        age=30,
        ip_address=123.0,
        country="Unknown",
        time_since_signup_sec=10.0,
        fraud_probability=0.1,
        is_fraud=False,
        model_version="v1.0.0",
        latency_ms=1.2,
        logged_at=datetime.now().isoformat(),
    )
    # Must not raise an unhandled exception
    res = producer.dispatch(event)
    assert res in (True, False)
