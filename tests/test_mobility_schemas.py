"""Unit tests for mobility domain Pydantic schemas."""

import pytest
from pydantic import ValidationError

from src.domains.mobility.schemas import (
    TripInput,
    TripPredictionResponse,
    TripTelemetryPayload,
)


def test_valid_trip_input() -> None:
    """Test valid trip input parsing and default values."""
    tx = TripInput(
        pickup_datetime="2020-03-25T14:30:00",
        pickup_location_id=161,
        dropoff_location_id=236,
        trip_distance=2.5,
        passenger_count=2,
    )
    assert tx.pickup_location_id == 161
    assert tx.trip_distance == 2.5
    assert tx.passenger_count == 2


def test_invalid_trip_datetime() -> None:
    """Test rejection of non-ISO formatted pickup timestamps."""
    with pytest.raises(ValidationError):
        TripInput(
            pickup_datetime="not-a-datetime",
            pickup_location_id=10,
            dropoff_location_id=20,
            trip_distance=3.0,
        )


def test_invalid_location_bounds() -> None:
    """Test rejection of location IDs outside the valid 1..265 range."""
    with pytest.raises(ValidationError):
        TripInput(
            pickup_datetime="2020-03-25T14:30:00",
            pickup_location_id=0,  # Below minimum (1)
            dropoff_location_id=20,
            trip_distance=3.0,
        )

    with pytest.raises(ValidationError):
        TripInput(
            pickup_datetime="2020-03-25T14:30:00",
            pickup_location_id=10,
            dropoff_location_id=999,  # Above maximum (265)
            trip_distance=3.0,
        )


def test_invalid_trip_distance() -> None:
    """Test rejection of non-positive trip distances."""
    with pytest.raises(ValidationError):
        TripInput(
            pickup_datetime="2020-03-25T14:30:00",
            pickup_location_id=10,
            dropoff_location_id=20,
            trip_distance=0.0,
        )


def test_trip_telemetry_payload() -> None:
    """Test valid composition of telemetry payload."""
    req = TripInput(
        pickup_datetime="2020-03-25T14:30:00",
        pickup_location_id=10,
        dropoff_location_id=20,
        trip_distance=3.0,
    )
    res = TripPredictionResponse(
        is_congested=True,
        congestion_probability=0.88,
        model_version="v1.0.0",
    )
    telemetry = TripTelemetryPayload(
        trip_input=req,
        prediction=res,
        latency_ms=0.45,
    )
    assert telemetry.latency_ms == 0.45
    assert telemetry.prediction.is_congested is True
    assert telemetry.trip_input.passenger_count == 1
