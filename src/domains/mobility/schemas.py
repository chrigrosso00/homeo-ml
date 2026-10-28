"""Data models and validation schemas for urban mobility trips and telemetry.

Defines Pydantic schemas for API inputs, model predictions, and asynchronous
event streaming telemetry payloads for traffic congestion / SLA forecasting.
"""

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class TripInput(BaseModel):
    """Validation schema for incoming online ride inference requests.

    Represents features strictly available at pickup request time (no data leakage).
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "pickup_datetime": "2020-03-25T14:30:00",
                "pickup_location_id": 161,
                "dropoff_location_id": 236,
                "trip_distance": 2.5,
                "passenger_count": 1,
            }
        }
    )

    pickup_datetime: str = Field(
        ...,
        description="ISO 8601 pickup timestamp string (e.g. '2020-03-25T14:30:00').",
    )
    pickup_location_id: int = Field(
        ...,
        ge=1,
        le=265,
        description="NYC TLC Taxi Zone Location ID for pickup (1 to 265).",
    )
    dropoff_location_id: int = Field(
        ...,
        ge=1,
        le=265,
        description="NYC TLC Taxi Zone Location ID for dropoff (1 to 265).",
    )
    trip_distance: float = Field(
        ...,
        gt=0.0,
        le=100.0,
        description="Estimated travel distance in miles.",
    )
    passenger_count: int = Field(
        default=1,
        ge=1,
        le=6,
        description="Number of passengers in the vehicle (1 to 6).",
    )

    @field_validator("pickup_datetime")
    @classmethod
    def validate_iso_format(cls, v: str) -> str:
        """Validate that the pickup timestamp is a valid parsable datetime."""
        try:
            datetime.fromisoformat(v)
        except ValueError as err:
            raise ValueError(
                f"Invalid pickup_datetime '{v}'. Expected ISO format (YYYY-MM-DDTHH:MM:SS)."
            ) from err
        return v


class TripPredictionResponse(BaseModel):
    """Schema for online inference API responses for urban mobility."""

    is_congested: bool = Field(
        ...,
        description="Binary prediction: True if congested (< 12 mph), False otherwise.",
    )
    congestion_probability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Predicted probability of experiencing severe traffic / congestion.",
    )
    model_version: str = Field(
        ...,
        description="Active serving model release version string (e.g. 'v1.0.0').",
    )


class TripTelemetryPayload(BaseModel):
    """Schema for streaming events published asynchronously to Redpanda / Kafka."""

    timestamp: str = Field(
        default_factory=lambda: datetime.now(UTC).isoformat(),
        description="UTC timestamp of the telemetry event creation.",
    )
    trip_input: TripInput = Field(
        ...,
        description="Original client request input payload.",
    )
    prediction: TripPredictionResponse = Field(
        ...,
        description="Model prediction returned to the client.",
    )
    latency_ms: float = Field(
        ...,
        ge=0.0,
        description="End-to-end serving execution latency in milliseconds.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional operational metadata (e.g. deployment stage, host).",
    )
