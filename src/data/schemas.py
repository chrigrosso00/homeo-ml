"""Data models and validation schemas for transactions and telemetry.

Defines Pydantic schemas for API inputs, model predictions, and asynchronous
event streaming telemetry payloads.
"""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class TransactionInput(BaseModel):
    """Schema for incoming transaction scoring requests."""

    user_id: int = Field(
        ..., description="Unique customer identifier", json_schema_extra={"example": 22058}
    )
    signup_time: datetime = Field(
        ...,
        description="Timestamp when account was created",
        json_schema_extra={"example": "2015-02-24T22:55:49"},
    )
    purchase_time: datetime = Field(
        ...,
        description="Timestamp of the transaction",
        json_schema_extra={"example": "2015-04-18T02:47:11"},
    )
    purchase_value: float = Field(
        ..., ge=0.0, description="Transaction monetary amount", json_schema_extra={"example": 34.0}
    )
    device_id: str = Field(
        ...,
        description="Hardware device fingerprint",
        json_schema_extra={"example": "QVPSPJUOCKZAR"},
    )
    source: Literal["SEO", "Ads", "Direct"] = Field(
        ...,
        description="Customer acquisition marketing channel",
        json_schema_extra={"example": "SEO"},
    )
    browser: Literal["Chrome", "Safari", "Firefox", "Opera", "IE"] = Field(
        ..., description="Client browser identifier", json_schema_extra={"example": "Chrome"}
    )
    sex: Literal["M", "F"] = Field(
        ..., description="Customer gender indicator", json_schema_extra={"example": "M"}
    )
    age: int = Field(
        ..., ge=18, le=120, description="Customer age in years", json_schema_extra={"example": 39}
    )
    ip_address: float = Field(
        ...,
        description="Numeric representation of client IP",
        json_schema_extra={"example": 732758368.0},
    )


class PredictionResponse(BaseModel):
    """Schema for model inference output returned to client."""

    transaction_id: str = Field(..., description="Unique tracking identifier for this request")
    fraud_probability: float = Field(
        ..., ge=0.0, le=1.0, description="Estimated probability of fraud"
    )
    is_fraud: bool = Field(..., description="Binary decision based on operational threshold")
    model_version: str = Field(..., description="Identifier of the serving model artifact")
    latency_ms: float = Field(
        ..., description="End-to-end inference execution time in milliseconds"
    )


class TelemetryEvent(BaseModel):
    """Schema for events dispatched to Redpanda streaming buffer."""

    transaction_id: str
    user_id: int
    signup_time: str
    purchase_time: str
    purchase_value: float
    device_id: str
    source: str
    browser: str
    sex: str
    age: int
    ip_address: float
    country: str
    time_since_signup_sec: float
    fraud_probability: float
    is_fraud: bool
    model_version: str
    latency_ms: float
    logged_at: str
