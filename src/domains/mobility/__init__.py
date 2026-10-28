"""Mobility domain schemas and feature transformers for urban traffic forecasting."""

from src.domains.mobility.schemas import (
    TripInput,
    TripPredictionResponse,
    TripTelemetryPayload,
)
from src.domains.mobility.transformer import (
    AIRPORT_LOCATION_IDS,
    TOP_LOCATIONS_DEFAULT,
    MobilityFeatureTransformer,
)

__all__ = [
    "TripInput",
    "TripPredictionResponse",
    "TripTelemetryPayload",
    "MobilityFeatureTransformer",
    "TOP_LOCATIONS_DEFAULT",
    "AIRPORT_LOCATION_IDS",
]
