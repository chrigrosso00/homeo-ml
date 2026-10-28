"""Unit tests for mobility domain feature transformer."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.domains.mobility.schemas import TripInput
from src.domains.mobility.transformer import MobilityFeatureTransformer


@pytest.fixture
def sample_taxi_df() -> pd.DataFrame:
    """Fixture providing a mock raw DataFrame of taxi trips."""
    return pd.DataFrame(
        {
            "tpep_pickup_datetime": [
                "2020-03-25 14:30:00",
                "2020-03-28 09:15:00",  # Saturday (weekend)
                "2020-03-20 23:45:00",
            ],
            "PULocationID": [161, 132, 236],  # 132 is JFK airport
            "DOLocationID": [236, 138, 161],  # 138 is LGA airport
            "trip_distance": [2.5, 15.0, 1.2],
            "passenger_count": [1, 2, 4],
        }
    )


def test_fit_and_feature_names(sample_taxi_df: pd.DataFrame) -> None:
    """Verify that fit extracts top locations and generates deterministic feature names."""
    transformer = MobilityFeatureTransformer(top_locations_count=2)
    transformer.fit(sample_taxi_df)

    assert len(transformer.top_pickup_locations_) == 2
    assert len(transformer.top_dropoff_locations_) == 2
    # 7 base features + 2 PU one-hots + 2 DO one-hots = 11 features
    assert len(transformer.feature_names_) == 11
    assert "trip_distance" in transformer.feature_names_
    assert "is_weekend" in transformer.feature_names_
    assert "is_airport_pickup" in transformer.feature_names_


def test_train_serve_parity(sample_taxi_df: pd.DataFrame) -> None:
    """Critical verification: transform_df and transform_single must yield identical arrays."""
    transformer = MobilityFeatureTransformer(top_locations_count=3)
    transformer.fit(sample_taxi_df)

    # 1. Transform via batch DataFrame
    df_transformed = transformer.transform_df(sample_taxi_df)
    batch_first_row = df_transformed.iloc[0].values.astype(np.float32)

    # 2. Transform the exact same trip via single Pydantic TripInput
    single_trip = TripInput(
        pickup_datetime="2020-03-25T14:30:00",
        pickup_location_id=161,
        dropoff_location_id=236,
        trip_distance=2.5,
        passenger_count=1,
    )
    single_transformed = transformer.transform_single(single_trip)[0]

    # Must be numerically identical (zero training/serving skew)
    assert np.allclose(batch_first_row, single_transformed, atol=1e-6)


def test_airport_flags(sample_taxi_df: pd.DataFrame) -> None:
    """Verify airport pickup/dropoff logic (JFK=132, LGA=138, EWR=1)."""
    transformer = MobilityFeatureTransformer()
    transformer.fit(sample_taxi_df)

    # Trip 1: neither is airport
    # Trip 2: pickup is JFK (132), dropoff is LGA (138)
    df_transformed = transformer.transform_df(sample_taxi_df)
    assert df_transformed.loc[0, "is_airport_pickup"] == 0
    assert df_transformed.loc[0, "is_airport_dropoff"] == 0
    assert df_transformed.loc[1, "is_airport_pickup"] == 1
    assert df_transformed.loc[1, "is_airport_dropoff"] == 1


def test_serialization(tmp_path: Path, sample_taxi_df: pd.DataFrame) -> None:
    """Verify that the transformer serializes and deserializes accurately."""
    transformer = MobilityFeatureTransformer(top_locations_count=2)
    transformer.fit(sample_taxi_df)

    save_path = tmp_path / "mobility_transformer.joblib"
    transformer.save(save_path)

    loaded = MobilityFeatureTransformer.load(save_path)
    assert loaded.top_pickup_locations_ == transformer.top_pickup_locations_
    assert loaded.feature_names_ == transformer.feature_names_
