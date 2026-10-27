"""Unit tests for data schemas and feature transformer."""

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.schemas import TransactionInput
from src.data.transformer import FraudFeatureTransformer, IpCountryLookup


@pytest.fixture
def sample_raw_df() -> pd.DataFrame:
    """Provide a small realistic DataFrame for testing."""
    return pd.DataFrame(
        {
            "user_id": [1, 2],
            "signup_time": ["2015-01-01 12:00:00", "2015-02-01 10:00:00"],
            "purchase_time": ["2015-01-01 12:00:02", "2015-02-01 15:00:00"],
            "purchase_value": [45.0, 120.0],
            "device_id": ["DEV_A", "DEV_B"],
            "source": ["SEO", "Ads"],
            "browser": ["Chrome", "Safari"],
            "sex": ["M", "F"],
            "age": [30, 45],
            "ip_address": [732758400.0, 350311400.0],
            "class": [1, 0],
        }
    )


def test_ip_country_lookup() -> None:
    """Verify binary search returns expected countries and handles unseen IPs."""
    lookup = IpCountryLookup()
    res = lookup.lookup([732758400.0, 350311400.0, 0.0])
    assert len(res) == 3
    assert res[0] == "Japan"
    assert res[1] == "United States"
    assert res[2] == "Unknown"


def test_transformer_fit_transform(sample_raw_df: pd.DataFrame) -> None:
    """Verify fitting transformer and output feature matrix."""
    transformer = FraudFeatureTransformer(top_countries_count=5)
    transformer.fit(sample_raw_df)

    feat_matrix = transformer.transform_df(sample_raw_df)
    assert len(feat_matrix) == 2
    assert "purchase_value" in feat_matrix.columns
    assert "time_since_signup_sec" in feat_matrix.columns
    assert "is_instant_purchase" in feat_matrix.columns
    # Row 0 had 2 seconds delta -> instant purchase flag must be 1
    assert feat_matrix.iloc[0]["is_instant_purchase"] == 1
    # Row 1 had 5 hours delta -> instant purchase flag must be 0
    assert feat_matrix.iloc[1]["is_instant_purchase"] == 0


def test_transformer_single_input(sample_raw_df: pd.DataFrame, tmp_path: Path) -> None:
    """Verify equivalence between batch transformation and single-item online inference."""
    transformer = FraudFeatureTransformer(top_countries_count=5)
    transformer.fit(sample_raw_df)
    batch_features = transformer.transform_df(sample_raw_df.iloc[[0]])

    single_input = TransactionInput(
        user_id=1,
        signup_time=datetime(2015, 1, 1, 12, 0, 0),
        purchase_time=datetime(2015, 1, 1, 12, 0, 2),
        purchase_value=45.0,
        device_id="DEV_A",
        source="SEO",
        browser="Chrome",
        sex="M",
        age=30,
        ip_address=732758400.0,
    )
    single_features = transformer.transform_single(single_input)

    assert single_features.shape == (1, len(transformer.feature_names_))
    np.testing.assert_allclose(batch_features.values, single_features, rtol=1e-5)

    # Test serialization
    saved_path = tmp_path / "transformer.joblib"
    transformer.save(saved_path)
    loaded = FraudFeatureTransformer.load(saved_path)
    loaded_features = loaded.transform_single(single_input)
    np.testing.assert_allclose(single_features, loaded_features)
