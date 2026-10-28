"""Unit tests for urban mobility training pipeline and ONNX serialization."""

from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd
import pytest

from src.models.train_mobility import (
    clean_and_label_taxi_data,
    find_optimal_threshold,
    train_mobility_pipeline,
)


@pytest.fixture
def mock_taxi_dir(tmp_path: Path) -> Path:
    """Fixture creating temporary dummy parquet trip records for fast testing."""
    records = []
    # Create 500 synthetic trip records with both congested (<12 mph) and fluid (>=12 mph) trips
    for i in range(500):
        # Even i: 1.0 mile in 15 min = 4 mph (congested)
        # Odd i: 8.0 miles in 15 min = 32 mph (fluid)
        dist = 1.0 if (i % 2 == 0) else 8.0
        records.append(
            {
                "tpep_pickup_datetime": f"2019-01-10 {(i % 24):02d}:{(i % 60):02d}:00",
                "tpep_dropoff_datetime": f"2019-01-10 {(i % 24):02d}:{(i % 60):02d}:00",
                "trip_distance": dist,
                "fare_amount": 10.0 + (i % 20),
                "passenger_count": 1 + (i % 4),
                "PULocationID": 10 + (i % 30),
                "DOLocationID": 20 + (i % 30),
            }
        )
    df = pd.DataFrame(records)
    pickup_dt = pd.to_datetime(df["tpep_pickup_datetime"])
    df["tpep_dropoff_datetime"] = pickup_dt + pd.Timedelta(minutes=15)
    df.to_parquet(tmp_path / "yellow_tripdata_2019-01.parquet")
    return tmp_path


def test_clean_and_label_taxi_data() -> None:
    """Verify data filtering and congestion labeling logic."""
    raw = pd.DataFrame(
        {
            "tpep_pickup_datetime": ["2019-01-01 10:00:00", "2019-01-01 12:00:00"],
            "tpep_dropoff_datetime": [
                "2019-01-01 10:30:00",
                "2019-01-01 12:05:00",
            ],  # 30 min vs 5 min
            "trip_distance": [2.0, 5.0],  # 4 mph (congested) vs 60 mph (not congested)
            "fare_amount": [10.0, 15.0],
            "passenger_count": [1, 2],
        }
    )
    cleaned = clean_and_label_taxi_data(raw)
    assert len(cleaned) == 2
    assert cleaned.loc[0, "is_congested_trip"] == 1  # 4 mph < 12 mph
    assert cleaned.loc[1, "is_congested_trip"] == 0  # 60 mph >= 12 mph


def test_find_optimal_threshold() -> None:
    """Verify optimal threshold computation from probabilities."""
    y_true = np.array([0, 0, 1, 1, 1])
    y_proba = np.array([0.1, 0.2, 0.7, 0.8, 0.9])
    thresh, f1, prec, rec = find_optimal_threshold(y_true, y_proba)
    assert 0.2 < thresh <= 0.8
    assert f1 == 1.0


def test_train_mobility_pipeline_end_to_end(mock_taxi_dir: Path, tmp_path: Path) -> None:
    """Verify complete training, ONNX export, and metadata generation."""
    artifacts_dir = tmp_path / "artifacts"
    meta = train_mobility_pipeline(
        data_dir=mock_taxi_dir,
        artifacts_dir=artifacts_dir,
        model_version="v1.0.0-test",
        samples_per_month=400,
    )

    # Check metadata structure
    assert meta["model_version"] == "v1.0.0-test"
    assert "metrics" in meta
    assert "suggested_guardrails" in meta
    assert meta["suggested_guardrails"]["min_roc_auc"] > 0.0

    # Check generated files
    assert (artifacts_dir / "mobility_model.onnx").exists()
    assert (artifacts_dir / "transformer.joblib").exists()
    assert (artifacts_dir / "reference_data.parquet").exists()
    assert (artifacts_dir / "metadata.json").exists()

    # Verify ONNX model executes properly
    sess = ort.InferenceSession(str(artifacts_dir / "mobility_model.onnx"))
    input_name = sess.get_inputs()[0].name
    dummy_input = np.ones((1, meta["n_features"]), dtype=np.float32)
    res = sess.run(None, {input_name: dummy_input})
    assert len(res) >= 2  # labels and probabilities
