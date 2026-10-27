"""Unit tests for statistical drift detection and sliding window evaluation."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.monitoring.drift_detector import DriftMonitoringEngine, DriftReport

REFERENCE_PATH: Path = Path("artifacts/models/reference_data.parquet")


@pytest.fixture
def engine(tmp_path: Path) -> DriftMonitoringEngine:
    """Fixture providing initialized DriftMonitoringEngine with temporary report dir."""
    assert REFERENCE_PATH.exists(), "Reference dataset missing. Run train.py first."
    return DriftMonitoringEngine(
        reference_path=REFERENCE_PATH,
        window_size=300,
        drift_share_threshold=0.40,
        reports_dir=tmp_path / "monitoring",
    )


def test_drift_detection_no_drift(engine: DriftMonitoringEngine) -> None:
    """Verify that a sample drawn from the reference distribution exhibits no dataset drift."""
    ref_df = pd.read_parquet(REFERENCE_PATH)
    # Draw an unperturbed sample
    sample_current = ref_df.sample(n=300, random_state=42).copy()

    report = engine.analyze_dataframe(sample_current)
    assert isinstance(report, DriftReport)
    assert report.window_size == 300
    assert report.features_analyzed >= 7
    # An identical distribution should have low drifted features
    assert report.share_of_drifted_features < 0.40
    assert report.dataset_drift_detected is False


def test_drift_detection_with_botnet_covariate_shift(engine: DriftMonitoringEngine) -> None:
    """Verify that an injected botnet pattern triggers dataset drift and pinpoints columns."""
    ref_df = pd.read_parquet(REFERENCE_PATH)
    drifted_current = ref_df.sample(n=300, random_state=99).copy()

    # Inject severe covariate shift: instant checkout, browser shift, inflated value
    n_samples = len(drifted_current)
    drifted_current["time_since_signup_sec"] = np.random.uniform(0.1, 1.5, size=n_samples)
    drifted_current["browser"] = "Opera"
    drifted_current["source"] = "Direct"
    drifted_current["purchase_value"] = np.random.uniform(500.0, 1500.0, size=n_samples)
    drifted_current["fraud_probability"] = np.random.uniform(0.7, 0.99, size=n_samples)

    report = engine.analyze_dataframe(drifted_current)
    assert report.dataset_drift_detected is True
    assert report.share_of_drifted_features >= 0.40

    drifted_cols = [f.column for f in report.drifted_features if f.is_drifted]
    assert "time_since_signup_sec" in drifted_cols
    assert "browser" in drifted_cols
    assert "purchase_value" in drifted_cols

    # Verify output prediction drift detected
    assert report.prediction_drift is not None
    assert report.prediction_drift.is_drifted is True

    # Verify generated report artifacts
    assert Path(report.html_report_path).exists()
    assert (engine.reports_dir / "latest_drift_report.json").exists()


def test_sliding_window_event_ingestion(engine: DriftMonitoringEngine) -> None:
    """Verify that ingesting events incrementally triggers analysis when window_size is hit."""
    ref_df = pd.read_parquet(REFERENCE_PATH)
    records = ref_df.head(engine.window_size).to_dict(orient="records")

    report: DriftReport | None = None
    for i, record in enumerate(records):
        eval_result = engine.ingest_event(record)
        if i < engine.window_size - 1:
            assert eval_result is None
        else:
            report = eval_result

    assert report is not None
    assert report.window_size == engine.window_size
