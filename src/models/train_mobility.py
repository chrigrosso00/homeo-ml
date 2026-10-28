"""Offline model training and ONNX serialization pipeline for urban mobility.

Trains a LightGBM traffic congestion / SLA breach classifier using chronological
temporal splitting on NYC taxi records, evaluates performance against dummy baselines,
derives dynamic candidate promotion guardrails, and exports to low-latency ONNX runtime.
"""

import json
import logging
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import onnx
import onnxruntime as ort
import pandas as pd
from onnxmltools import convert_lightgbm
from onnxmltools.convert.common.data_types import FloatTensorType
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.domains.mobility.transformer import MobilityFeatureTransformer

logger = logging.getLogger("mobility.training")

DEFAULT_TAXI_DIR: Path = Path("data/raw/taxi")
DEFAULT_ARTIFACTS_DIR: Path = Path("artifacts/mobility")


def clean_and_label_taxi_data(df: pd.DataFrame) -> pd.DataFrame:
    """Filter raw records and compute ground truth traffic congestion binary label.

    A trip is labeled as congested (1) if effective speed < 12.0 mph (19.3 km/h).

    Args:
        df: Raw DataFrame of taxi trip records.

    Returns:
        Cleaned DataFrame with computed duration and 'is_congested_trip' target.
    """
    df = df.copy()

    # Valid physical bounds
    df = df[
        (df["trip_distance"] > 0)
        & (df["trip_distance"] < 100)
        & (df["fare_amount"] > 2.5)
        & (df["fare_amount"] < 500)
        & (df["passenger_count"] > 0)
        & (df["passenger_count"] <= 6)
    ]

    pickup = pd.to_datetime(df["tpep_pickup_datetime"])
    dropoff = pd.to_datetime(df["tpep_dropoff_datetime"])
    duration_min = (dropoff - pickup).dt.total_seconds() / 60.0
    df["duration_minutes"] = duration_min
    df = df[(df["duration_minutes"] >= 1.0) & (df["duration_minutes"] <= 180.0)]

    # Compute speed in miles per hour
    speed_mph = df["trip_distance"] / (df["duration_minutes"] / 60.0)
    df["speed_mph"] = speed_mph
    df["is_congested_trip"] = (df["speed_mph"] < 12.0).astype(int)

    return df


def load_and_split_chronological(
    data_dir: Path = DEFAULT_TAXI_DIR,
    months: list[str] | None = None,
    samples_per_month: int = 50_000,
    train_ratio: float = 0.8,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load sampled NYC taxi records across multiple months with chronological split.

    Preserves time-ordering to eliminate look-ahead bias and data leakage.

    Args:
        data_dir: Directory containing yellow_tripdata_*.parquet files.
        months: List of month identifiers (e.g. ['2019-01', '2019-02', '2019-03']).
        samples_per_month: Uniform random samples per month to maintain efficiency.
        train_ratio: Fraction of historical time series for training.

    Returns:
        Tuple of (train_df, val_df) ordered chronologically.
    """
    if months is None:
        months = ["2019-01", "2019-02", "2019-03"]

    dfs: list[pd.DataFrame] = []
    for m in months:
        file_path = data_dir / f"yellow_tripdata_{m}.parquet"
        if not file_path.exists():
            logger.warning("Month file %s not found, skipping.", file_path)
            continue
        raw_df = pd.read_parquet(file_path)
        sample_size = min(samples_per_month, len(raw_df))
        sampled = raw_df.sample(n=sample_size, random_state=42)
        dfs.append(sampled)

    if not dfs:
        raise FileNotFoundError(f"No parquet files found in {data_dir} for months {months}")

    combined = pd.concat(dfs, ignore_index=True)
    combined["tpep_pickup_datetime"] = pd.to_datetime(combined["tpep_pickup_datetime"])
    combined = combined.sort_values("tpep_pickup_datetime").reset_index(drop=True)

    cleaned = clean_and_label_taxi_data(combined)
    split_idx = int(len(cleaned) * train_ratio)
    train_df = cleaned.iloc[:split_idx].copy().reset_index(drop=True)
    val_df = cleaned.iloc[split_idx:].copy().reset_index(drop=True)

    return train_df, val_df


def find_optimal_threshold(
    y_true: np.ndarray, y_proba: np.ndarray
) -> tuple[float, float, float, float]:
    """Calculate the decision threshold that maximizes F1-score on validation probabilities.

    Args:
        y_true: Ground truth binary labels.
        y_proba: Predicted probabilities for the congested class.

    Returns:
        Tuple of (optimal_threshold, f1, precision, recall).
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    # Avoid division by zero
    f1_scores = np.where(
        (precisions + recalls) > 0,
        2 * (precisions * recalls) / (precisions + recalls),
        0.0,
    )
    opt_idx = int(np.argmax(f1_scores[:-1])) if len(thresholds) > 0 else 0
    opt_thresh = float(thresholds[opt_idx]) if len(thresholds) > 0 else 0.5
    opt_f1 = float(f1_scores[opt_idx]) if len(thresholds) > 0 else 0.0
    opt_prec = float(precisions[opt_idx]) if len(thresholds) > 0 else 0.0
    opt_rec = float(recalls[opt_idx]) if len(thresholds) > 0 else 0.0

    return opt_thresh, opt_f1, opt_prec, opt_rec


def export_to_onnx(model: lgb.LGBMClassifier, n_features: int, destination_path: Path) -> None:
    """Export fitted LightGBM model to ONNX runtime format.

    Args:
        model: Trained LightGBM classifier.
        n_features: Number of input features expected by the model.
        destination_path: Path to save the .onnx binary graph.
    """
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    initial_type = [("float_input", FloatTensorType([None, n_features]))]
    onnx_model = convert_lightgbm(
        model,
        initial_types=initial_type,
        target_opset=15,
    )
    onnx.save_model(onnx_model, str(destination_path))


def train_mobility_pipeline(
    data_dir: Path = DEFAULT_TAXI_DIR,
    artifacts_dir: Path = DEFAULT_ARTIFACTS_DIR,
    model_version: str = "v1.0.0",
    samples_per_month: int = 50_000,
) -> dict[str, Any]:
    """Execute end-to-end training, ONNX serialization, and dynamic guardrail calibration.

    Args:
        data_dir: Root directory of taxi parquet files.
        artifacts_dir: Directory where model artifacts and metadata are written.
        model_version: Semantic release string for this model build.
        samples_per_month: Sample count per month to balance speed and accuracy.

    Returns:
        Metadata dictionary summarizing training metrics and artifact locations.
    """
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load and chronologically split data
    train_df, val_df = load_and_split_chronological(
        data_dir=data_dir,
        months=["2019-01", "2019-02", "2019-03"],
        samples_per_month=samples_per_month,
        train_ratio=0.8,
    )

    # 2. Fit feature transformer
    transformer = MobilityFeatureTransformer(top_locations_count=25)
    transformer.fit(train_df)

    X_train = transformer.transform_df(train_df).values.astype(np.float32)
    y_train = train_df["is_congested_trip"].values.astype(np.int32)

    X_val = transformer.transform_df(val_df).values.astype(np.float32)
    y_val = val_df["is_congested_trip"].values.astype(np.int32)

    n_features = X_train.shape[1]

    # 3. Train Baseline Dummy Classifier (for honest benchmark)
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train, y_train)
    dummy_val_preds = dummy.predict(X_val)
    dummy_f1 = float(f1_score(y_val, dummy_val_preds, zero_division=0))

    # 4. Train LightGBM Classifier
    clf = lgb.LGBMClassifier(
        n_estimators=120,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42,
        verbose=-1,
    )
    clf.fit(X_train, y_train)

    # 5. Evaluate on Validation Set
    y_val_proba = clf.predict_proba(X_val)[:, 1]
    roc_auc = float(roc_auc_score(y_val, y_val_proba))
    pr_auc = float(average_precision_score(y_val, y_val_proba))
    opt_thresh, opt_f1, opt_prec, opt_rec = find_optimal_threshold(y_val, y_val_proba)

    # Default threshold 0.5 metrics
    default_preds = (y_val_proba >= 0.5).astype(int)
    default_f1 = float(f1_score(y_val, default_preds))
    default_prec = float(precision_score(y_val, default_preds))
    default_rec = float(recall_score(y_val, default_preds))

    # 6. Export to ONNX and verify numerical parity
    transformer_path = artifacts_dir / "transformer.joblib"
    onnx_path = artifacts_dir / "mobility_model.onnx"
    reference_path = artifacts_dir / "reference_data.parquet"
    metadata_path = artifacts_dir / "metadata.json"

    transformer.save(transformer_path)
    export_to_onnx(clf, n_features=n_features, destination_path=onnx_path)

    # Verify ONNX runtime execution parity
    sess = ort.InferenceSession(str(onnx_path))
    input_name = sess.get_inputs()[0].name
    onnx_out = sess.run(None, {input_name: X_val[:100]})
    # ONNX produces list of dicts: [{0: p0, 1: p1}, ...]
    onnx_proba = np.array([d[1] for d in onnx_out[1]], dtype=np.float32)
    np.testing.assert_allclose(onnx_proba, y_val_proba[:100], atol=1e-5)

    # 7. Save Reference Dataset for Evidently drift monitoring
    ref_sample = val_df.sample(n=min(5000, len(val_df)), random_state=42).copy()
    ref_X = transformer.transform_df(ref_sample).values.astype(np.float32)
    ref_sample["congestion_probability"] = clf.predict_proba(ref_X)[:, 1]
    ref_sample.to_parquet(reference_path)

    # 8. Compute dynamically calibrated guardrails
    suggested_guardrails = {
        "min_roc_auc": round(roc_auc * 0.90, 4),  # 90% of validation ROC-AUC
        "min_pr_auc": round(pr_auc * 0.85, 4),
        "min_f1": round(opt_f1 * 0.85, 4),
        "max_latency_p95_ms": 10.0,
    }

    metadata = {
        "model_version": model_version,
        "domain": "urban_mobility",
        "target": "is_congested_trip",
        "algorithm": "LightGBM",
        "n_features": n_features,
        "feature_names": transformer.feature_names_,
        "metrics": {
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "optimal_threshold": opt_thresh,
            "optimal_f1": opt_f1,
            "optimal_precision": opt_prec,
            "optimal_recall": opt_rec,
            "default_threshold_f1": default_f1,
            "default_precision": default_prec,
            "default_recall": default_rec,
            "dummy_baseline_f1": dummy_f1,
        },
        "suggested_guardrails": suggested_guardrails,
        "train_period": {
            "start": str(train_df["tpep_pickup_datetime"].min()),
            "end": str(train_df["tpep_pickup_datetime"].max()),
            "samples": len(train_df),
        },
        "val_period": {
            "start": str(val_df["tpep_pickup_datetime"].min()),
            "end": str(val_df["tpep_pickup_datetime"].max()),
            "samples": len(val_df),
        },
        "artifacts": {
            "onnx_model": str(onnx_path),
            "transformer": str(transformer_path),
            "reference_data": str(reference_path),
        },
    }

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return metadata


def train_mobility_recent_window(
    data_dir: Path = DEFAULT_TAXI_DIR,
    artifacts_dir: Path = Path("artifacts/canary_mobility"),
    model_version: str = "v1.1.0-canary",
    recent_month: str = "2020-03",
    sample_size: int = 40_000,
) -> dict[str, Any]:
    """Retrain candidate model on the most recent data window to adapt to drift.

    Args:
        data_dir: Root directory of taxi parquet records.
        artifacts_dir: Destination path for canary model and transformer.
        model_version: Target semantic release version.
        recent_month: Most recent month identifier reflecting drifted conditions.
        sample_size: Number of recent samples used for retraining.

    Returns:
        Metadata dictionary for the newly trained candidate model.
    """
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    recent_file = data_dir / f"yellow_tripdata_{recent_month}.parquet"
    if not recent_file.exists():
        raise FileNotFoundError(f"Recent window file {recent_file} not found")

    raw = pd.read_parquet(recent_file)
    raw["tpep_pickup_datetime"] = pd.to_datetime(raw["tpep_pickup_datetime"])
    # Sort chronologically to preserve real-world causality
    sorted_df = raw.sort_values("tpep_pickup_datetime").reset_index(drop=True)
    # Take the most recent samples
    recent_slice = sorted_df.tail(sample_size).copy()
    cleaned = clean_and_label_taxi_data(recent_slice)

    split_idx = int(len(cleaned) * 0.8)
    train_df = cleaned.iloc[:split_idx].copy().reset_index(drop=True)
    val_df = cleaned.iloc[split_idx:].copy().reset_index(drop=True)

    # Fit fresh transformer on recent conditions
    transformer = MobilityFeatureTransformer(top_locations_count=25)
    transformer.fit(train_df)

    X_train = transformer.transform_df(train_df).values.astype(np.float32)
    y_train = train_df["is_congested_trip"].values.astype(np.int32)

    X_val = transformer.transform_df(val_df).values.astype(np.float32)
    y_val = val_df["is_congested_trip"].values.astype(np.int32)

    n_features = X_train.shape[1]

    clf = lgb.LGBMClassifier(
        n_estimators=100,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42,
        verbose=-1,
    )
    clf.fit(X_train, y_train)

    y_val_proba = clf.predict_proba(X_val)[:, 1]
    roc_auc = float(roc_auc_score(y_val, y_val_proba))
    pr_auc = float(average_precision_score(y_val, y_val_proba))
    opt_thresh, opt_f1, opt_prec, opt_rec = find_optimal_threshold(y_val, y_val_proba)

    # Save artifacts
    transformer_path = artifacts_dir / "transformer.joblib"
    onnx_path = artifacts_dir / "mobility_model.onnx"
    reference_path = artifacts_dir / "reference_data.parquet"
    metadata_path = artifacts_dir / "metadata.json"

    transformer.save(transformer_path)
    export_to_onnx(clf, n_features=n_features, destination_path=onnx_path)

    ref_sample = val_df.sample(n=min(2000, len(val_df)), random_state=42).copy()
    ref_sample.to_parquet(reference_path)

    metadata = {
        "model_version": model_version,
        "domain": "urban_mobility",
        "target": "is_congested_trip",
        "algorithm": "LightGBM",
        "n_features": n_features,
        "feature_names": transformer.feature_names_,
        "metrics": {
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "optimal_threshold": opt_thresh,
            "optimal_f1": opt_f1,
            "optimal_precision": opt_prec,
            "optimal_recall": opt_rec,
        },
        "artifacts": {
            "onnx_model": str(onnx_path),
            "transformer": str(transformer_path),
            "reference_data": str(reference_path),
        },
    }

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(
        "Candidate %s retrained on recent window %s: ROC-AUC=%.4f, PR-AUC=%.4f, F1=%.4f",
        model_version,
        recent_month,
        roc_auc,
        pr_auc,
        opt_f1,
    )
    return metadata


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    meta = train_mobility_pipeline()
    print("Training finished successfully:")
    print(json.dumps(meta["metrics"], indent=2))
