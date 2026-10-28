"""Evaluation script demonstrating real-world concept and covariate drift on NYC Taxi data (2020).

Evaluates the pre-drift trained LightGBM model across consecutive periods of 2020:
- Jan 2020 (Stable pre-lockdown baseline)
- Feb 2020 (Stable pre-lockdown)
- March 2020 Early (Normal) vs Late (Lockdown)

Generates statistical drift reports via Evidently and plots performance degradation curves.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from src.domains.mobility.transformer import MobilityFeatureTransformer
from src.models.train_mobility import (
    DEFAULT_ARTIFACTS_DIR,
    DEFAULT_TAXI_DIR,
    clean_and_label_taxi_data,
)

OUTPUT_DIR: Path = Path("artifacts/mobility")


def evaluate_temporal_performance(
    data_dir: Path = DEFAULT_TAXI_DIR,
    artifacts_dir: Path = DEFAULT_ARTIFACTS_DIR,
    samples_per_slice: int = 25_000,
) -> dict[str, Any]:
    """Evaluate trained model across multiple temporal slices in 2020.

    Args:
        data_dir: Directory containing taxi parquet records.
        artifacts_dir: Path containing saved model artifacts and transformer.
        samples_per_slice: Sample count for each evaluation window.

    Returns:
        Dictionary mapping period labels to performance metrics.
    """
    transformer_path = artifacts_dir / "transformer.joblib"
    metadata_path = artifacts_dir / "metadata.json"

    if not transformer_path.exists() or not metadata_path.exists():
        raise FileNotFoundError("Mobility artifacts not found. Run train_mobility.py first.")

    transformer = MobilityFeatureTransformer.load(transformer_path)
    with open(metadata_path) as f:
        meta = json.load(f)

    opt_threshold = float(meta["metrics"]["optimal_threshold"])

    # Load trained LightGBM model from native training script or reconstruct from parquet
    # For fast python evaluation we train a replica or score using ONNX runtime
    # Let's inspect test periods
    periods = {
        "2020-01 (Pre-COVID)": data_dir / "yellow_tripdata_2020-01.parquet",
        "2020-02 (Pre-COVID)": data_dir / "yellow_tripdata_2020-02.parquet",
        "2020-03-Early (Pre-Lockdown)": data_dir / "yellow_tripdata_2020-03.parquet",
        "2020-03-Late (Lockdown)": data_dir / "yellow_tripdata_2020-03.parquet",
    }

    results: dict[str, dict[str, float]] = {}

    import onnxruntime as ort

    onnx_path = artifacts_dir / "mobility_model.onnx"
    sess = ort.InferenceSession(str(onnx_path))
    input_name = sess.get_inputs()[0].name

    for period_name, file_path in periods.items():
        if not file_path.exists():
            continue

        raw = pd.read_parquet(file_path)
        raw["tpep_pickup_datetime"] = pd.to_datetime(raw["tpep_pickup_datetime"])

        if "Early" in period_name:
            filtered = raw[raw["tpep_pickup_datetime"] < "2020-03-15"]
        elif "Late" in period_name:
            filtered = raw[raw["tpep_pickup_datetime"] >= "2020-03-22"]
        else:
            filtered = raw

        sampled = filtered.sample(n=min(samples_per_slice, len(filtered)), random_state=42)
        cleaned = clean_and_label_taxi_data(sampled)

        X = transformer.transform_df(cleaned).values.astype(np.float32)
        y = cleaned["is_congested_trip"].values.astype(np.int32)

        onnx_out = sess.run(None, {input_name: X})
        probas = np.array([d[1] for d in onnx_out[1]], dtype=np.float32)
        preds = (probas >= opt_threshold).astype(int)

        roc = float(roc_auc_score(y, probas))
        pr = float(average_precision_score(y, probas))
        f1 = float(f1_score(y, preds))
        congested_rate = float(y.mean())

        results[period_name] = {
            "roc_auc": round(roc, 4),
            "pr_auc": round(pr, 4),
            "f1_score": round(f1, 4),
            "actual_congestion_rate": round(congested_rate, 4),
        }

    return results


def generate_evidently_drift_report(
    data_dir: Path = DEFAULT_TAXI_DIR,
    artifacts_dir: Path = DEFAULT_ARTIFACTS_DIR,
    output_dir: Path = OUTPUT_DIR,
) -> Path:
    """Generate comprehensive Evidently drift report comparing 2019 baseline vs March 2020.

    Args:
        data_dir: Directory containing taxi parquet records.
        artifacts_dir: Path containing reference data and transformer.
        output_dir: Destination directory for HTML drift report.

    Returns:
        Path to saved HTML drift report.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    ref_path = artifacts_dir / "reference_data.parquet"
    lockdown_path = data_dir / "yellow_tripdata_2020-03.parquet"

    ref_df = pd.read_parquet(ref_path)

    raw_2020 = pd.read_parquet(lockdown_path)
    raw_2020["tpep_pickup_datetime"] = pd.to_datetime(raw_2020["tpep_pickup_datetime"])
    lockdown_df = raw_2020[raw_2020["tpep_pickup_datetime"] >= "2020-03-25"]
    curr_sample = clean_and_label_taxi_data(lockdown_df.sample(n=3000, random_state=42))

    # Features to analyze with Evidently
    eval_cols = [
        "trip_distance",
        "passenger_count",
        "duration_minutes",
        "speed_mph",
        "is_congested_trip",
    ]

    report = Report([DataDriftPreset()])
    snapshot = report.run(
        reference_data=ref_df[eval_cols],
        current_data=curr_sample[eval_cols],
    )

    html_path = output_dir / "latest_drift_report.html"
    json_path = output_dir / "latest_drift_report.json"

    snapshot.save_html(str(html_path))
    json_str = json.dumps(snapshot.dict(), indent=2)
    with open(json_path, "w") as f:
        f.write(json_str)

    return html_path


if __name__ == "__main__":
    print("=" * 70)
    print("EVALUATION: Temporal Performance Degradation on 2020 Data")
    perf = evaluate_temporal_performance()
    for period, metrics in perf.items():
        print(f"\n[{period}]")
        for k, v in metrics.items():
            print(f"  {k}: {v}")

    print("\n" + "=" * 70)
    print("GENERATING EVIDENTLY STATISTICAL DRIFT REPORT...")
    html_report = generate_evidently_drift_report()
    print(f"Evidently report generated at: {html_report}")
    print("=" * 70)
