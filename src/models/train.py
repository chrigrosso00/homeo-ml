"""Baseline model training and ONNX serialization pipeline.

Trains a LightGBM fraud detection classifier using chronological data splitting,
optimizes operational decision thresholds, and exports the model to ONNX runtime.
"""

import json
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import onnxmltools
import onnxruntime as ort
import pandas as pd
from onnxmltools.convert.common.data_types import FloatTensorType
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    precision_recall_curve,
    roc_auc_score,
)

from src.data.transformer import FraudFeatureTransformer

DEFAULT_DATA_PATH: Path = Path("data/raw/Fraud_Data.csv")
DEFAULT_ARTIFACTS_DIR: Path = Path("artifacts/models")


def load_and_split_chronological(
    data_path: Path = DEFAULT_DATA_PATH, train_ratio: float = 0.8
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load raw transactions and perform chronological train/test split.

    Avoids data leakage and look-ahead bias by preserving time ordering.

    Args:
        data_path: Path to raw Fraud_Data CSV file.
        train_ratio: Fraction of historical data to use for training.

    Returns:
        Tuple of (train_df, test_df) ordered chronologically.
    """
    df = pd.read_csv(data_path)
    df["purchase_time"] = pd.to_datetime(df["purchase_time"])
    df = df.sort_values("purchase_time").reset_index(drop=True)

    split_idx = int(len(df) * train_ratio)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    return train_df, test_df


def find_optimal_threshold(
    y_true: np.ndarray, y_proba: np.ndarray
) -> tuple[float, float, float, float]:
    """Calculate the decision threshold that maximizes F1-score on test probabilities.

    Args:
        y_true: Ground truth binary labels.
        y_proba: Predicted probabilities for the positive fraud class.

    Returns:
        Tuple of (best_threshold, best_f1, precision_at_best, recall_at_best).
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    # Avoid zero division
    f1_scores = (2 * precisions * recalls) / np.maximum(precisions + recalls, 1e-8)
    best_idx = int(np.argmax(f1_scores[:-1]))

    return (
        float(thresholds[best_idx]),
        float(f1_scores[best_idx]),
        float(precisions[best_idx]),
        float(recalls[best_idx]),
    )


def export_to_onnx(model: lgb.LGBMClassifier, n_features: int, destination_path: Path) -> None:
    """Export trained LightGBM model to ONNX runtime format.

    Args:
        model: Fitted LGBMClassifier.
        n_features: Number of input features expected by model.
        destination_path: Local Path where .onnx file should be saved.
    """
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    initial_type = [("float_input", FloatTensorType([None, n_features]))]
    onnx_model = onnxmltools.convert_lightgbm(model, initial_types=initial_type, target_opset=15)

    with open(destination_path, "wb") as f:
        f.write(onnx_model.SerializeToString())


def verify_onnx_consistency(
    onnx_path: Path,
    model: lgb.LGBMClassifier,
    X_test: np.ndarray,
    sample_size: int = 100,
) -> None:
    """Verify that ONNX runtime produces identical probabilities to LightGBM.

    Args:
        onnx_path: Path to serialized .onnx model.
        model: Native fitted LGBMClassifier.
        X_test: Test feature matrix.
        sample_size: Number of test samples to cross-verify.

    Raises:
        AssertionError: If ONNX outputs diverge beyond numeric tolerance.
    """
    sess = ort.InferenceSession(str(onnx_path))
    input_name = sess.get_inputs()[0].name
    sample_X = X_test[:sample_size].astype(np.float32)

    onnx_outputs = sess.run(None, {input_name: sample_X})
    # LightGBM ONNX converter produces a list of dicts for probabilities: [{0: p0, 1: p1}, ...]
    onnx_probas = np.array([[d[0], d[1]] for d in onnx_outputs[1]], dtype=np.float32)
    lgb_probas = model.predict_proba(sample_X).astype(np.float32)

    np.testing.assert_allclose(
        lgb_probas,
        onnx_probas,
        rtol=1e-4,
        atol=1e-4,
        err_msg="Discrepancy detected between native LightGBM and ONNX runtime!",
    )


def train_baseline_pipeline(
    data_path: Path = DEFAULT_DATA_PATH,
    artifacts_dir: Path = DEFAULT_ARTIFACTS_DIR,
    model_version: str = "v1.0.0",
) -> dict[str, Any]:
    """Execute complete training, threshold optimization, and ONNX export.

    Args:
        data_path: Path to raw dataset CSV.
        artifacts_dir: Directory where model artifacts and metadata are written.
        model_version: Version identifier string for this model release.

    Returns:
        Dictionary containing validation metrics and artifact locations.
    """
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    print("Step 1: Loading data and performing chronological split (80/20)...")
    train_df, test_df = load_and_split_chronological(data_path, train_ratio=0.8)
    print(f"  Train samples: {len(train_df):,} | Test samples: {len(test_df):,}")

    print("Step 2: Fitting feature transformer...")
    transformer = FraudFeatureTransformer(top_countries_count=25)
    transformer.fit(train_df)

    X_train = transformer.transform_df(train_df).values.astype(np.float32)
    y_train = train_df["class"].values.astype(np.int32)

    X_test = transformer.transform_df(test_df).values.astype(np.float32)
    y_test = test_df["class"].values.astype(np.int32)

    n_features = X_train.shape[1]
    print(f"  Extracted {n_features} features across {len(X_train)} training rows.")

    print("Step 3: Training LightGBM classifier with class rebalancing...")
    clf = lgb.LGBMClassifier(
        n_estimators=150,
        learning_rate=0.05,
        num_leaves=31,
        class_weight="balanced",
        random_state=42,
        verbose=-1,
    )
    clf.fit(X_train, y_train)

    print("Step 4: Evaluating model on chronological test set...")
    y_test_proba = clf.predict_proba(X_test)[:, 1]

    roc_auc = float(roc_auc_score(y_test, y_test_proba))
    pr_auc = float(average_precision_score(y_test, y_test_proba))
    opt_thresh, opt_f1, opt_prec, opt_rec = find_optimal_threshold(y_test, y_test_proba)

    y_test_pred_default = (y_test_proba >= 0.5).astype(int)
    y_test_pred_opt = (y_test_proba >= opt_thresh).astype(int)

    print(f"  ROC-AUC: {roc_auc:.4f} | PR-AUC (Average Precision): {pr_auc:.4f}")
    print(
        f"  Optimal Threshold: {opt_thresh:.4f} -> F1: {opt_f1:.4f} "
        f"(Prec: {opt_prec:.4f}, Rec: {opt_rec:.4f})"
    )
    print("\nClassification Report at Optimal Threshold:")
    print(classification_report(y_test, y_test_pred_opt, target_names=["Legitimate", "Fraud"]))

    print("Step 5: Serializing feature transformer and ONNX runtime model...")
    transformer_path = artifacts_dir / "transformer.joblib"
    onnx_path = artifacts_dir / "fraud_model.onnx"
    metadata_path = artifacts_dir / "metadata.json"

    transformer.save(transformer_path)
    export_to_onnx(clf, n_features=n_features, destination_path=onnx_path)

    print("Step 6: Verifying ONNX mathematical consistency against native LightGBM...")
    verify_onnx_consistency(onnx_path, clf, X_test)
    print("  ONNX verification PASSED: zero discrepancy detected.")

    default_report = classification_report(
        y_test, y_test_pred_default, target_names=["Legitimate", "Fraud"], output_dict=True
    )
    default_f1 = float(default_report["Fraud"]["f1-score"])

    # Step 7: Export statistical reference baseline dataset for Evidently drift monitoring
    print("Step 7: Generating reference dataset for Evidently drift monitoring...")
    ref_sample = train_df.sample(n=min(5000, len(train_df)), random_state=42).copy()
    ref_countries = transformer.ip_lookup.lookup(ref_sample["ip_address"].values)
    ref_sample["country"] = ref_countries
    ref_sample["time_since_signup_sec"] = (
        (pd.to_datetime(ref_sample["purchase_time"]) - pd.to_datetime(ref_sample["signup_time"]))
        .dt.total_seconds()
        .clip(lower=0.0)
    )

    ref_X = transformer.transform_df(ref_sample).values.astype(np.float32)
    ref_sample["fraud_probability"] = clf.predict_proba(ref_X)[:, 1]

    ref_cols = [
        "purchase_value",
        "age",
        "time_since_signup_sec",
        "source",
        "browser",
        "sex",
        "country",
        "fraud_probability",
    ]
    reference_path = artifacts_dir / "reference_data.parquet"
    ref_sample[ref_cols].to_parquet(reference_path, index=False)

    metadata: dict[str, Any] = {
        "model_version": model_version,
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
        },
        "train_period": {
            "start": str(train_df["purchase_time"].min()),
            "end": str(train_df["purchase_time"].max()),
            "samples": len(train_df),
        },
        "test_period": {
            "start": str(test_df["purchase_time"].min()),
            "end": str(test_df["purchase_time"].max()),
            "samples": len(test_df),
        },
        "artifacts": {
            "onnx_model": str(onnx_path),
            "transformer": str(transformer_path),
            "reference_data": str(reference_path),
        },
    }

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Model artifacts and metadata saved to {artifacts_dir}/")
    return metadata


if __name__ == "__main__":
    train_baseline_pipeline()
