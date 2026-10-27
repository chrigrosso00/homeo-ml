"""Unit tests for the trained ONNX model inference and latency benchmarks."""

import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest

ONNX_MODEL_PATH: Path = Path("artifacts/models/fraud_model.onnx")
METADATA_PATH: Path = Path("artifacts/models/metadata.json")


@pytest.fixture
def onnx_session() -> ort.InferenceSession:
    """Fixture providing initialized ONNX runtime inference session."""
    assert ONNX_MODEL_PATH.exists(), "ONNX model artifact not found. Run train.py first."
    return ort.InferenceSession(str(ONNX_MODEL_PATH))


@pytest.fixture
def model_metadata() -> dict:
    """Fixture providing model metadata dict."""
    assert METADATA_PATH.exists(), "Metadata artifact not found. Run train.py first."
    with open(METADATA_PATH) as f:
        return json.load(f)


def test_onnx_model_metadata(model_metadata: dict) -> None:
    """Verify metadata contains required MLOps tracking schema."""
    assert model_metadata["model_version"] == "v1.0.0"
    assert model_metadata["algorithm"] == "LightGBM"
    assert model_metadata["n_features"] == 41
    assert "optimal_threshold" in model_metadata["metrics"]
    assert 0.0 < model_metadata["metrics"]["optimal_threshold"] < 1.0


def test_onnx_inference_output_shape(onnx_session: ort.InferenceSession) -> None:
    """Verify input/output shapes and valid probability ranges."""
    input_name = onnx_session.get_inputs()[0].name
    # Generate 5 sample transactions with 41 features
    sample_batch = np.random.randn(5, 41).astype(np.float32)

    outputs = onnx_session.run(None, {input_name: sample_batch})
    assert len(outputs) == 2, "Expected label and probabilities outputs from LightGBM ONNX"

    # outputs[1] is list of dictionaries {0: prob0, 1: prob1}
    probas = outputs[1]
    assert len(probas) == 5
    for p in probas:
        p0 = p[0]
        p1 = p[1]
        assert 0.0 <= p0 <= 1.0
        assert 0.0 <= p1 <= 1.0
        assert abs((p0 + p1) - 1.0) < 1e-4


def test_onnx_inference_latency_benchmark(onnx_session: ort.InferenceSession) -> None:
    """Verify single-transaction inference SLA is sub-5ms."""
    input_name = onnx_session.get_inputs()[0].name
    single_tx = np.random.randn(1, 41).astype(np.float32)

    # Warmup
    for _ in range(10):
        _ = onnx_session.run(None, {input_name: single_tx})

    # Benchmark 100 sequential inferences
    latencies: list[float] = []
    for _ in range(100):
        t0 = time.perf_counter()
        _ = onnx_session.run(None, {input_name: single_tx})
        latencies.append((time.perf_counter() - t0) * 1000.0)

    p95_latency = float(np.percentile(latencies, 95))
    mean_latency = float(np.mean(latencies))

    print(f"\nONNX Inference Latency: Mean = {mean_latency:.3f}ms | P95 = {p95_latency:.3f}ms")
    # P95 must be strictly below 5ms (typically <0.1ms on modern CPU)
    assert p95_latency < 5.0, f"P95 latency {p95_latency}ms exceeds 5ms SLA!"
