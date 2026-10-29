"""FastAPI Model Serving layer for real-time fraud scoring.

Serves low-latency predictions via ONNX Runtime and streams asynchronous
telemetry events to Redpanda / Kafka.
"""

import json
import logging
import os
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path

import onnxruntime as ort
from fastapi import FastAPI, HTTPException, status

from src.data.schemas import PredictionResponse, TelemetryEvent, TransactionInput
from src.data.transformer import FraudFeatureTransformer
from src.domains.mobility.schemas import (
    TripInput,
    TripPredictionResponse,
    TripTelemetryPayload,
)
from src.domains.mobility.transformer import MobilityFeatureTransformer
from src.serving.telemetry import KafkaTelemetryProducer

logger = logging.getLogger("serving.app")

MODEL_DIR: Path = Path(os.getenv("MODEL_DIR", "artifacts/models"))
DEFAULT_MODEL_PATH: Path = Path(os.getenv("MODEL_PATH", str(MODEL_DIR / "fraud_model.onnx")))
DEFAULT_TRANSFORMER_PATH: Path = Path(
    os.getenv("TRANSFORMER_PATH", str(MODEL_DIR / "transformer.joblib"))
)
DEFAULT_METADATA_PATH: Path = Path(os.getenv("METADATA_PATH", str(MODEL_DIR / "metadata.json")))

MOBILITY_MODEL_DIR: Path = Path(os.getenv("MOBILITY_MODEL_DIR", "artifacts/mobility"))
DEFAULT_MOBILITY_MODEL_PATH: Path = Path(
    os.getenv("MOBILITY_MODEL_PATH", str(MOBILITY_MODEL_DIR / "mobility_model.onnx"))
)
DEFAULT_MOBILITY_TRANSFORMER_PATH: Path = Path(
    os.getenv("MOBILITY_TRANSFORMER_PATH", str(MOBILITY_MODEL_DIR / "transformer.joblib"))
)
DEFAULT_MOBILITY_METADATA_PATH: Path = Path(
    os.getenv("MOBILITY_METADATA_PATH", str(MOBILITY_MODEL_DIR / "metadata.json"))
)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Lifespan context manager for loading models and managing telemetry resources.

    Args:
        app: FastAPI application instance.

    Yields:
        None when application startup is complete.
    """
    logger.info("Starting up Fraud Serving API...")

    if not DEFAULT_MODEL_PATH.exists() or not DEFAULT_TRANSFORMER_PATH.exists():
        logger.error("Required model artifacts missing in artifacts/models/")
        raise RuntimeError("Model or transformer artifacts not found. Run train.py first.")

    # 1. Initialize ONNX runtime session with multithreading optimization
    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = 2
    app.state.onnx_session = ort.InferenceSession(
        str(DEFAULT_MODEL_PATH), sess_options=session_options
    )
    app.state.onnx_input_name = app.state.onnx_session.get_inputs()[0].name

    # 2. Load feature transformer and metadata
    app.state.transformer = FraudFeatureTransformer.load(DEFAULT_TRANSFORMER_PATH)

    with open(DEFAULT_METADATA_PATH) as f:
        metadata = json.load(f)
    app.state.metadata = metadata
    app.state.model_version = metadata.get("model_version", "v1.0.0")
    app.state.optimal_threshold = float(metadata["metrics"].get("optimal_threshold", 0.5))

    # 3. Load mobility model if available
    if DEFAULT_MOBILITY_MODEL_PATH.exists() and DEFAULT_MOBILITY_TRANSFORMER_PATH.exists():
        app.state.mobility_onnx_session = ort.InferenceSession(
            str(DEFAULT_MOBILITY_MODEL_PATH), sess_options=session_options
        )
        app.state.mobility_onnx_input_name = app.state.mobility_onnx_session.get_inputs()[0].name
        app.state.mobility_transformer = MobilityFeatureTransformer.load(
            DEFAULT_MOBILITY_TRANSFORMER_PATH
        )
        if DEFAULT_MOBILITY_METADATA_PATH.exists():
            with open(DEFAULT_MOBILITY_METADATA_PATH) as f:
                mob_meta = json.load(f)
            app.state.mobility_metadata = mob_meta
            app.state.mobility_model_version = mob_meta.get("model_version", "v1.0.0")
            app.state.mobility_optimal_threshold = float(
                mob_meta["metrics"].get("optimal_threshold", 0.5)
            )
        else:
            app.state.mobility_model_version = "v1.0.0"
            app.state.mobility_optimal_threshold = 0.5
        logger.info(
            "Mobility Serving initialized: Model %s, Threshold %.4f",
            app.state.mobility_model_version,
            app.state.mobility_optimal_threshold,
        )

    # 4. Initialize Redpanda telemetry producer
    app.state.telemetry = KafkaTelemetryProducer()

    logger.info(
        "Serving initialized: Model %s, Threshold %.4f, Input %s",
        app.state.model_version,
        app.state.optimal_threshold,
        app.state.onnx_input_name,
    )

    yield

    # Shutdown: flush Kafka buffer
    logger.info("Shutting down Fraud Serving API: Flushing telemetry buffer...")
    remaining = app.state.telemetry.flush(timeout=3.0)
    logger.info("Shutdown complete. Unflushed messages: %d", remaining)


app = FastAPI(
    title="Self-Healing MLOps Runtime - Fraud Serving Layer",
    description="Low-latency fraud scoring with ONNX Runtime & Redpanda telemetry buffer",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", status_code=status.HTTP_200_OK)
def health_check() -> dict[str, str | float | int]:
    """Check service health and inspect currently deployed model metadata.

    Returns:
        Dictionary containing operational status and model version attributes.
    """
    if not hasattr(app.state, "onnx_session"):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model uninitialized"
        )

    return {
        "status": "healthy",
        "model_version": app.state.model_version,
        "optimal_threshold": app.state.optimal_threshold,
        "n_features": len(app.state.transformer.feature_names_),
    }


@app.post("/predict", response_model=PredictionResponse, status_code=status.HTTP_200_OK)
def predict(tx: TransactionInput) -> PredictionResponse:
    """Score incoming transaction for fraud probability in real time.

    Executes feature transformation, ONNX inference, and fires an asynchronous
    telemetry event to Redpanda without adding latency to the response.

    Args:
        tx: Validated customer transaction payload.

    Returns:
        PredictionResponse with fraud probability, binary decision, and latency.
    """
    t0 = time.perf_counter()
    tx_id = str(uuid.uuid4())

    try:
        # 1. Microsecond feature transformation
        features = app.state.transformer.transform_single(tx)

        # 2. ONNX Runtime inference
        outputs = app.state.onnx_session.run(None, {app.state.onnx_input_name: features})
        # LightGBM ONNX outputs: [predicted_label, [{0: prob_0, 1: prob_1}]]
        fraud_proba = float(outputs[1][0][1])
        is_fraud = fraud_proba >= app.state.optimal_threshold

        latency_ms = (time.perf_counter() - t0) * 1000.0

        # 3. Fire-and-forget asynchronous telemetry logging to Redpanda
        time_diff = max((tx.purchase_time - tx.signup_time).total_seconds(), 0.0)
        country = str(app.state.transformer.ip_lookup.lookup(tx.ip_address))

        telemetry_event = TelemetryEvent(
            transaction_id=tx_id,
            user_id=tx.user_id,
            signup_time=tx.signup_time.isoformat(),
            purchase_time=tx.purchase_time.isoformat(),
            purchase_value=float(tx.purchase_value),
            device_id=tx.device_id,
            source=tx.source,
            browser=tx.browser,
            sex=tx.sex,
            age=tx.age,
            ip_address=float(tx.ip_address),
            country=country,
            time_since_signup_sec=float(time_diff),
            fraud_probability=fraud_proba,
            is_fraud=is_fraud,
            model_version=app.state.model_version,
            latency_ms=latency_ms,
            logged_at=datetime.now(UTC).isoformat(),
        )
        app.state.telemetry.dispatch(telemetry_event)

        return PredictionResponse(
            transaction_id=tx_id,
            fraud_probability=round(fraud_proba, 5),
            is_fraud=is_fraud,
            model_version=app.state.model_version,
            latency_ms=round(latency_ms, 3),
        )

    except Exception as exc:
        logger.error("Error during prediction for tx %s: %s", tx_id, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference failure: {exc}",
        ) from exc


@app.post(
    "/predict/mobility",
    response_model=TripPredictionResponse,
    status_code=status.HTTP_200_OK,
)
def predict_mobility(trip: TripInput) -> TripPredictionResponse:
    """Predict traffic congestion and SLA breach for an urban mobility ride request.

    Executes feature transformation, ONNX inference, and fires an asynchronous
    telemetry event to Redpanda topic 'mobility.telemetry'.

    Args:
        trip: Validated TripInput payload.

    Returns:
        TripPredictionResponse with congestion decision and probability.
    """
    if not hasattr(app.state, "mobility_onnx_session"):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Mobility model is not loaded",
        )

    t0 = time.perf_counter()
    try:
        # 1. Feature extraction
        features = app.state.mobility_transformer.transform_single(trip)

        # 2. ONNX inference
        outputs = app.state.mobility_onnx_session.run(
            None, {app.state.mobility_onnx_input_name: features}
        )
        # outputs: [labels, [{0: p0, 1: p1}]]
        prob_congested = float(outputs[1][0][1])
        is_congested = prob_congested >= app.state.mobility_optimal_threshold

        latency_ms = (time.perf_counter() - t0) * 1000.0

        # 3. Fire-and-forget telemetry to Redpanda
        res = TripPredictionResponse(
            is_congested=is_congested,
            congestion_probability=round(prob_congested, 5),
            model_version=app.state.mobility_model_version,
        )
        payload = TripTelemetryPayload(
            trip_input=trip,
            prediction=res,
            latency_ms=round(latency_ms, 3),
        )
        app.state.telemetry.dispatch(payload, topic="mobility.telemetry")

        return res

    except Exception as exc:
        logger.error("Error during mobility prediction: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Mobility inference failure: {exc}",
        ) from exc
