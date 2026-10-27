"""Statistical drift monitoring engine using Evidently and Redpanda streaming events.

Consumes real-time inference telemetry from Redpanda, accumulates events into
sliding evaluation windows, and executes multivariate drift detection tests
(Wasserstein distance and Jensen-Shannon divergence).
"""

import json
import logging
from collections import deque
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from confluent_kafka import Consumer, KafkaError, KafkaException
from evidently import Report
from evidently.presets import DataDriftPreset
from pydantic import BaseModel, Field

logger = logging.getLogger("monitoring.drift")

DEFAULT_REFERENCE_PATH: Path = Path("artifacts/models/reference_data.parquet")
DEFAULT_REPORTS_DIR: Path = Path("artifacts/monitoring")


class FeatureDriftDetail(BaseModel):
    """Detailed drift metric for an individual feature column."""

    column: str
    method: str
    score: float
    threshold: float
    is_drifted: bool


class DriftReport(BaseModel):
    """Structured drift assessment report produced for each evaluated window."""

    timestamp: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    window_size: int
    dataset_drift_detected: bool
    share_of_drifted_features: float
    features_analyzed: int
    drifted_features: list[FeatureDriftDetail]
    prediction_drift: FeatureDriftDetail | None = None
    html_report_path: str | None = None


class DriftMonitoringEngine:
    """Sliding-window drift analyzer comparing live telemetry to training baseline."""

    def __init__(
        self,
        reference_path: Path = DEFAULT_REFERENCE_PATH,
        window_size: int = 300,
        drift_share_threshold: float = 0.40,
        reports_dir: Path = DEFAULT_REPORTS_DIR,
    ) -> None:
        """Initialize drift monitoring engine with reference baseline.

        Args:
            reference_path: Path to baseline parquet dataset.
            window_size: Number of transactions per evaluation window.
            drift_share_threshold: Ratio of drifted features required to trigger alert.
            reports_dir: Directory where drift reports and HTML snapshots are saved.
        """
        self.reference_path = reference_path
        self.window_size = window_size
        self.drift_share_threshold = drift_share_threshold
        self.reports_dir = reports_dir
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        self._reference_df: pd.DataFrame | None = None
        self._window_buffer: deque[dict[str, Any]] = deque(maxlen=window_size)
        self._load_reference()

    def _load_reference(self) -> None:
        """Load reference dataset into memory."""
        if not self.reference_path.exists():
            logger.warning(
                "Reference dataset not found at %s. Drift evaluation pending.",
                self.reference_path,
            )
            return

        self._reference_df = pd.read_parquet(self.reference_path)
        logger.info(
            "Loaded reference baseline with %d samples and columns: %s",
            len(self._reference_df),
            list(self._reference_df.columns),
        )

    def analyze_dataframe(self, current_df: pd.DataFrame) -> DriftReport:
        """Execute Evidently statistical tests comparing current DataFrame against reference.

        Args:
            current_df: DataFrame representing live production transactions.

        Returns:
            Structured DriftReport with feature-level breakdown.
        """
        if self._reference_df is None:
            self._load_reference()
            if self._reference_df is None:
                raise RuntimeError("Reference baseline dataset is unavailable!")

        # Align columns to evaluated features
        eval_cols = [c for c in self._reference_df.columns if c in current_df.columns]
        ref_subset = self._reference_df[eval_cols]
        curr_subset = current_df[eval_cols]

        report = Report([DataDriftPreset()])
        snapshot = report.run(reference_data=ref_subset, current_data=curr_subset)
        report_data = snapshot.dict()

        feature_details: list[FeatureDriftDetail] = []
        pred_drift: FeatureDriftDetail | None = None

        for metric in report_data.get("metrics", []):
            cfg = metric.get("config", {})
            if cfg.get("type") == "evidently:metric_v2:ValueDrift":
                col = cfg.get("column", "unknown")
                method = cfg.get("method", "statistical test")
                threshold = float(cfg.get("threshold", 0.1))
                score = float(metric.get("value", 0.0))
                is_drifted = score > threshold

                detail = FeatureDriftDetail(
                    column=col,
                    method=method,
                    score=round(score, 4),
                    threshold=threshold,
                    is_drifted=is_drifted,
                )

                if col == "fraud_probability":
                    pred_drift = detail
                else:
                    feature_details.append(detail)

        total_features = len(feature_details)
        drifted_count = sum(1 for f in feature_details if f.is_drifted)
        share_drifted = drifted_count / max(total_features, 1)
        dataset_drift = share_drifted >= self.drift_share_threshold

        # Save HTML snapshot for visual dashboard
        html_path = self.reports_dir / "latest_drift_report.html"
        snapshot.save_html(str(html_path))

        drift_report = DriftReport(
            window_size=len(current_df),
            dataset_drift_detected=dataset_drift,
            share_of_drifted_features=round(share_drifted, 4),
            features_analyzed=total_features,
            drifted_features=feature_details,
            prediction_drift=pred_drift,
            html_report_path=str(html_path),
        )

        # Save JSON artifact
        json_path = self.reports_dir / "latest_drift_report.json"
        with open(json_path, "w") as f:
            f.write(drift_report.model_dump_json(indent=2))

        return drift_report

    def ingest_event(self, event_data: dict[str, Any]) -> DriftReport | None:
        """Ingest a single telemetry event into sliding window buffer.

        Evaluates drift whenever the buffer fills to window_size.

        Args:
            event_data: Raw telemetry event payload from Kafka.

        Returns:
            DriftReport if an evaluation was triggered, None otherwise.
        """
        self._window_buffer.append(event_data)

        if len(self._window_buffer) >= self.window_size:
            current_df = pd.DataFrame(list(self._window_buffer))
            return self.analyze_dataframe(current_df)

        return None

    def consume_stream(
        self,
        bootstrap_servers: str = "localhost:9092",
        topic: str = "fraud.telemetry",
        group_id: str = "drift-monitoring-worker",
        max_messages: int | None = None,
    ) -> list[DriftReport]:
        """Consume live stream from Kafka and evaluate drift windows continuously.

        Args:
            bootstrap_servers: Kafka broker addresses.
            topic: Telemetry event topic.
            group_id: Kafka consumer group identifier.
            max_messages: Optional limit of messages to consume (useful for testing).

        Returns:
            List of DriftReports generated during execution.
        """
        conf = {
            "bootstrap.servers": bootstrap_servers,
            "group.id": group_id,
            "auto.offset.reset": "earliest",
            "enable.auto.commit": True,
        }

        consumer = Consumer(conf)
        consumer.subscribe([topic])
        logger.info("Subscribed to %s on %s as %s", topic, bootstrap_servers, group_id)

        reports: list[DriftReport] = []
        messages_read = 0

        try:
            while True:
                if max_messages is not None and messages_read >= max_messages:
                    break

                msg = consumer.poll(timeout=1.0)
                if msg is None:
                    continue
                if msg.error():
                    if msg.error().code() == KafkaError._PARTITION_EOF:
                        continue
                    raise KafkaException(msg.error())

                payload = json.loads(msg.value().decode("utf-8"))
                report = self.ingest_event(payload)
                if report is not None:
                    reports.append(report)
                    logger.warning(
                        "DRIFT EVALUATION: Drift=%s (Share=%.2f%%)",
                        report.dataset_drift_detected,
                        report.share_of_drifted_features * 100,
                    )

                messages_read += 1
        finally:
            consumer.close()

        return reports
