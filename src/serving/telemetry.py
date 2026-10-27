"""Asynchronous event telemetry producer for Kafka / Redpanda.

Dispatches inference events and feature metadata to a streaming buffer
using non-blocking fire-and-forget message buffering.
"""

import logging
import os
from typing import Any

from confluent_kafka import KafkaError, KafkaException, Producer

from src.data.schemas import TelemetryEvent

logger = logging.getLogger(__name__)

DEFAULT_BOOTSTRAP_SERVERS: str = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
DEFAULT_TELEMETRY_TOPIC: str = os.getenv("KAFKA_TELEMETRY_TOPIC", "fraud.telemetry")


class KafkaTelemetryProducer:
    """Non-blocking telemetry dispatcher backed by librdkafka."""

    def __init__(
        self,
        bootstrap_servers: str = DEFAULT_BOOTSTRAP_SERVERS,
        topic: str = DEFAULT_TELEMETRY_TOPIC,
    ) -> None:
        """Initialize Kafka producer client.

        Args:
            bootstrap_servers: Comma-separated broker host:port addresses.
            topic: Destination Kafka topic name.
        """
        self.topic = topic
        self.bootstrap_servers = bootstrap_servers
        self._producer: Producer | None = None
        self._is_connected: bool = False

        self._init_producer()

    def _init_producer(self) -> None:
        """Attempt to instantiate confluent_kafka Producer."""
        conf = {
            "bootstrap.servers": self.bootstrap_servers,
            "client.id": "fraud-serving-telemetry",
            "queue.buffering.max.messages": 100000,
            "queue.buffering.max.ms": 20,
            "acks": 1,
        }
        try:
            self._producer = Producer(conf)
            self._is_connected = True
            logger.info("KafkaTelemetryProducer initialized on %s", self.bootstrap_servers)
        except (KafkaException, Exception) as exc:
            self._is_connected = False
            logger.warning("Kafka producer unavailable (%s). Telemetry disabled.", exc)

    @staticmethod
    def _delivery_report(err: KafkaError | None, msg: Any) -> None:
        """Callback invoked once message is delivered or permanently failed.

        Args:
            err: KafkaError instance or None if delivered successfully.
            msg: Delivered Message instance.
        """
        if err is not None:
            logger.warning("Telemetry delivery failure: %s", err)

    def dispatch(self, event: TelemetryEvent) -> bool:
        """Dispatch a telemetry event to Redpanda asynchronously.

        Execution completes in microseconds by appending to librdkafka memory queue.

        Args:
            event: Validated TelemetryEvent instance.

        Returns:
            True if queued successfully, False if failed or telemetry disabled.
        """
        if not self._is_connected or self._producer is None:
            return False

        try:
            payload = event.model_dump_json().encode("utf-8")
            self._producer.produce(
                topic=self.topic,
                key=event.transaction_id.encode("utf-8"),
                value=payload,
                on_delivery=self._delivery_report,
            )
            # poll(0) services delivery callbacks without blocking
            self._producer.poll(0)
            return True
        except BufferError:
            logger.warning(
                "Kafka local buffer full! Dropping telemetry event %s", event.transaction_id
            )
            return False
        except Exception as exc:
            logger.error("Failed to queue telemetry event: %s", exc)
            return False

    def flush(self, timeout: float = 3.0) -> int:
        """Wait for pending messages to be delivered on service shutdown.

        Args:
            timeout: Maximum seconds to wait.

        Returns:
            Number of un-delivered messages remaining in queue.
        """
        if self._producer is not None:
            return self._producer.flush(timeout=timeout)
        return 0
