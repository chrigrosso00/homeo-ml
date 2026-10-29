"""Deterministic finite state machine supervisor for self-healing MLOps runtime.

Orchestrates automated closed-loop remediation upon drift detection: root-cause
slice diagnostics, candidate retraining, guardrail verification, and canary routing.
"""

import json
import logging
import re
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
from pydantic import BaseModel, Field

from src.models.train import DEFAULT_DATA_PATH, train_baseline_pipeline
from src.models.train_mobility import (
    DEFAULT_TAXI_DIR,
    train_mobility_recent_window,
)
from src.monitoring.drift_detector import DriftReport

logger = logging.getLogger("supervisor.fsm")

DEFAULT_TRAEFIK_CONFIG_PATH: Path = Path("config/traefik_dynamic.yml")
DEFAULT_SUPERVISOR_DIR: Path = Path("artifacts/supervisor")


class SupervisorState(StrEnum):
    """Possible discrete states of the autonomous self-healing supervisor."""

    IDLE_MONITORING = "IDLE_MONITORING"
    DRIFT_DETECTED = "DRIFT_DETECTED"
    DIAGNOSING_SLICE = "DIAGNOSING_SLICE"
    RETRAINING_CANDIDATE = "RETRAINING_CANDIDATE"
    VALIDATING_GUARDRAILS = "VALIDATING_GUARDRAILS"
    CANARY_ROUTING = "CANARY_ROUTING"
    PROMOTED = "PROMOTED"
    ROLLBACK = "ROLLBACK"


class GuardrailsConfig(BaseModel):
    """Performance and safety guardrails enforced before canary deployment."""

    min_pr_auc: float = Field(default=0.045, description="Minimum acceptable PR-AUC")
    min_recall: float = Field(default=0.85, description="Minimum fraud recall rate")
    min_roc_auc: float = Field(default=0.70, description="Minimum ROC-AUC for mobility")
    min_f1: float = Field(default=0.55, description="Minimum F1 score for mobility")
    max_latency_p95_ms: float = Field(default=20.0, description="Maximum P95 latency in ms")


class AuditRecord(BaseModel):
    """Immutable audit trail record for state transitions."""

    timestamp: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    from_state: str
    to_state: str
    event: str
    details: dict[str, Any] = Field(default_factory=dict)


class TrafficManager:
    """Manages Traefik dynamic Weighted Round-Robin (WRR) traffic configuration."""

    def __init__(self, config_path: Path = DEFAULT_TRAEFIK_CONFIG_PATH) -> None:
        """Initialize traffic manager with dynamic configuration file path.

        Args:
            config_path: Path to Traefik dynamic YAML configuration.
        """
        self.config_path = config_path

    def get_current_weights(self) -> tuple[int, int]:
        """Read current production and canary weights from configuration.

        Returns:
            Tuple of (production_weight, canary_weight).
        """
        if not self.config_path.exists():
            return 100, 0

        content = self.config_path.read_text()
        prod_match = re.search(r"name:\s*production-service\s+weight:\s*(\d+)", content)
        canary_match = re.search(r"name:\s*canary-service\s+weight:\s*(\d+)", content)

        prod_w = int(prod_match.group(1)) if prod_match else 100
        canary_w = int(canary_match.group(1)) if canary_match else 0
        return prod_w, canary_w

    def set_weights(self, production_weight: int, canary_weight: int) -> None:
        """Update traffic weights in Traefik configuration file.

        Traefik automatically hot-reloads file changes in memory without restarting.

        Args:
            production_weight: Percentage of traffic sent to production service (0-100).
            canary_weight: Percentage of traffic sent to canary service (0-100).
        """
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        template = f"""# Dynamic configuration for Traefik WRR Canary Traffic Splitting
http:
  routers:
    fraud-api-router:
      rule: "PathPrefix(`/`)"
      service: fraud-canary-weighted-service
      entryPoints:
        - web

  services:
    fraud-canary-weighted-service:
      weighted:
        services:
          - name: production-service
            weight: {production_weight}
          - name: canary-service
            weight: {canary_weight}

    production-service:
      loadBalancer:
        servers:
          - url: "http://serving-prod:8000"

    canary-service:
      loadBalancer:
        servers:
          - url: "http://serving-canary:8000"
"""
        self.config_path.write_text(template)
        logger.info(
            "Traefik weights updated: Production=%d%%, Canary=%d%%",
            production_weight,
            canary_weight,
        )


class StateMachineSupervisor:
    """Deterministic closed-loop self-healing runtime supervisor."""

    def __init__(
        self,
        guardrails: GuardrailsConfig | None = None,
        traffic_manager: TrafficManager | None = None,
        supervisor_dir: Path = DEFAULT_SUPERVISOR_DIR,
    ) -> None:
        """Initialize supervisor state machine and audit tracking.

        Args:
            guardrails: Performance thresholds for candidate promotion.
            traffic_manager: Controller for ingress canary routing.
            supervisor_dir: Directory where audit trail and incident reports are stored.
        """
        self.guardrails = guardrails or GuardrailsConfig()
        self.traffic_manager = traffic_manager or TrafficManager()
        self.supervisor_dir = supervisor_dir
        self.supervisor_dir.mkdir(parents=True, exist_ok=True)

        self.current_state: SupervisorState = SupervisorState.IDLE_MONITORING
        self.audit_trail: list[AuditRecord] = []
        self._load_audit_trail()

    def _load_audit_trail(self) -> None:
        """Load persistent audit trail from disk if exists."""
        audit_file = self.supervisor_dir / "audit_trail.json"
        if audit_file.exists():
            try:
                data = json.loads(audit_file.read_text())
                self.audit_trail = [AuditRecord(**record) for record in data]
            except Exception as exc:
                logger.warning("Could not parse existing audit trail: %s", exc)

    def _persist_audit_trail(self) -> None:
        """Save audit trail to disk."""
        audit_file = self.supervisor_dir / "audit_trail.json"
        with open(audit_file, "w") as f:
            json.dump([record.model_dump() for record in self.audit_trail], f, indent=2)

    def transition_to(
        self, new_state: SupervisorState, event: str, details: dict[str, Any] | None = None
    ) -> None:
        """Execute a state transition with immutable audit recording.

        Args:
            new_state: Target SupervisorState.
            event: Event trigger name.
            details: Contextual metadata explaining the transition.
        """
        old_state = self.current_state
        self.current_state = new_state
        record = AuditRecord(
            from_state=old_state.value,
            to_state=new_state.value,
            event=event,
            details=details or {},
        )
        self.audit_trail.append(record)
        self._persist_audit_trail()
        logger.info(
            "FSM TRANSITION: [%s] -> [%s] on event '%s'",
            old_state.value,
            new_state.value,
            event,
        )

    def diagnose_slice(self, report: DriftReport) -> dict[str, Any]:
        """Perform root-cause diagnostics to identify drifted feature slices.

        Args:
            report: Evidently DriftReport triggering the remediation.

        Returns:
            Dictionary with diagnosed root-cause columns and remediation strategy.
        """
        drifted_cols = [f.column for f in report.drifted_features if f.is_drifted]
        slice_diagnostic = {
            "drifted_features": drifted_cols,
            "drift_share": report.share_of_drifted_features,
            "prediction_drift": (
                report.prediction_drift.model_dump() if report.prediction_drift else None
            ),
            "remediation_strategy": "CHRONOLOGICAL_ADAPTATION",
        }

        # Check for specific known attack or drift patterns
        if "time_since_signup_sec" in drifted_cols and "browser" in drifted_cols:
            slice_diagnostic["attack_pattern"] = "BOTNET_INSTANT_CHECKOUT"
        elif "speed_mph" in drifted_cols or "trip_distance" in drifted_cols:
            slice_diagnostic["attack_pattern"] = "LOCKDOWN_MOBILITY_DRIFT"
        elif "purchase_value" in drifted_cols:
            slice_diagnostic["attack_pattern"] = "MONETARY_INFLATION_SHIFT"
        else:
            slice_diagnostic["attack_pattern"] = "GENERIC_COVARIATE_SHIFT"

        return slice_diagnostic

    def validate_candidate(
        self, candidate_metadata: dict[str, Any], candidate_onnx_path: Path
    ) -> tuple[bool, str]:
        """Verify candidate model against strict performance and latency guardrails.

        Args:
            candidate_metadata: Metadata dictionary produced by training pipeline.
            candidate_onnx_path: Path to serialized ONNX candidate artifact.

        Returns:
            Tuple of (passed: bool, reason: str).
        """
        domain = candidate_metadata.get("domain", "fraud")
        metrics = candidate_metadata.get("metrics", {})

        if domain in ("mobility", "urban_mobility"):
            roc_auc = metrics.get("roc_auc", 0.0)
            if roc_auc < self.guardrails.min_roc_auc:
                msg = (
                    f"Candidate ROC-AUC ({roc_auc:.4f}) below guardrail "
                    f"({self.guardrails.min_roc_auc})"
                )
                return False, msg

            f1 = metrics.get("optimal_f1", metrics.get("f1_score", 0.0))
            if f1 < self.guardrails.min_f1:
                msg = f"Candidate F1 ({f1:.4f}) below guardrail ({self.guardrails.min_f1})"
                return False, msg
        else:
            pr_auc = metrics.get("pr_auc", 0.0)
            recall = metrics.get("optimal_recall", 0.0)

            # 1. PR-AUC guardrail
            if pr_auc < self.guardrails.min_pr_auc:
                msg = (
                    f"Candidate PR-AUC ({pr_auc:.4f}) below guardrail "
                    f"({self.guardrails.min_pr_auc})"
                )
                return False, msg

            # 2. Recall guardrail
            if recall < self.guardrails.min_recall:
                msg = (
                    f"Candidate Recall ({recall:.4f}) below guardrail "
                    f"({self.guardrails.min_recall})"
                )
                return False, msg

        # 3. ONNX latency benchmark guardrail
        try:
            sess = ort.InferenceSession(str(candidate_onnx_path))
            input_name = sess.get_inputs()[0].name
            n_feat = candidate_metadata.get("n_features", 41)
            sample_input = np.random.randn(1, n_feat).astype(np.float32)

            latencies: list[float] = []
            for _ in range(50):
                t0 = datetime.now()
                sess.run(None, {input_name: sample_input})
                latencies.append((datetime.now() - t0).total_seconds() * 1000.0)

            p95_lat = float(np.percentile(latencies, 95))
            if p95_lat > self.guardrails.max_latency_p95_ms:
                limit = self.guardrails.max_latency_p95_ms
                return False, f"Candidate P95 latency ({p95_lat:.2f}ms) exceeds SLA ({limit}ms)"

        except Exception as exc:
            return False, f"Candidate ONNX validation crashed: {exc}"

        return True, "All guardrails passed successfully."

    def evaluate_canary_window(
        self,
        production_metrics: dict[str, float],
        canary_metrics: dict[str, float],
        metric_key: str = "f1_score",
        min_improvement_margin: float = 0.0,
    ) -> tuple[bool, str]:
        """Statistically compare real traffic performance between production and canary services.

        Args:
            production_metrics: Metrics computed from live traffic served by production.
            canary_metrics: Metrics computed from live traffic served by canary.
            metric_key: Target evaluation metric name (e.g. 'f1_score' or 'roc_auc').
            min_improvement_margin: Minimum margin by which canary must exceed production.

        Returns:
            Tuple of (should_promote: bool, reason: str).
        """
        prod_val = production_metrics.get(metric_key, 0.0)
        canary_val = canary_metrics.get(metric_key, 0.0)

        diff = canary_val - prod_val
        if diff >= min_improvement_margin:
            msg = (
                f"Canary {metric_key} ({canary_val:.4f}) outperforms "
                f"Production ({prod_val:.4f}) by {diff:+.4f}. Promotion approved."
            )
            return True, msg

        msg = (
            f"Canary {metric_key} ({canary_val:.4f}) did not beat "
            f"Production ({prod_val:.4f}) by margin {min_improvement_margin}. Rollback triggered."
        )
        return False, msg

    def run_healing_cycle(
        self,
        report: DriftReport,
        data_path: Path = DEFAULT_DATA_PATH,
        canary_artifacts_dir: Path = Path("artifacts/canary"),
        domain: str = "fraud",
        canary_eval_metrics: tuple[dict[str, float], dict[str, float]] | None = None,
    ) -> SupervisorState:
        """Execute autonomous end-to-end self-healing cycle.

        Supports multi-domain self-healing (fraud or urban mobility) and real
        metric-driven canary evaluation.

        Args:
            report: Incoming DriftReport.
            data_path: Dataset path for retraining.
            canary_artifacts_dir: Destination for canary model artifacts.
            domain: Domain name ('fraud' or 'mobility').
            canary_eval_metrics: Optional tuple of (prod_metrics, canary_metrics) from live traffic.

        Returns:
            Final SupervisorState reached by the FSM.
        """
        if not report.dataset_drift_detected:
            logger.info("Drift within operational bounds. Supervisor remains IDLE.")
            return self.current_state

        # 1. DRIFT_DETECTED
        self.transition_to(
            SupervisorState.DRIFT_DETECTED,
            event="DRIFT_THRESHOLD_EXCEEDED",
            details={
                "domain": domain,
                "share_drifted": report.share_of_drifted_features,
            },
        )

        # 2. DIAGNOSING_SLICE
        slice_info = self.diagnose_slice(report)
        self.transition_to(
            SupervisorState.DIAGNOSING_SLICE,
            event="ROOT_CAUSE_ISOLATED",
            details=slice_info,
        )

        # 3. RETRAINING_CANDIDATE
        target_version = f"v1.1.0-canary-{datetime.now(UTC).strftime('%Y%m%d%H%M')}"
        self.transition_to(
            SupervisorState.RETRAINING_CANDIDATE,
            event="START_CANDIDATE_TRAINING",
            details={"target_version": target_version, "domain": domain},
        )
        canary_artifacts_dir.mkdir(parents=True, exist_ok=True)

        if domain in ("mobility", "urban_mobility"):
            candidate_meta = train_mobility_recent_window(
                data_dir=DEFAULT_TAXI_DIR,
                artifacts_dir=canary_artifacts_dir,
                model_version=target_version,
            )
            candidate_onnx = canary_artifacts_dir / "mobility_model.onnx"
        else:
            candidate_meta = train_baseline_pipeline(
                data_path=data_path,
                artifacts_dir=canary_artifacts_dir,
                model_version=target_version,
            )
            candidate_onnx = canary_artifacts_dir / "fraud_model.onnx"

        # 4. VALIDATING_GUARDRAILS
        self.transition_to(
            SupervisorState.VALIDATING_GUARDRAILS,
            event="EVALUATING_SAFETY_GUARDRAILS",
            details={"candidate_version": target_version},
        )
        passed, reason = self.validate_candidate(candidate_meta, candidate_onnx)

        if not passed:
            self.transition_to(
                SupervisorState.ROLLBACK,
                event="GUARDRAIL_VIOLATION",
                details={"reason": reason},
            )
            self.traffic_manager.set_weights(production_weight=100, canary_weight=0)
            return self.current_state

        # 5. CANARY_ROUTING (95% Production / 5% Canary)
        self.transition_to(
            SupervisorState.CANARY_ROUTING,
            event="DEPLOY_CANARY_TRAFFIC",
            details={"traffic_split": "95/5", "reason": reason},
        )
        self.traffic_manager.set_weights(production_weight=95, canary_weight=5)

        # 6. EVALUATE LIVE CANARY TRAFFIC (Real metric comparison vs Simulated fallback)
        if canary_eval_metrics is not None:
            prod_m, canary_m = canary_eval_metrics
            should_promote, eval_reason = self.evaluate_canary_window(
                production_metrics=prod_m,
                canary_metrics=canary_m,
                metric_key=(
                    "f1_score" if domain in ("mobility", "urban_mobility") else "optimal_f1"
                ),
            )
            if not should_promote:
                self.transition_to(
                    SupervisorState.ROLLBACK,
                    event="CANARY_PERFORMANCE_REGRESSION",
                    details={"reason": eval_reason},
                )
                self.traffic_manager.set_weights(production_weight=100, canary_weight=0)
                return self.current_state
        else:
            eval_reason = "Simulated healthy canary traffic observation window."

        # 7. PROMOTED -> 100% Canary
        self.transition_to(
            SupervisorState.PROMOTED,
            event="PROMOTE_CANARY_TO_PRODUCTION",
            details={
                "promoted_version": target_version,
                "traffic_split": "0/100",
                "evaluation": eval_reason,
            },
        )
        self.traffic_manager.set_weights(production_weight=0, canary_weight=100)

        # 8. Return to monitoring with new baseline
        self.transition_to(
            SupervisorState.IDLE_MONITORING,
            event="HEALING_CYCLE_COMPLETE",
            details={"active_version": target_version},
        )

        return self.current_state
