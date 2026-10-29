"""Unit and integration tests for StateMachineSupervisor and Traefik traffic management."""

from pathlib import Path

import pytest

from src.monitoring.drift_detector import DriftReport, FeatureDriftDetail
from src.supervisor.state_machine import (
    GuardrailsConfig,
    StateMachineSupervisor,
    SupervisorState,
    TrafficManager,
)


@pytest.fixture
def traffic_manager(tmp_path: Path) -> TrafficManager:
    """Provide TrafficManager configured with temporary traefik_dynamic.yml."""
    config_file = tmp_path / "traefik_dynamic.yml"
    manager = TrafficManager(config_path=config_file)
    manager.set_weights(production_weight=100, canary_weight=0)
    return manager


@pytest.fixture
def supervisor(tmp_path: Path, traffic_manager: TrafficManager) -> StateMachineSupervisor:
    """Provide StateMachineSupervisor with isolated test audit trail."""
    return StateMachineSupervisor(
        traffic_manager=traffic_manager,
        supervisor_dir=tmp_path / "supervisor",
    )


def test_traffic_manager_weight_updates(traffic_manager: TrafficManager) -> None:
    """Verify Traefik configuration file updates and weight extraction."""
    prod_w, canary_w = traffic_manager.get_current_weights()
    assert prod_w == 100
    assert canary_w == 0

    traffic_manager.set_weights(production_weight=95, canary_weight=5)
    prod_w, canary_w = traffic_manager.get_current_weights()
    assert prod_w == 95
    assert canary_w == 5

    traffic_manager.set_weights(production_weight=0, canary_weight=100)
    prod_w, canary_w = traffic_manager.get_current_weights()
    assert prod_w == 0
    assert canary_w == 100


def test_diagnose_slice_botnet_pattern(supervisor: StateMachineSupervisor) -> None:
    """Verify root cause diagnosis pinpoints botnet instant checkout pattern."""
    report = DriftReport(
        window_size=300,
        dataset_drift_detected=True,
        share_of_drifted_features=0.6,
        features_analyzed=7,
        drifted_features=[
            FeatureDriftDetail(
                column="time_since_signup_sec",
                method="Wasserstein distance (normed)",
                score=0.45,
                threshold=0.1,
                is_drifted=True,
            ),
            FeatureDriftDetail(
                column="browser",
                method="Jensen-Shannon distance",
                score=0.35,
                threshold=0.1,
                is_drifted=True,
            ),
            FeatureDriftDetail(
                column="sex",
                method="Jensen-Shannon distance",
                score=0.01,
                threshold=0.1,
                is_drifted=False,
            ),
        ],
    )

    diagnostic = supervisor.diagnose_slice(report)
    assert "time_since_signup_sec" in diagnostic["drifted_features"]
    assert "browser" in diagnostic["drifted_features"]
    assert diagnostic["attack_pattern"] == "BOTNET_INSTANT_CHECKOUT"


def test_diagnose_slice_mobility_lockdown_pattern(supervisor: StateMachineSupervisor) -> None:
    """Verify root cause diagnosis pinpoints lockdown mobility drift pattern."""
    report = DriftReport(
        window_size=300,
        dataset_drift_detected=True,
        share_of_drifted_features=0.6,
        features_analyzed=5,
        drifted_features=[
            FeatureDriftDetail(
                column="speed_mph",
                method="Wasserstein distance (normed)",
                score=0.48,
                threshold=0.1,
                is_drifted=True,
            ),
            FeatureDriftDetail(
                column="trip_distance",
                method="Wasserstein distance (normed)",
                score=0.32,
                threshold=0.1,
                is_drifted=True,
            ),
        ],
    )

    diagnostic = supervisor.diagnose_slice(report)
    assert "speed_mph" in diagnostic["drifted_features"]
    assert diagnostic["attack_pattern"] == "LOCKDOWN_MOBILITY_DRIFT"


def test_mobility_candidate_guardrails_validation(
    supervisor: StateMachineSupervisor, tmp_path: Path
) -> None:
    """Verify candidate validation for mobility domain with ROC-AUC and F1."""
    onnx_file = tmp_path / "mobility_dummy.onnx"
    # Create empty dummy file
    onnx_file.touch()

    # Case 1: Pass
    pass_meta = {
        "domain": "mobility",
        "metrics": {"roc_auc": 0.81, "optimal_f1": 0.58},
        "n_features": 9,
    }
    # Mock ort.InferenceSession within validate_candidate
    import unittest.mock as mock

    with mock.patch("onnxruntime.InferenceSession"):
        passed, reason = supervisor.validate_candidate(pass_meta, onnx_file)
        assert passed is True
        assert "passed" in reason

    # Case 2: Fail ROC-AUC
    fail_meta = {
        "domain": "mobility",
        "metrics": {"roc_auc": 0.65, "optimal_f1": 0.58},
        "n_features": 9,
    }
    passed, reason = supervisor.validate_candidate(fail_meta, onnx_file)
    assert passed is False
    assert "ROC-AUC" in reason


def test_guardrails_validation_rejection(
    supervisor: StateMachineSupervisor, tmp_path: Path
) -> None:
    """Verify candidate failing PR-AUC or Recall is rejected by the validation gate."""
    failing_metadata = {
        "metrics": {
            "pr_auc": 0.01,  # Below min 0.045
            "optimal_recall": 0.50,  # Below min 0.85
        },
        "n_features": 41,
    }

    dummy_onnx = tmp_path / "dummy.onnx"
    dummy_onnx.touch()

    passed, reason = supervisor.validate_candidate(failing_metadata, dummy_onnx)
    assert passed is False
    assert "PR-AUC" in reason


def test_full_autonomous_self_healing_cycle(
    supervisor: StateMachineSupervisor, tmp_path: Path
) -> None:
    """Verify complete closed-loop cycle: drift -> retrain -> canary -> promotion."""
    drift_report = DriftReport(
        window_size=300,
        dataset_drift_detected=True,
        share_of_drifted_features=0.75,
        features_analyzed=7,
        drifted_features=[
            FeatureDriftDetail(
                column="time_since_signup_sec",
                method="Wasserstein distance (normed)",
                score=0.55,
                threshold=0.1,
                is_drifted=True,
            ),
            FeatureDriftDetail(
                column="browser",
                method="Jensen-Shannon distance",
                score=0.38,
                threshold=0.1,
                is_drifted=True,
            ),
            FeatureDriftDetail(
                column="purchase_value",
                method="Wasserstein distance (normed)",
                score=0.32,
                threshold=0.1,
                is_drifted=True,
            ),
        ],
    )

    canary_dir = tmp_path / "canary"
    final_state = supervisor.run_healing_cycle(
        report=drift_report,
        canary_artifacts_dir=canary_dir,
    )

    # After full successful cycle, supervisor returns to IDLE_MONITORING with updated model
    assert final_state == SupervisorState.IDLE_MONITORING

    # Verify audit trail contains all expected transitions
    states_visited = [record.to_state for record in supervisor.audit_trail]
    assert SupervisorState.DRIFT_DETECTED.value in states_visited
    assert SupervisorState.DIAGNOSING_SLICE.value in states_visited
    assert SupervisorState.RETRAINING_CANDIDATE.value in states_visited
    assert SupervisorState.VALIDATING_GUARDRAILS.value in states_visited
    assert SupervisorState.CANARY_ROUTING.value in states_visited
    assert SupervisorState.PROMOTED.value in states_visited

    # Verify final traffic weights: promoted to 100%
    prod_w, canary_w = supervisor.traffic_manager.get_current_weights()
    assert canary_w == 100
    assert prod_w == 0

    # Verify audit trail was persisted to disk
    audit_file = supervisor.supervisor_dir / "audit_trail.json"
    assert audit_file.exists()


def test_guardrails_failure_triggers_rollback(
    tmp_path: Path, traffic_manager: TrafficManager
) -> None:
    """Verify that if candidate model fails guardrails, supervisor executes rollback."""
    # Custom guardrails with unattainable PR-AUC
    strict_guardrails = GuardrailsConfig(min_pr_auc=0.99)
    strict_supervisor = StateMachineSupervisor(
        guardrails=strict_guardrails,
        traffic_manager=traffic_manager,
        supervisor_dir=tmp_path / "supervisor_strict",
    )

    drift_report = DriftReport(
        window_size=300,
        dataset_drift_detected=True,
        share_of_drifted_features=0.6,
        features_analyzed=7,
        drifted_features=[
            FeatureDriftDetail(
                column="time_since_signup_sec",
                method="Wasserstein distance (normed)",
                score=0.45,
                threshold=0.1,
                is_drifted=True,
            )
        ],
    )

    final_state = strict_supervisor.run_healing_cycle(
        report=drift_report,
        canary_artifacts_dir=tmp_path / "canary_fail",
    )

    assert final_state == SupervisorState.ROLLBACK

    # Verify traffic was rolled back to 100% production
    prod_w, canary_w = traffic_manager.get_current_weights()
    assert prod_w == 100
    assert canary_w == 0


def test_canary_evaluation_promotion_and_rollback(
    supervisor: StateMachineSupervisor,
) -> None:
    """Verify evaluate_canary_window approves promotion on win and aborts on regression."""
    # Case 1: Canary outperforms Production
    prod_metrics = {"f1_score": 0.50, "roc_auc": 0.72}
    canary_metrics = {"f1_score": 0.86, "roc_auc": 0.88}

    should_promote, reason = supervisor.evaluate_canary_window(
        production_metrics=prod_metrics,
        canary_metrics=canary_metrics,
        metric_key="f1_score",
    )
    assert should_promote is True
    assert "Promotion approved" in reason

    # Case 2: Canary regresses compared to Production
    regressed_canary = {"f1_score": 0.45, "roc_auc": 0.68}
    should_promote, reason = supervisor.evaluate_canary_window(
        production_metrics=prod_metrics,
        canary_metrics=regressed_canary,
        metric_key="f1_score",
    )
    assert should_promote is False
    assert "Rollback triggered" in reason
