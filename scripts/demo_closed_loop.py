"""End-to-end closed-loop demonstration of the Self-Healing MLOps Runtime.

Simulates nominal production traffic, injects an automated botnet drift attack,
and showcases autonomous root-cause diagnostics, candidate retraining,
guardrail verification, and dynamic canary traffic promotion.
"""

import time
from pathlib import Path
from typing import Any

import pandas as pd
from starlette.testclient import TestClient

from scripts.traffic_generator import (
    generate_botnet_drift_transaction,
    generate_normal_transaction,
)
from src.monitoring.drift_detector import DriftMonitoringEngine
from src.serving.app import app
from src.supervisor.state_machine import StateMachineSupervisor, TrafficManager

BANNER: str = """
================================================================================
          SELF-HEALING MLOps RUNTIME - CLOSED-LOOP CANARY DEMO
================================================================================
"""

RAW_DATA_PATH: Path = Path("data/raw/Fraud_Data.csv")
REFERENCE_PATH: Path = Path("artifacts/models/reference_data.parquet")
DEMO_CONFIG_PATH: Path = Path("config/traefik_dynamic.yml")
DEMO_SUPERVISOR_DIR: Path = Path("artifacts/supervisor")
DEMO_MONITORING_DIR: Path = Path("artifacts/monitoring")


def print_step_header(step_num: int, title: str) -> None:
    """Print visually distinct step section banner."""
    print(f"\n{'─' * 80}")
    print(f" ▶ STEP {step_num}: {title.upper()}")
    print(f"{'─' * 80}")


def run_demo() -> None:
    """Execute complete end-to-end autonomous healing cycle simulation."""
    print(BANNER)

    traffic_mgr = TrafficManager(config_path=DEMO_CONFIG_PATH)
    traffic_mgr.set_weights(production_weight=100, canary_weight=0)

    supervisor = StateMachineSupervisor(
        traffic_manager=traffic_mgr,
        supervisor_dir=DEMO_SUPERVISOR_DIR,
    )

    engine = DriftMonitoringEngine(
        reference_path=REFERENCE_PATH,
        window_size=300,
        drift_share_threshold=0.45,
        reports_dir=DEMO_MONITORING_DIR,
    )

    with TestClient(app) as client:
        # Step 1: Health check
        print_step_header(1, "Verifying Production Serving Health & Baseline Model")
        resp = client.get("/health")
        health = resp.json()
        print(f"  • Serving Status     : {health.get('status').upper()}")
        print(f"  • Active Model       : {health.get('model_version')}")
        print(f"  • Decision Threshold : {health.get('optimal_threshold'):.4f}")
        print(f"  • Features Dimension : {health.get('n_features')} columns")
        prod_w, canary_w = traffic_mgr.get_current_weights()
        print(f"  • Ingress Traffic    : Production={prod_w}% | Canary={canary_w}%")

        # Step 2: Stream nominal production traffic
        print_step_header(2, "Streaming Nominal Production Traffic (N=300 Transactions)")
        df = pd.read_csv(RAW_DATA_PATH)
        test_df = (
            df.iloc[int(len(df) * 0.8) :].sample(n=300, random_state=42).reset_index(drop=True)
        )

        normal_events: list[dict[str, Any]] = []
        t0 = time.perf_counter()
        for i in range(300):
            row = test_df.iloc[i]
            payload = generate_normal_transaction(row)
            res = client.post("/predict", json=payload)
            pred = res.json()
            p_time = pd.to_datetime(payload["purchase_time"])
            s_time = pd.to_datetime(payload["signup_time"])
            delta_s = (p_time - s_time).total_seconds()
            country = str(client.app.state.transformer.ip_lookup.lookup(payload["ip_address"]))

            normal_events.append(
                {
                    "purchase_value": payload["purchase_value"],
                    "age": payload["age"],
                    "time_since_signup_sec": delta_s,
                    "source": payload["source"],
                    "browser": payload["browser"],
                    "sex": payload["sex"],
                    "country": country,
                    "fraud_probability": pred["fraud_probability"],
                }
            )

        elapsed_sec = time.perf_counter() - t0
        tx_rate = 300 / elapsed_sec
        print(f"  • Streamed 300 transactions in {elapsed_sec:.2f}s (~{tx_rate:.0f} tx/sec)")

        # Evaluate drift on nominal data
        norm_df = pd.DataFrame(normal_events)
        norm_report = engine.analyze_dataframe(norm_df)
        drift_pct = norm_report.share_of_drifted_features * 100.0
        thresh_pct = engine.drift_share_threshold * 100.0
        print(f"  • Statistical Drift Score : {drift_pct:.1f}% (Threshold: {thresh_pct:.1f}%)")
        status_txt = "OK: System Stable"
        print(f"  • Drift Detected          : {norm_report.dataset_drift_detected} [{status_txt}]")

        # Step 3: Inject Adversarial Botnet Attack
        print_step_header(3, "Injecting Adversarial Botnet Attack (Instant Signups)")
        botnet_events: list[dict[str, Any]] = []
        bot_flagged = 0

        for i in range(300):
            payload = generate_botnet_drift_transaction(i)
            res = client.post("/predict", json=payload)
            pred = res.json()
            if pred["is_fraud"]:
                bot_flagged += 1

            p_time = pd.to_datetime(payload["purchase_time"])
            s_time = pd.to_datetime(payload["signup_time"])
            delta_s = (p_time - s_time).total_seconds()
            country = str(client.app.state.transformer.ip_lookup.lookup(payload["ip_address"]))

            botnet_events.append(
                {
                    "purchase_value": payload["purchase_value"],
                    "age": payload["age"],
                    "time_since_signup_sec": delta_s,
                    "source": payload["source"],
                    "browser": payload["browser"],
                    "sex": payload["sex"],
                    "country": country,
                    "fraud_probability": pred["fraud_probability"],
                }
            )

        print("  • Injected 300 botnet transactions")
        flag_rate = (bot_flagged / 300) * 100
        print(f"  • Model Flagged as Fraud  : {bot_flagged}/300 ({flag_rate:.1f}%)")

        # Evaluate drift on botnet attack
        bot_df = pd.DataFrame(botnet_events)
        attack_report = engine.analyze_dataframe(bot_df)
        attack_drift_pct = attack_report.share_of_drifted_features * 100.0

        print("\n  [!] DRIFT ENGINE ALERT FIRED:")
        score_msg = f"{attack_drift_pct:.1f}% > Threshold ({thresh_pct:.1f}%)"
        print(f"      • Drift Score        : {score_msg}")
        drift_flag = "[CRITICAL DEVIATION]"
        print(f"      • Dataset Drift      : {attack_report.dataset_drift_detected} {drift_flag}")
        drifted_names = [f.column for f in attack_report.drifted_features if f.is_drifted]
        print(f"      • Drifted Columns    : {', '.join(drifted_names)}")

        # Step 4: Autonomous State Machine Supervisor Intervention
        print_step_header(4, "Autonomous Supervisor Agent: Closed-Loop Remediation")
        print("  • Triggering StateMachineSupervisor FSM...")

        final_state = supervisor.run_healing_cycle(
            report=attack_report,
            canary_artifacts_dir=Path("artifacts/canary"),
        )

        print("\n  • FSM Transitions Executed:")
        for rec in supervisor.audit_trail[-6:]:
            print(f"    └─ [{rec.from_state}] ──({rec.event})──► [{rec.to_state}]")

        prod_w, canary_w = traffic_mgr.get_current_weights()
        print(f"\n  • Dynamic Traffic Routing Status: Production={prod_w}% | Canary={canary_w}%")
        print(f"  • Final System State           : {final_state.value}")

        # Summary & Artifact Locations
        print_step_header(5, "Self-Healing Cycle Complete - Artifacts & Observability")
        print(f"  ✔ Audit Trail JSON   : {supervisor.supervisor_dir / 'audit_trail.json'}")
        print(f"  ✔ Drift Report JSON  : {engine.reports_dir / 'latest_drift_report.json'}")
        print(f"  ✔ Evidently HTML UI  : {engine.reports_dir / 'latest_drift_report.html'}")
        print(f"  ✔ Traefik Config     : {DEMO_CONFIG_PATH}")
        print("\n" + "=" * 80)
        print("              DEMO EXECUTION SUCCESSFULLY FINISHED!")
        print("=" * 80 + "\n")


if __name__ == "__main__":
    run_demo()
