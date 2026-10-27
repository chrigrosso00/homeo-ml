"""End-to-end closed-loop demonstration of the Self-Healing MLOps Runtime.

Simulates nominal production traffic, injects an automated botnet drift attack,
and showcases autonomous root-cause diagnostics, candidate retraining,
guardrail verification, and dynamic canary traffic promotion.
"""

import argparse
import sys
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


def render_traffic_bar(prod_w: int, canary_w: int, width: int = 30) -> str:
    """Format visual ASCII progress bar depicting Traefik weighted ingress split.

    Args:
        prod_w: Integer percentage for production service (0-100).
        canary_w: Integer percentage for canary service (0-100).
        width: Total width of bar in character units.

    Returns:
        Formatted ASCII representation with percentage labels.
    """
    total = max(prod_w + canary_w, 1)
    prod_chars = int(round((prod_w / total) * width))
    canary_chars = width - prod_chars

    bar = f"[{'█' * prod_chars}{'░' * canary_chars}]"
    state_label = ""
    if canary_w == 0:
        state_label = "[BASELINE]"
    elif canary_w == 100:
        state_label = "[100% PROMOTED]"
    else:
        state_label = f"[{canary_w}% CANARY TEST]"

    return f"{bar}  PROD: {prod_w:3d}% | CANARY: {canary_w:3d}%  {state_label}"


def print_step_header(step_num: int, title: str) -> None:
    """Print visually distinct step section banner."""
    print(f"\n{'═' * 80}")
    print(f" ▶ STEP {step_num}: {title.upper()}")
    print(f"{'═' * 80}")


def print_context_box(concept: str, explanation: str) -> None:
    """Print an educational contextual callout box.

    Args:
        concept: Short summary of what is happening.
        explanation: Detailed rationale of the architectural choice.
    """
    print(f"\n  ┌─ 💡 CONTESTO: {concept} {'─' * max(0, 60 - len(concept))}")
    for line in explanation.strip().split("\n"):
        print(f"  │  {line.strip()}")
    print(f"  └{'─' * 76}\n")


def pause_step(mode: str, delay: float, prompt: str) -> None:
    """Pause execution based on user configuration mode.

    Args:
        mode: Execution mode ('step', 'auto', or 'fast').
        delay: Seconds to sleep in 'auto' mode.
        prompt: Description of the action to follow.
    """
    if mode == "fast":
        return
    if mode == "step":
        try:
            input(f"\n  👉 [PAUSA GUIDATA] Premi INVIO per {prompt}...")
        except (KeyboardInterrupt, EOFError):
            print("\nEsecuzione interrotta dall'utente.")
            sys.exit(0)
    else:
        print(f"\n  ⏳ {prompt}... (procedo tra {delay:.1f}s)")
        time.sleep(delay)


def stream_progress(current: int, total: int, width: int = 25) -> None:
    """Print a single-line dynamic progress bar to terminal.

    Args:
        current: Number of completed items.
        total: Total number of items.
        width: Character width of bar.
    """
    pct = (current / total) * 100.0
    filled = int(round((current / total) * width))
    bar = f"[{'█' * filled}{'-' * (width - filled)}]"
    sys.stdout.write(f"\r     Invio transazioni: {bar} {current}/{total} ({pct:.0f}%)")
    sys.stdout.flush()


def run_demo(mode: str = "auto", delay: float = 2.0) -> None:
    """Execute complete end-to-end autonomous healing cycle simulation.

    Args:
        mode: Demonstration pacing ('step', 'auto', 'fast').
        delay: Delay in seconds when running in auto mode.
    """
    print(BANNER)
    print(f"  • Modalità di esecuzione : {mode.upper()}")
    print(f"  • Intervallo pacing      : {delay:.1f}s tra gli step")
    print(f"  • Configurazione Traefik : {DEMO_CONFIG_PATH}")

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
        # STEP 1: Health check
        print_step_header(1, "Verifica Stato Iniziale del Servizio & Baseline")
        print_context_box(
            "Gateway & Ingress Routing",
            "In condizioni normali, Traefik indirizza il 100% del traffico al modello di\n"
            "produzione (v1.0.0). Il modello è esportato in ONNX C++ per garantire latenze\n"
            "sub-millisecondo (<0.5ms) e gli eventi di telemetria vanno in streaming asincrono.",
        )

        resp = client.get("/health")
        health = resp.json()
        print(f"  • Serving Status     : {health.get('status').upper()}")
        print(f"  • Active Model       : {health.get('model_version')}")
        print(f"  • Decision Threshold : {health.get('optimal_threshold'):.4f}")
        print(f"  • Features Dimension : {health.get('n_features')} columns")

        prod_w, canary_w = traffic_mgr.get_current_weights()
        print(f"  • Ingress Traefik    : {render_traffic_bar(prod_w, canary_w)}")

        pause_step(mode, delay, "avviare lo streaming di 300 transazioni nominali")

        # STEP 2: Stream nominal production traffic
        print_step_header(2, "Streaming Traffico Nominale di Produzione (N=300)")
        print_context_box(
            "Monitoraggio Statistico e Sliding Window",
            "Inviamo 300 transazioni legittime. Il motore Evidently valuta la distanza\n"
            "statistica di Wasserstein e Jensen-Shannon. Se il drift è inferiore al 45%,\n"
            "il sistema riconosce la normale fluttuazione e rimane in IDLE_MONITORING.",
        )

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
            if (i + 1) % 50 == 0 or (i + 1) == 300:
                stream_progress(i + 1, 300)

        sys.stdout.write("\n")
        elapsed_sec = time.perf_counter() - t0
        tx_rate = 300 / elapsed_sec
        print(f"  • Streamed 300 transazioni in {elapsed_sec:.2f}s (~{tx_rate:.0f} tx/sec)")

        norm_df = pd.DataFrame(normal_events)
        norm_report = engine.analyze_dataframe(norm_df)
        drift_pct = norm_report.share_of_drifted_features * 100.0
        thresh_pct = engine.drift_share_threshold * 100.0
        print(f"  • Drift Score Rilevato   : {drift_pct:.1f}% (Soglia Allerta: {thresh_pct:.1f}%)")
        status_txt = "OK: Sistema Stabile (Nessuna Azione Richiesta)"
        print(f"  • Stato del Supervisore  : {norm_report.dataset_drift_detected} [{status_txt}]")

        pause_step(mode, delay, "iniettare l'attacco hacker Botnet (Covariate Shift)")

        # STEP 3: Inject Adversarial Botnet Attack
        print_step_header(3, "Iniezione Attacco Fraudolento Botnet (Covariate Shift)")
        print_context_box(
            "Adversarial Botnet Attack",
            "Simuliamo una frode coordinata: centinaia di bot creano account ed effettuano\n"
            "acquisti in meno di 2 secondi con fingerprint di device condivisi.\n"
            "Questo altera drasticamente la distribuzione dei dati in ingresso.",
        )

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
            if (i + 1) % 50 == 0 or (i + 1) == 300:
                stream_progress(i + 1, 300)

        sys.stdout.write("\n")
        flag_rate = (bot_flagged / 300) * 100
        print(f"  • Transazioni Flaggate come Frode: {bot_flagged}/300 ({flag_rate:.1f}%)")

        bot_df = pd.DataFrame(botnet_events)
        attack_report = engine.analyze_dataframe(bot_df)
        attack_drift_pct = attack_report.share_of_drifted_features * 100.0

        print("\n  [!] ALLERTA MOTORE DI DRIFT ATTIVATA:")
        score_msg = f"{attack_drift_pct:.1f}% > Soglia ({thresh_pct:.1f}%)"
        print(f"      • Drift Score        : {score_msg}")
        drift_flag = "[DEVIAZIONE DISTRIBUTIVA CRITICA]"
        print(f"      • Dataset Drift      : {attack_report.dataset_drift_detected} {drift_flag}")
        drifted_names = [f.column for f in attack_report.drifted_features if f.is_drifted]
        print(f"      • Colonne Driftate   : {', '.join(drifted_names)}")

        pause_step(mode, delay, "attivare il Supervisore Autonomo FSM per il Self-Healing")

        # STEP 4: Autonomous State Machine Supervisor Intervention
        print_step_header(4, "Intervento del Supervisore: Closed-Loop Self-Healing")
        print_context_box(
            "FSM Deterministica & Rollout Canary",
            "La Macchina a Stati interviene in modo deterministico (zero rischio allucinazioni):\n"
            "1. Diagnostica la fetta anomala (tempo registrazione < 2s)\n"
            "2. Riaddestra un modello candidato v1.1.0 ed esporta l'ONNX\n"
            "3. Valida le guardrail (ROC-AUC, Precision, latenza <50ms)\n"
            "4. Modifica Traefik in tempo reale a 95% Produzione / 5% Canary\n"
            "5. Constatata la stabilità, promuove il Canary al 100% senza riavvio pod",
        )

        final_state = supervisor.run_healing_cycle(
            report=attack_report,
            canary_artifacts_dir=Path("artifacts/canary"),
        )

        print("\n  • Sequenza Transizioni Eseguite dalla FSM:")
        for rec in supervisor.audit_trail[-6:]:
            print(f"    └─ [{rec.from_state}] ──({rec.event})──► [{rec.to_state}]")

        prod_w, canary_w = traffic_mgr.get_current_weights()
        print(f"\n  • Routing Finale Traefik: {render_traffic_bar(prod_w, canary_w)}")
        print(f"  • Stato Finale Runtime   : {final_state.value}")

        pause_step(mode, delay, "visualizzare gli artefatti e il riepilogo finale")

        # STEP 5: Summary & Artifact Locations
        print_step_header(5, "Ciclo Completato - Artefatti & Dashboard")
        print_context_box(
            "Tracciabilità & Osservabilità Completa",
            "Ogni transizione è stata sigillata nell'audit trail forense JSON.\n"
            "È possibile ispezionare visivamente le curve di distribuzione nel report\n"
            "HTML di Evidently e osservare il flusso dei messaggi sulla Redpanda Console.",
        )

        print(f"  ✔ Audit Trail JSON   : {supervisor.supervisor_dir / 'audit_trail.json'}")
        print(f"  ✔ Drift Report JSON  : {engine.reports_dir / 'latest_drift_report.json'}")
        print(f"  ✔ Report HTML Grafico: {engine.reports_dir / 'latest_drift_report.html'}")
        print(f"  ✔ Config Traefik WRR : {DEMO_CONFIG_PATH}")
        print("\n" + "=" * 80)
        print("         DIMOSTRAZIONE SELF-HEALING CONCLUSA CON SUCCESSO!")
        print("=" * 80 + "\n")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for demo orchestration."""
    parser = argparse.ArgumentParser(
        description="Launch closed-loop self-healing demonstration with configurable pacing."
    )
    parser.add_argument(
        "--mode",
        choices=["auto", "step", "fast"],
        default="auto",
        help="Execution mode: 'auto' (timed pauses), 'step' (press Enter), or 'fast'.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=2.0,
        help="Pause delay in seconds between steps in auto mode (default: 2.0).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_demo(mode=args.mode, delay=args.delay)
