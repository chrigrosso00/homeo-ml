#!/usr/bin/env bash
# ==============================================================================
# Self-Healing MLOps Runtime - Automated Demonstration Launcher
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

PYTHON_BIN="/home/cgrosso/miniforge3/envs/mlops-runtime/bin/python"

if [[ ! -x "${PYTHON_BIN}" ]]; then
    if command -v python &>/dev/null; then
        PYTHON_BIN="$(command -v python)"
    else
        echo "Error: Python interpreter not found in conda env 'mlops-runtime' or PATH."
        exit 1
    fi
fi

echo "================================================================================"
echo "          SELF-HEALING MLOps RUNTIME: CLOSED-LOOP ORCHESTRATION"
echo "================================================================================"
echo "Interpreter: ${PYTHON_BIN}"
echo "Project Dir: ${SCRIPT_DIR}"
echo ""

# Check Docker containers
if command -v docker &>/dev/null; then
    if docker ps --format '{{.Names}}' | grep -q 'mlops-redpanda'; then
        echo "✔ Redpanda Event Broker is running."
    else
        echo "ℹ Starting Redpanda telemetry streaming broker..."
        docker compose up -d
        sleep 3
    fi
fi

echo ""
echo "Opzioni disponibili per il controllo dell'avanzamento:"
echo "  • ./demo.sh --mode step       (avanza manualmente premendo [INVIO])"
echo "  • ./demo.sh --mode auto       (avanza con pause temporizzate di 2s)"
echo "  • ./demo.sh --mode fast       (esecuzione immediata senza pause)"
echo ""

# Forward any user arguments or default to --mode auto --delay 2.0
if [[ $# -eq 0 ]]; then
    ARGS=(--mode auto --delay 2.0)
else
    ARGS=("$@")
fi

echo "Avvio della Pipeline con argomenti: ${ARGS[*]}"
echo ""
"${PYTHON_BIN}" -m scripts.demo_closed_loop "${ARGS[@]}"

echo ""
echo "================================================================================"
echo "Observability Dashboards & Artifacts:"
echo "  • Redpanda Console    : http://localhost:8080"
echo "  • Evidently Drift HTML: file://${SCRIPT_DIR}/artifacts/monitoring/latest_drift_report.html"
echo "  • Supervisor Audit Log: ${SCRIPT_DIR}/artifacts/supervisor/audit_trail.json"
echo "  • Traefik Dynamic YML : ${SCRIPT_DIR}/config/traefik_dynamic.yml"
echo "================================================================================"
