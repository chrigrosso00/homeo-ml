# Self-Healing MLOps Runtime

Autonomous Canary Deployment & Closed-Loop Drift Remediation for Real-World E-Commerce Fraud Detection.

## Architecture
- **Model Serving Layer**: FastAPI + ONNX Runtime for low-latency inference (<20ms).
- **Streaming Buffer**: Redpanda (Kafka-compatible event broker) for asynchronous telemetry logging.
- **Drift Detection Engine**: Evidently / Whylogs for real-time statistical drift (PSI, Wasserstein distance).
- **State Machine Supervisor**: Deterministic autonomous workflow for slice analysis, targeted retraining, and canary rollout.
- **Canary Ingress**: Traefik with weighted dynamic traffic splitting (95% Production / 5% Canary).
