# System Architecture

GlucoTwin is an India-aware Digital Twin designed for proactive Type 2 Diabetes (T2D) management. The architecture is organized into distinct decoupled layers to ensure scalability, clinical traceability, and real-time responsiveness.

## Architectural Diagram

```
+-------------------------------------------------------------+
|                      Client Layer                           |
|  Next.js 14 Dashboard (App Router, Tailwind, Recharts)     |
+------------------------------+------------------------------+
                               | REST & WebSockets
                               v
+-------------------------------------------------------------+
|                      Serving Layer                          |
|  FastAPI Gateway (FHIR Pydantic, Structured Logs, Metrics)  |
+---------------+-----------------------------+---------------+
                |                             |
                v                             v
+-------------------------------+  +--------------------------+
|       Twin Core Engine        |  |     Streaming Engine     |
| - State Management            |  | - Kafka Message Broker   |
| - Bergman Physiology Sim      |  | - Telemetry Replay       |
| - Indian Meal Modeler         |  | - Adverse Alert Pub/Sub  |
+---------------+---------------+  +--------------------------+
                |
                v
+-------------------------------------------------------------+
|                      Model Zoo                              |
| - Baseline Pinball Regressors (LightGBM, XGBoost)           |
| - Deep Multimodal Encoders (GRU, Temporal Fusion Xformer)   |
| - Distribution-Free Conformal Calibration (90% Coverage)    |
| - Per-Patient LoRA Residual Adapters                        |
| - Feature Attributions (SHAP & Multi-Head Attention)        |
+------------------------------+------------------------------+
                               |
                               v
+-------------------------------------------------------------+
|                      Storage Layer                          |
| - PostgreSQL + TimescaleDB (Hypertables for CGM Streams)    |
| - MLflow Experiment Registry & Model Artifacts              |
+-------------------------------------------------------------+
```

## Layer Descriptions

1. **Client Layer**: Next.js 14 frontend with TypeScript and Tailwind CSS. Exposes patient selection, real-time telemetry streaming, interactive what-if simulation, explainability heatmaps, and doctor-facing clinical summaries.
2. **Serving Layer**: FastAPI with FHIR-aligned Pydantic schemas, Prometheus metrics collection, correlation IDs, and versioned `/api/v1` routes.
3. **Twin Core Engine**: Two-compartment glucoregulatory minimal model adjusted for South Asian insulin resistance ($S_I$) and beta-cell deficit decay.
4. **Model Zoo**: Multi-horizon forecasting models (30, 60, 120 minutes) outputting multi-quantiles ($q_{10}, q_{50}, q_{90}$) wrapped in conformal prediction bands for finite-sample coverage guarantees.
5. **Streaming & Storage**: Kafka event bus feeding real-time time-series streams into TimescaleDB hypertables.
