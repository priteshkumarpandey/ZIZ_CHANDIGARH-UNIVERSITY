"""Prometheus telemetry metrics for GlucoTwin serving and streaming."""

from __future__ import annotations

try:
    from prometheus_client import Counter, Histogram

    HAS_PROMETHEUS = True
except ImportError:
    HAS_PROMETHEUS = False

    class DummyMetric:
        def labels(self, *args, **kwargs):
            return self

        def inc(self, *args, **kwargs):
            pass

        def observe(self, *args, **kwargs):
            pass

        def set(self, *args, **kwargs):
            pass

    def Counter(*args, **kwargs):  # type: ignore
        return DummyMetric()

    def Histogram(*args, **kwargs):  # type: ignore
        return DummyMetric()


# 1. API request metrics
REQUEST_COUNT = Counter(
    "glucotwin_api_requests_total",
    "Total API requests processed",
    ["method", "endpoint", "status_code"],
)
REQUEST_LATENCY = Histogram(
    "glucotwin_api_request_duration_seconds",
    "API request latency in seconds",
    ["endpoint"],
)

# 2. Forecasting inference latency
PREDICTION_LATENCY = Histogram(
    "glucotwin_prediction_latency_seconds",
    "Inference latency for glucose forecast in seconds",
    ["model_version", "serving_fold"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)

# 3. Stream ingestion lag (wall clock time minus reading timestamp)
STREAM_LAG = Histogram(
    "glucotwin_stream_lag_seconds",
    "Stream processing lag in seconds",
    ["patient_id"],
    buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 300.0),
)

# 4. Adverse glycemic alerts fired
ALERTS_FIRED = Counter(
    "glucotwin_alerts_fired_total",
    "Number of adverse event alerts fired",
    ["event_type", "horizon_minutes", "alert_usable"],
)

# 5. Predictions evaluated and scored
PREDICTIONS_SCORED = Counter(
    "glucotwin_predictions_scored_total",
    "Number of predictions evaluated against realised CGM readings",
    ["patient_id", "horizon_minutes"],
)
