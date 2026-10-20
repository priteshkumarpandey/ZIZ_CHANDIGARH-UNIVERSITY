from datetime import datetime
from typing import Any

from pydantic import BaseModel

from src.glucotwin.serving.schemas.fhir import AdverseEventType, PatientProfile


class ForecastPoint(BaseModel):
    horizon_minutes: int
    target_timestamp: datetime
    q10: float
    q50: float
    q90: float
    calibrated_lower_95: float
    calibrated_upper_95: float
    spike_risk_probability: float
    hypo_risk_probability: float
    hyper_risk_probability: float | None = None
    severe_hyper_risk_probability: float | None = None
    predicted_event: AdverseEventType
    operating_thresholds: dict[str, float] | None = None
    event_reliabilities: dict[str, bool] | None = None
    reliability_notes: dict[str, str] | None = None
    alert_usable: dict[str, bool] | None = None
    risk_labels: dict[str, str] | None = None
    operating_metrics: dict[str, Any] | None = None


class RuleBasedAlert(BaseModel):
    alert_type: str
    severity: str
    title: str
    message: str
    observed_glucose: float
    threshold_mgdl: float
    rate_mgdl_per_min: float | None = None
    is_rule_based: bool = True
    provenance: str = "Rule-based alert (observed, not predicted)"


class ForecastResponse(BaseModel):
    patient_id: str
    forecast_generated_at: datetime
    current_glucose: float
    model_version: str
    forecast_horizons: list[ForecastPoint]
    adverse_event_warning: bool
    summary_message: str
    inference_latency_ms: float | None = None
    serving_fold: int | None = None
    operating_thresholds_summary: dict[str, Any] | None = None
    is_synthetic: bool = False
    rule_based_alerts: list[RuleBasedAlert] = []


class FeatureAttribution(BaseModel):
    feature_name: str
    shap_value: float
    base_value: float
    user_value: float
    description: str
    model: str | None = None


class ExplainabilityResponse(BaseModel):
    patient_id: str
    horizon_minutes: int
    model_name: str = "Multimodal GRU"
    top_features: list[FeatureAttribution]
    gru_permutation_importance: list[FeatureAttribution] | None = None
    lightgbm_shap_values: list[FeatureAttribution] | None = None
    temporal_attention_weights: list[dict[str, Any]]
    static_covariate_weights: dict[str, float]
    variation_note: str | None = None
    as_of: str | None = None
    current_glucose: float | None = None


class SimulationResponse(BaseModel):
    patient_id: str
    simulated_at: datetime
    timestamps: list[datetime]
    baseline_trajectory: list[float]
    scenario_trajectory: list[float]
    scenario_q10: list[float]
    scenario_q90: list[float]
    baseline_peak_glucose: float
    scenario_peak_glucose: float
    peak_reduction_mgdl: float
    peak_change_mgdl: float | None = None
    baseline_horizons: dict[str, dict[str, float]] | None = None
    scenario_horizons: dict[str, dict[str, float]] | None = None
    deltas: dict[str, float] | None = None
    provenance: dict[str, Any] | None = None
    interventions_applied: list[dict[str, Any]] | None = None
    constraint_applied: bool | None = False
    constraint_correction_mgdl: float | None = 0.0
    clinical_takeaway: str
    is_synthetic: bool = False
    as_of: str | None = None
    starting_glucose: float | None = None


class PersonalizationStatusResponse(BaseModel):
    patient_id: str
    adapter_version: int
    last_calibrated_at: datetime
    historical_mard_percentage: list[float]
    historical_rmse_mgdl: list[float]
    num_samples_seen: int
    is_active: bool


class ClinicalMetrics(BaseModel):
    time_in_range_percentage: float
    time_above_range_percentage: float
    time_above_range_level2_percentage: float | None = 0.0
    time_below_range_percentage: float
    time_below_range_level2_percentage: float | None = 0.0
    gmi: float
    mean_glucose: float
    standard_deviation: float
    coefficient_of_variation: float


class ClinicalSummaryResponse(BaseModel):
    patient_id: str
    patient_name: str
    generated_at: datetime
    metrics: ClinicalMetrics
    adverse_event_count_24h: dict[str, int]
    latest_forecast: list[ForecastPoint] | None = None
    adverse_events_usable: dict[str, Any] | None = None
    personalization_status: str | None = "calibrated"
    key_findings: list[str]
    actionable_recommendations: list[str]
    plain_language_narrative: str
    is_llm_enhanced: bool
    monitoring_window_text: str | None = None
    disclaimer: str = (
        "Decision support only. Not intended as medical diagnosis or direct prescription."
    )
    is_synthetic: bool = False


class TwinStateResponse(BaseModel):
    patient: PatientProfile
    current_glucose: float
    trend_arrow: str
    last_updated: datetime
    active_adverse_events: list[str]
    is_streaming: bool
    is_synthetic: bool = False
    rule_based_alerts: list[RuleBasedAlert] = []
