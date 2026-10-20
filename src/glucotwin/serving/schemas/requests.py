from datetime import datetime

from pydantic import BaseModel

from src.glucotwin.serving.schemas.fhir import ActivityRecord, CGMReading, MealRecord


class InterventionItem(BaseModel):
    intervention_type: str = "meal"  # "meal" or "activity"
    food_name: str | None = None
    carbs_g: float | None = 0.0
    protein_g: float | None = 0.0
    fat_g: float | None = 0.0
    fiber_g: float | None = 0.0
    glycemic_index: float | None = None
    activity_type: str | None = None
    duration_minutes: int | None = 0
    mets: float | None = 1.0
    offset_minutes: int = 0
    provenance: str | None = None
    is_approximate: bool | None = False
    notes: str | None = None


class PredictRequest(BaseModel):
    patient_id: str
    recent_cgm_series: list[float]
    recent_carbs_g: float | None = 0.0
    recent_glycemic_index: float | None = 0.0
    recent_heart_rate: float | None = 72.0
    recent_mets: float | None = 1.0
    as_of: datetime | None = None


class SimulateRequest(BaseModel):
    patient_id: str
    as_of: datetime | None = None
    interventions: list[InterventionItem] | None = None
    recent_cgm_series: list[float] | None = None
    baseline_cgm: list[CGMReading] = []
    planned_meal: MealRecord | None = None
    planned_activity: ActivityRecord | None = None
    simulation_duration_minutes: int = 180
    calibrate_meal_response: bool = False


class PersonalizeRequest(BaseModel):
    patient_id: str
    recent_embeddings: list[list[float]] | None = None
    actual_targets: list[float] | None = None
    learning_rate: float = 0.005
    steps: int = 5
