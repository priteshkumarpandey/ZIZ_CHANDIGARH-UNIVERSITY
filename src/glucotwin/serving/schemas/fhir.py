from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class Gender(StrEnum):
    MALE = "male"
    FEMALE = "female"
    OTHER = "other"


class Region(StrEnum):
    NORTH_INDIA = "North India"
    SOUTH_INDIA = "South India"
    EAST_INDIA = "East India"
    WEST_INDIA = "West India"


class DietaryHabit(StrEnum):
    VEGETARIAN = "Vegetarian"
    NON_VEGETARIAN = "Non-Vegetarian"
    EGGETARIAN = "Eggetarian"
    JAIN = "Jain Vegetarian"


class MedicationClass(StrEnum):
    METFORMIN = "Biguanide (Metformin)"
    SULFONYLUREA = "Sulfonylurea (e.g. Glimepiride)"
    DPP4_INHIBITOR = "DPP-4 Inhibitor (e.g. Teneligliptin)"
    SGLT2_INHIBITOR = "SGLT2 Inhibitor (e.g. Dapagliflozin)"
    GLP1_RA = "GLP-1 RA (e.g. Semaglutide)"
    BASAL_INSULIN = "Basal Insulin (e.g. Glargine)"
    BOLUS_INSULIN = "Bolus Insulin (e.g. Lispro)"


class AdverseEventType(StrEnum):
    NONE = "none"
    HYPOGLYCEMIA_LEVEL_1 = "hypoglycemia_level_1"  # < 70 mg/dL
    HYPOGLYCEMIA_LEVEL_2 = "hypoglycemia_level_2"  # < 54 mg/dL
    HYPERGLYCEMIA_LEVEL_1 = "hyperglycemia_level_1"  # > 180 mg/dL
    HYPERGLYCEMIA_LEVEL_2 = "hyperglycemia_level_2"  # > 250 mg/dL


class PatientProfile(BaseModel):
    id: str
    name: str
    age: int
    gender: Gender
    region: Region | None = None
    occupation: str | None = None
    dietary_habit: DietaryHabit | None = None
    bmi: float
    hba1c: float
    diabetes_duration_years: float | None = None
    egfr: float | None = None
    systolic_bp: int | None = None
    diastolic_bp: int | None = None
    fasting_glucose: float
    postprandial_glucose: float | None = None
    active_medications: list[str] = Field(default_factory=list)
    medication_classes: list[MedicationClass] = Field(default_factory=list)
    phenotype_notes: str = ""
    ethnicity: str | None = None
    matched_participant_id: str | None = None
    matched_test_fold: int | None = None
    linkage_type: str | None = None
    typical_meal_templates: list[dict[str, Any]] | None = None
    is_synthetic: bool = False
    created_at: datetime = Field(default_factory=datetime.utcnow)


class CGMReading(BaseModel):
    patient_id: str
    timestamp: datetime
    glucose_mgdl: float
    trend_arrow: str | None = "Flat"
    heart_rate_bpm: float | None = None
    steps: int | None = 0
    mets: float | None = 1.0


class FoodItem(BaseModel):
    id: str
    name: str
    regional_category: str
    serving_size_g: float
    carbohydrates_g: float
    fiber_g: float
    protein_g: float
    fat_g: float
    glycemic_index: float
    source: str = "derived from IFCT 2017 raw ingredient values (local extraction, not redistributed) and team recipe assumptions"
    is_approximate: bool | None = False
    notes: str | None = None
    composition_source: str | None = (
        "derived from IFCT 2017 raw ingredient values (local extraction, not redistributed) and team recipe assumptions"
    )
    gi_source: str | None = "approximate, not verified"


class MealRecord(BaseModel):
    timestamp: datetime
    name: str
    items: list[FoodItem] = []
    total_carbohydrates_g: float
    total_fiber_g: float
    total_protein_g: float
    total_fat_g: float
    effective_glycemic_load: float


class ActivityRecord(BaseModel):
    timestamp: datetime
    activity_type: str
    duration_minutes: int
    mets: float
    steps_estimate: int
