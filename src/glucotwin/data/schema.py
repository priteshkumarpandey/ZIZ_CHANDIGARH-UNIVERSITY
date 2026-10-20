import pandas as pd
import pandera as pa
from pandera.typing import Series


class PatientsSchema(pa.DataFrameModel):
    patient_id: Series[str] = pa.Field(nullable=False, unique=True)
    source_dataset: Series[str] = pa.Field(
        nullable=False, isin=["cgmacros", "ai_readi_synthetic", "hupa_ucm", "synthetic"]
    )
    is_synthetic: Series[bool] = pa.Field(nullable=False)
    diabetes_type: Series[str] = pa.Field(
        nullable=False, isin=["T2D", "T1D", "none", "prediabetes", "unknown"]
    )
    diabetes_status: Series[str] = pa.Field(
        nullable=False, isin=["T2D", "prediabetes", "healthy", "T1D", "unknown"]
    )
    age: Series[float] = pa.Field(ge=0.0, le=120.0, nullable=True)
    sex: Series[str] = pa.Field(
        nullable=True, isin=["M", "F", "other", "Male", "Female", "UNKNOWN", None]
    )
    bmi: Series[float] = pa.Field(ge=10.0, le=70.0, nullable=True)
    hba1c: Series[float] = pa.Field(ge=3.0, le=20.0, nullable=True)
    fasting_glucose: Series[float] = pa.Field(ge=30.0, le=500.0, nullable=True)
    medications: Series[object] = pa.Field(nullable=True)
    systolic_bp: Series[float] = pa.Field(ge=50.0, le=300.0, nullable=True)
    diastolic_bp: Series[float] = pa.Field(ge=30.0, le=200.0, nullable=True)
    cholesterol: Series[float] = pa.Field(ge=20.0, le=1000.0, nullable=True)
    triglycerides: Series[float] = pa.Field(ge=10.0, le=2500.0, nullable=True)
    hdl: Series[float] = pa.Field(ge=5.0, le=300.0, nullable=True)
    ldl: Series[float] = pa.Field(ge=5.0, le=1000.0, nullable=True)
    creatinine: Series[float] = pa.Field(ge=0.0, le=25.0, nullable=True)
    insulin: Series[float] = pa.Field(ge=0.0, le=500.0, nullable=True)
    height_cm: Series[float] = pa.Field(ge=50.0, le=250.0, nullable=True)
    weight_kg: Series[float] = pa.Field(ge=20.0, le=300.0, nullable=True)

    class Config:
        strict = False
        coerce = True


class TimeseriesSchema(pa.DataFrameModel):
    patient_id: Series[str] = pa.Field(nullable=False)
    timestamp: Series[pa.DateTime] = pa.Field(coerce=True)
    glucose_mgdl: Series[float] = pa.Field(ge=30.0, le=500.0, nullable=True)
    heart_rate: Series[float] = pa.Field(ge=25.0, le=250.0, nullable=True)
    steps: Series[float] = pa.Field(ge=0.0, le=20000.0, nullable=True)
    activity_level: Series[float] = pa.Field(ge=0.0, le=30.0, nullable=True)
    sleep_stage: Series[str] = pa.Field(nullable=True)
    hrv: Series[float] = pa.Field(ge=0.0, le=500.0, nullable=True)
    missing_flag: Series[bool] = pa.Field(nullable=False)

    class Config:
        strict = False
        coerce = True


class MealsSchema(pa.DataFrameModel):
    patient_id: Series[str] = pa.Field(nullable=False)
    timestamp: Series[pa.DateTime] = pa.Field(coerce=True)
    carbs_g: Series[float] = pa.Field(ge=0.0, le=2000.0)
    protein_g: Series[float] = pa.Field(ge=0.0, le=1000.0, nullable=True)
    fat_g: Series[float] = pa.Field(ge=0.0, le=1000.0, nullable=True)
    fiber_g: Series[float] = pa.Field(ge=0.0, le=10000.0, nullable=True)
    calories: Series[float] = pa.Field(ge=0.0, le=25000.0, nullable=True)
    meal_label: Series[str] = pa.Field(nullable=True)

    class Config:
        strict = False
        coerce = True


class ValidationGlucoseSchema(pa.DataFrameModel):
    patient_id: Series[str] = pa.Field(nullable=False)
    timestamp: Series[pa.DateTime] = pa.Field(coerce=True)
    fingerstick_glucose_mgdl: Series[float] = pa.Field(ge=30.0, le=500.0)
    measurement_number: Series[int] = pa.Field(ge=1, le=10)

    class Config:
        strict = False
        coerce = True


def validate_patients_df(df: pd.DataFrame) -> pd.DataFrame:
    return PatientsSchema.validate(df)


def validate_timeseries_df(df: pd.DataFrame) -> pd.DataFrame:
    return TimeseriesSchema.validate(df)


def validate_meals_df(df: pd.DataFrame) -> pd.DataFrame:
    return MealsSchema.validate(df)


def validate_validation_glucose_df(df: pd.DataFrame) -> pd.DataFrame:
    return ValidationGlucoseSchema.validate(df)
