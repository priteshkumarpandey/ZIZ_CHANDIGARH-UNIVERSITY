import pandas as pd

try:
    import pandera as pa
    from pandera.typing import Series

    class CGMTimeseriesSchema(pa.DataFrameModel):
        timestamp: Series[pd.DatetimeTZDtype] = pa.Field(coerce=True)
        patient_id: Series[str] = pa.Field(nullable=False)
        glucose_mgdl: Series[float] = pa.Field(ge=30.0, le=500.0)
        heart_rate_bpm: Series[float] = pa.Field(ge=35.0, le=220.0, nullable=True)
        steps: Series[int] = pa.Field(ge=0, le=10000, nullable=True)
        mets: Series[float] = pa.Field(ge=0.8, le=16.0, nullable=True)
        carbs_g: Series[float] = pa.Field(ge=0.0, le=300.0, nullable=True)
        glycemic_index: Series[float] = pa.Field(ge=0.0, le=100.0, nullable=True)

        class Config:
            strict = False
            coerce = True

    class StaticEHRSchema(pa.DataFrameModel):
        patient_id: Series[str] = pa.Field(nullable=False, unique=True)
        age: Series[int] = pa.Field(ge=18, le=100)
        bmi: Series[float] = pa.Field(ge=14.0, le=60.0)
        hba1c: Series[float] = pa.Field(ge=4.0, le=18.0)
        diabetes_duration_years: Series[float] = pa.Field(ge=0.0, le=60.0)
        egfr: Series[float] = pa.Field(ge=10.0, le=150.0)
        systolic_bp: Series[int] = pa.Field(ge=70, le=250)
        diastolic_bp: Series[int] = pa.Field(ge=40, le=150)

        class Config:
            strict = False
            coerce = True

    def validate_timeseries(df: pd.DataFrame) -> pd.DataFrame:
        return CGMTimeseriesSchema.validate(df)

    def validate_static_ehr(df: pd.DataFrame) -> pd.DataFrame:
        return StaticEHRSchema.validate(df)

except ImportError:

    def validate_timeseries(df: pd.DataFrame) -> pd.DataFrame:
        return df

    def validate_static_ehr(df: pd.DataFrame) -> pd.DataFrame:
        return df
