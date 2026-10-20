from datetime import datetime
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
)
from sqlalchemy.orm import Mapped, mapped_column

from src.glucotwin.db.session import Base


class PatientRecord(Base):
    __tablename__ = "patients"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, index=True)
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    age: Mapped[int] = mapped_column(Integer, nullable=False)
    gender: Mapped[str] = mapped_column(String(16), nullable=False)
    region: Mapped[str] = mapped_column(String(32), nullable=False, default="West India")
    dietary_habit: Mapped[str] = mapped_column(String(32), nullable=False, default="Vegetarian")
    bmi: Mapped[float] = mapped_column(Float, nullable=False)
    hba1c: Mapped[float] = mapped_column(Float, nullable=False)
    diabetes_duration_years: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    egfr: Mapped[float] = mapped_column(Float, nullable=False, default=90.0)
    static_ehr_vector: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    active_medications: Mapped[list[Any]] = mapped_column(JSON, nullable=False, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow
    )


class TwinWindowReading(Base):
    __tablename__ = "cgm_observations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    patient_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("patients.id"), nullable=False, index=True
    )
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    glucose_mgdl: Mapped[float] = mapped_column(Float, nullable=False)
    heart_rate_bpm: Mapped[float | None] = mapped_column(Float, nullable=True)
    steps: Mapped[int | None] = mapped_column(Integer, nullable=True, default=0)
    mets: Mapped[float | None] = mapped_column(Float, nullable=True, default=1.0)
    carbs_g: Mapped[float | None] = mapped_column(Float, nullable=True, default=0.0)


# Backwards compatible alias for earlier integration tests
CGMObservation = TwinWindowReading


class TwinPrediction(Base):
    __tablename__ = "twin_predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    patient_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("patients.id"), nullable=False, index=True
    )
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    horizon_minutes: Mapped[int] = mapped_column(Integer, nullable=False)
    target_timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )
    model_version: Mapped[str] = mapped_column(String(64), nullable=False)
    serving_fold: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    q10: Mapped[float] = mapped_column(Float, nullable=False)
    q50: Mapped[float] = mapped_column(Float, nullable=False)
    q90: Mapped[float] = mapped_column(Float, nullable=False)
    calibrated_lower_95: Mapped[float] = mapped_column(Float, nullable=False)
    calibrated_upper_95: Mapped[float] = mapped_column(Float, nullable=False)
    realised_glucose: Mapped[float | None] = mapped_column(Float, nullable=True)
    prediction_error: Mapped[float | None] = mapped_column(Float, nullable=True)
    absolute_error: Mapped[float | None] = mapped_column(Float, nullable=True)
    relative_error_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)


class TwinAlertEpisode(Base):
    __tablename__ = "twin_alert_episodes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    patient_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("patients.id"), nullable=False, index=True
    )
    event_type: Mapped[str] = mapped_column(String(64), nullable=False)
    horizon_minutes: Mapped[int] = mapped_column(Integer, nullable=False)
    start_timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )
    end_timestamp: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    operating_threshold: Mapped[float] = mapped_column(Float, nullable=False)
    trigger_probability: Mapped[float] = mapped_column(Float, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    was_true_alert: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    lead_time_minutes: Mapped[float | None] = mapped_column(Float, nullable=True)


class PersonalizationAdapter(Base):
    __tablename__ = "personalization_adapters"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    patient_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("patients.id"), nullable=False, index=True
    )
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    train_window_start: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    train_window_end: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    eval_window_start: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    eval_window_end: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    baseline_rmse: Mapped[float] = mapped_column(Float, nullable=False)
    baseline_mard: Mapped[float] = mapped_column(Float, nullable=False)
    adapter_rmse: Mapped[float] = mapped_column(Float, nullable=False)
    adapter_mard: Mapped[float] = mapped_column(Float, nullable=False)
    promoted: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    adapter_weights: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)
