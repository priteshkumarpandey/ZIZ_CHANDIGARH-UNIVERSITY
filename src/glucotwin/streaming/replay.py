"""Telemetry replay service for GlucoTwin.

Reads a patient series (from demo parquet or real processed dataset)
and publishes events to Kafka topics: cgm, activity, and meals at a
configurable playback speed (1x to 60x).
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.glucotwin.config.settings import settings
from src.glucotwin.streaming.kafka import kafka_manager
from src.glucotwin.utils.io import as_pandas_df, read_parquet
from src.glucotwin.utils.logging import logger


class TelemetryReplaySimulator:
    """Reads patient series and publishes to Kafka topics at configurable speed."""

    def __init__(
        self,
        patient_id: str | None = None,
        speed_multiplier: float = 30.0,
        demo_mode: bool = True,
    ) -> None:
        self.patient_id = patient_id or settings.REPLAY_PATIENT_ID
        self.speed_multiplier = max(1.0, min(60.0, float(speed_multiplier)))
        self.demo_mode = demo_mode or settings.DEMO_MODE
        self.readings: list[dict[str, Any]] = []

    def load_series(self) -> None:
        """Load series from demo data or processed dataset."""
        demo_file = Path("data/demo/timeseries_demo.parquet")
        processed_file = Path("data/processed/timeseries_processed.parquet")

        chosen_file = None
        if self.demo_mode or not processed_file.exists():
            if not demo_file.exists():
                try:
                    from scripts.make_demo_data import generate_demo_data

                    generate_demo_data()
                except Exception as exc:
                    logger.warning(f"Could not generate demo data: {exc}")
            chosen_file = demo_file if demo_file.exists() else None
        else:
            chosen_file = processed_file

        if not chosen_file or not chosen_file.exists():
            # Generate fallback in-memory synthetic readings
            logger.info("Using dynamic in-memory synthetic generator for replay.")
            self._generate_fallback_readings()
            return

        df = as_pandas_df(read_parquet(chosen_file))
        patient_df = df[df["patient_id"] == self.patient_id]
        if patient_df.empty:
            # Fall back to first patient in dataset
            first_pid = df["patient_id"].iloc[0]
            logger.info(
                f"Patient {self.patient_id} not found in {chosen_file}; falling back to {first_pid}"
            )
            self.patient_id = str(first_pid)
            patient_df = df[df["patient_id"] == self.patient_id]

        patient_df = patient_df.sort_values("timestamp")
        self.readings = patient_df.to_dict(orient="records")
        logger.info(
            f"Loaded {len(self.readings)} replay records for patient {self.patient_id} from {chosen_file}"
        )

    def _generate_fallback_readings(self) -> None:
        """In-memory 288-step synthetic fallback."""
        from scripts.make_demo_data import generate_synthetic_patient_timeseries

        now = datetime.now(UTC)
        self.readings = generate_synthetic_patient_timeseries(
            patient_id=self.patient_id,
            base_glucose=130.0,
            carb_sensitivity=0.8,
            clearance_rate=0.04,
            daily_meals=[(8.5, 50.0, "Breakfast"), (13.0, 65.0, "Lunch"), (20.0, 55.0, "Dinner")],
            start_time=now,
            days=1,
        )

    async def run(self) -> None:
        """Main replay loop publishing to Kafka."""
        self.load_series()
        if not self.readings:
            logger.error("No readings available for replay.")
            return

        # Base cadence is 300 seconds (5 minutes real world) divided by speed multiplier
        cadence_seconds = max(0.1, 300.0 / self.speed_multiplier)
        logger.info(
            f"Starting Telemetry Replay for {self.patient_id} at {self.speed_multiplier:.1f}x speed "
            f"({cadence_seconds:.2f}s per 5m reading)"
        )

        connected = await kafka_manager.connect_producer()
        if not connected:
            logger.warning("Kafka producer unavailable; running in offline simulation mode.")

        seq_id = 0
        while True:
            for item in self.readings:
                seq_id += 1
                now_str = datetime.now(UTC).isoformat()

                glucose_val = float(item.get("glucose", item.get("glucose_mgdl", 120.0)))
                carbs_val = float(item.get("carbohydrates", item.get("carbs_g", 0.0)))
                mets_val = float(item.get("mets", 1.0))
                steps_val = int(item.get("steps", 0))
                hr_val = float(item.get("heart_rate", item.get("heart_rate_bpm", 72.0)))
                act_type = str(item.get("activity_type", "sedentary"))
                meal_type = str(item.get("meal_type", "none"))

                publish_dt = datetime.now(UTC)
                # 1. Publish CGM reading
                cgm_payload = {
                    "patient_id": self.patient_id,
                    "timestamp": now_str,
                    "published_at": publish_dt.isoformat(),
                    "published_epoch_ms": int(publish_dt.timestamp() * 1000),
                    "glucose_mgdl": round(glucose_val, 1),
                    "heart_rate_bpm": round(hr_val, 1),
                    "steps": steps_val,
                    "mets": round(mets_val, 2),
                    "carbs_g": round(carbs_val, 1),
                    "sequence_id": seq_id,
                }
                await kafka_manager.publish_cgm(self.patient_id, cgm_payload)

                # 2. Publish Meal event if carbs consumed
                if carbs_val > 0.0:
                    meal_payload = {
                        "patient_id": self.patient_id,
                        "timestamp": now_str,
                        "carbs_g": round(carbs_val, 1),
                        "meal_type": meal_type,
                        "sequence_id": seq_id,
                    }
                    await kafka_manager.publish_meal(self.patient_id, meal_payload)

                # 3. Publish Activity event if active
                if mets_val > 1.5 or steps_val > 50:
                    act_payload = {
                        "patient_id": self.patient_id,
                        "timestamp": now_str,
                        "activity_type": act_type,
                        "mets": round(mets_val, 2),
                        "steps": steps_val,
                        "duration_minutes": 30,
                        "sequence_id": seq_id,
                    }
                    await kafka_manager.publish_activity(self.patient_id, act_payload)

                logger.debug(
                    f"[Replay tick {seq_id}] Patient {self.patient_id}: Glucose={glucose_val:.1f} mg/dL"
                )
                await asyncio.sleep(cadence_seconds)


async def main():
    parser = argparse.ArgumentParser(description="GlucoTwin Telemetry Replay Service")
    parser.add_argument(
        "--patient", type=str, default=settings.REPLAY_PATIENT_ID, help="Patient ID to replay"
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=settings.REPLAY_SPEED,
        help="Replay speed multiplier (1x-60x)",
    )
    parser.add_argument("--demo", action="store_true", default=True, help="Force demo mode data")
    args = parser.parse_args()

    simulator = TelemetryReplaySimulator(
        patient_id=args.patient,
        speed_multiplier=args.speed,
        demo_mode=args.demo,
    )
    await simulator.run()


if __name__ == "__main__":
    asyncio.run(main())
