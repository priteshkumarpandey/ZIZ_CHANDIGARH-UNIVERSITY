"""Unit tests for streaming replay, Kafka integration, and alert gating."""

from __future__ import annotations

import unittest
from datetime import UTC, datetime

from src.glucotwin.serving.routers.websocket import (
    is_alert_usable,
    process_and_broadcast_cgm_reading,
)
from src.glucotwin.streaming.kafka import kafka_manager
from src.glucotwin.streaming.replay import TelemetryReplaySimulator


class TestStreaming(unittest.IsolatedAsyncioTestCase):
    async def test_replay_simulator_series_load(self):
        sim = TelemetryReplaySimulator(
            patient_id="DEMO_PAT_001", speed_multiplier=60.0, demo_mode=True
        )
        sim.load_series()
        self.assertGreater(len(sim.readings), 0)
        first = sim.readings[0]
        self.assertEqual(first["patient_id"], "DEMO_PAT_001")
        self.assertIn("glucose", first)

    async def test_kafka_manager_topics_offline(self):
        # Graceful fallback when Kafka is offline
        await kafka_manager.connect_producer()
        self.assertFalse(kafka_manager.is_connected)

        pub_cgm = await kafka_manager.publish_cgm("DEMO_PAT_001", {"glucose_mgdl": 120.0})
        self.assertFalse(pub_cgm)

        pub_act = await kafka_manager.publish_activity("DEMO_PAT_001", {"mets": 3.0})
        self.assertFalse(pub_act)

        pub_meal = await kafka_manager.publish_meal("DEMO_PAT_001", {"carbs_g": 50.0})
        self.assertFalse(pub_meal)

        await kafka_manager.disconnect()

    def test_alert_usable_gating(self):
        # Usable events in operating_thresholds.json
        self.assertTrue(is_alert_usable("hyperglycemia_30m"))
        self.assertTrue(is_alert_usable("severe_hyperglycemia_30m"))
        self.assertTrue(is_alert_usable("spike_30m"))
        self.assertTrue(is_alert_usable("severe_hyperglycemia_60m"))
        self.assertTrue(is_alert_usable("spike_60m"))

        # Gated / unusable events (recall < 0.40 or low precision)
        self.assertFalse(is_alert_usable("hypoglycemia_30m"))
        self.assertFalse(is_alert_usable("hyperglycemia_60m"))
        self.assertFalse(is_alert_usable("hypoglycemia_60m"))
        self.assertFalse(is_alert_usable("hyperglycemia_120m"))
        self.assertFalse(is_alert_usable("severe_hyperglycemia_120m"))
        self.assertFalse(is_alert_usable("hypoglycemia_120m"))
        self.assertFalse(is_alert_usable("spike_120m"))

    async def test_process_and_broadcast_cgm_reading(self):
        reading = {
            "patient_id": "DEMO_PAT_001",
            "timestamp": datetime.now(UTC).isoformat(),
            "glucose_mgdl": 145.0,
            "heart_rate_bpm": 76.0,
            "steps": 25,
            "mets": 1.2,
            "carbs_g": 0.0,
        }
        payload = await process_and_broadcast_cgm_reading(reading)
        self.assertEqual(payload["patient_id"], "DEMO_PAT_001")
        self.assertEqual(payload["glucose_mgdl"], 145.0)
        self.assertIn("forecast_30m", payload)
        self.assertIn("forecast_60m", payload)
        self.assertIn("forecast_120m", payload)
        self.assertIn("active_alerts", payload)
        self.assertTrue(payload["is_synthetic"])
        # All alerts must be strictly usable
        for alert in payload["active_alerts"]:
            self.assertTrue(alert["alert_usable"])
