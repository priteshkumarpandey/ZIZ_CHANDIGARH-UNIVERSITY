"""WebSocket router for real-time patient telemetry streaming and alerts."""

from __future__ import annotations

import asyncio
import json
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from src.glucotwin.serving.metrics import ALERTS_FIRED, PREDICTIONS_SCORED
from src.glucotwin.streaming.kafka import kafka_manager
from src.glucotwin.twin.state import twin_state_manager
from src.glucotwin.utils.logging import logger

router = APIRouter(tags=["Streaming"])

# Load operating thresholds to strictly filter usable alerts
_THRESHOLDS_PATH = Path("models/checkpoints/operating_thresholds.json")
_OPERATING_THRESHOLDS: dict[str, Any] = {}
if _THRESHOLDS_PATH.exists():
    try:
        with open(_THRESHOLDS_PATH, encoding="utf-8") as fp:
            _OPERATING_THRESHOLDS = json.load(fp)
    except Exception as exc:
        logger.warning(f"Failed to load operating thresholds for alerts: {exc}")


def is_alert_usable(event_name: str) -> bool:
    """Check if an event is designated alert_usable."""
    events = _OPERATING_THRESHOLDS.get("events", {})
    ev_info = events.get(event_name, {})
    return bool(ev_info.get("alert_usable", False))


class ConnectionManager:
    """Manages active WebSocket connections per patient."""

    def __init__(self) -> None:
        self.active_connections: dict[str, set[WebSocket]] = defaultdict(set)
        self._lock = asyncio.Lock()

    async def connect(self, patient_id: str, websocket: WebSocket) -> None:
        await websocket.accept()
        async with self._lock:
            self.active_connections[patient_id].add(websocket)
        logger.info(
            f"WebSocket client connected for patient {patient_id}. Total: {len(self.active_connections[patient_id])}"
        )

    async def disconnect(self, patient_id: str, websocket: WebSocket) -> None:
        async with self._lock:
            if websocket in self.active_connections[patient_id]:
                self.active_connections[patient_id].remove(websocket)
                if not self.active_connections[patient_id]:
                    del self.active_connections[patient_id]
        logger.info(f"WebSocket client disconnected for patient {patient_id}")

    async def broadcast(self, patient_id: str, payload: dict[str, Any]) -> None:
        async with self._lock:
            clients = list(self.active_connections.get(patient_id, []))

        for ws in clients:
            try:
                await ws.send_json(payload)
            except Exception:
                await self.disconnect(patient_id, ws)


ws_manager = ConnectionManager()


async def process_and_broadcast_cgm_reading(reading_data: dict[str, Any]) -> dict[str, Any]:
    """Ingest reading into twin state, evaluate predictions and usable alerts, broadcast to WS."""
    patient_id = str(reading_data.get("patient_id", "DEMO_PAT_001"))
    glucose = float(reading_data.get("glucose_mgdl", 120.0))
    carbs = float(reading_data.get("carbs_g", 0.0))
    mets = float(reading_data.get("mets", 1.0))
    steps = int(reading_data.get("steps", 0))
    hr = float(reading_data.get("heart_rate_bpm", 72.0))

    ts_raw = reading_data.get("timestamp")
    if isinstance(ts_raw, str):
        try:
            ts = datetime.fromisoformat(ts_raw.replace("Z", "+00:00"))
        except ValueError:
            ts = datetime.now(UTC)
    elif isinstance(ts_raw, datetime):
        ts = ts_raw
    else:
        ts = datetime.now(UTC)

    # Ingest into twin state manager
    result = await twin_state_manager.ingest_cgm_reading(
        patient_id=patient_id,
        timestamp=ts,
        glucose_mgdl=glucose,
        heart_rate_bpm=hr,
        steps=steps,
        mets=mets,
        carbs_g=carbs,
    )

    forecast = result["forecast"]
    scored_count = result.get("scored_pending_count", 0)

    # Record Prometheus scored predictions metric
    if scored_count > 0:
        for fp in forecast.forecast_horizons:
            PREDICTIONS_SCORED.labels(
                patient_id=patient_id, horizon_minutes=str(fp.horizon_minutes)
            ).inc(scored_count)

    # Extract horizon predictions
    f_30m, f_60m, f_120m = glucose, glucose, glucose
    active_alerts: list[dict[str, Any]] = []

    for fp in forecast.forecast_horizons:
        h = fp.horizon_minutes
        if h == 30:
            f_30m = fp.q50
        elif h == 60:
            f_60m = fp.q50
        elif h == 120:
            f_120m = fp.q50

        # Check usable alerts: alert sent ONLY if alert_usable is True
        if fp.operating_thresholds:
            for ev_name, th_val in fp.operating_thresholds.items():
                event_key = f"{ev_name}_{h}m"
                usable = is_alert_usable(event_key)

                prob_val = (
                    fp.spike_risk_probability
                    if ev_name == "spike"
                    else (
                        fp.hypo_risk_probability
                        if ev_name == "hypoglycemia"
                        else (
                            fp.severe_hyper_risk_probability
                            if ev_name == "severe_hyperglycemia"
                            else (fp.hyper_risk_probability if ev_name == "hyperglycemia" else 0.0)
                        )
                    )
                )

                if usable and prob_val >= th_val:
                    alert_dict = {
                        "event": event_key,
                        "probability": round(prob_val, 2),
                        "threshold": round(th_val, 2),
                        "horizon_minutes": h,
                        "alert_usable": True,
                        "message": f"Actionable alert: {event_key} triggered (p={prob_val:.2f} >= {th_val:.2f})",
                    }
                    active_alerts.append(alert_dict)
                    ALERTS_FIRED.labels(
                        event_type=event_key, horizon_minutes=str(h), alert_usable="True"
                    ).inc()

    trend = "Rising" if f_30m - glucose > 2.0 else ("Falling" if glucose - f_30m > 2.0 else "Flat")
    rule_alerts_list = [
        a.model_dump() if hasattr(a, "model_dump") else (a.dict() if hasattr(a, "dict") else a)
        for a in getattr(forecast, "rule_based_alerts", [])
    ]
    has_rule_alerts = len(rule_alerts_list) > 0
    has_warning = len(active_alerts) > 0 or has_rule_alerts

    if active_alerts:
        summary_msg = f"Adverse excursion alert: {', '.join([a['event'] for a in active_alerts])}"
    elif has_rule_alerts:
        summary_msg = f"Rule-based safety alert: {rule_alerts_list[0].get('title', 'Observed Low')}"
    else:
        summary_msg = "Telemetry normal"

    broadcast_payload = {
        "patient_id": patient_id,
        "timestamp": ts.isoformat(),
        "published_at": reading_data.get("published_at", reading_data.get("timestamp")),
        "processed_at": datetime.now(UTC).isoformat(),
        "glucose_mgdl": round(glucose, 1),
        "trend_arrow": trend,
        "heart_rate_bpm": round(hr, 1),
        "steps": steps,
        "forecast_30m": round(f_30m, 1),
        "forecast_60m": round(f_60m, 1),
        "forecast_120m": round(f_120m, 1),
        "active_alerts": active_alerts,
        "rule_based_alerts": rule_alerts_list,
        "adverse_event_warning": has_warning,
        "summary_message": summary_msg,
        "is_synthetic": True,
    }

    await ws_manager.broadcast(patient_id, broadcast_payload)
    return broadcast_payload


@router.websocket("/stream/{patient_id}")
async def cgm_websocket_stream(websocket: WebSocket, patient_id: str):
    """WebSocket stream delivering real-time predictions and usable alerts."""
    await ws_manager.connect(patient_id, websocket)

    # Send initial snapshot immediately so client is hydrated upon reconnect
    try:
        init_state = twin_state_manager.get_twin_state(patient_id)
        init_rule_alerts = [
            a.model_dump() if hasattr(a, "model_dump") else (a.dict() if hasattr(a, "dict") else a)
            for a in getattr(init_state, "rule_based_alerts", [])
        ]
        init_payload = {
            "patient_id": patient_id,
            "timestamp": datetime.now(UTC).isoformat(),
            "glucose_mgdl": init_state.current_glucose,
            "trend_arrow": init_state.trend_arrow,
            "heart_rate_bpm": 72.0,
            "steps": 0,
            "forecast_30m": round(init_state.current_glucose + 2.0, 1),
            "forecast_60m": round(init_state.current_glucose + 4.0, 1),
            "forecast_120m": round(init_state.current_glucose + 1.0, 1),
            "active_alerts": [],
            "rule_based_alerts": init_rule_alerts,
            "adverse_event_warning": len(init_rule_alerts) > 0,
            "summary_message": (
                f"Rule-based safety alert: {init_rule_alerts[0].get('title', 'Observed Low')}"
                if init_rule_alerts
                else "Stream connected - awaiting telemetry ticks"
            ),
            "is_synthetic": True,
        }
        await websocket.send_json(init_payload)
    except Exception as exc:
        logger.warning(f"Failed to send initial snapshot to {patient_id}: {exc}")

    # Fallback simulation generator if Kafka is not active
    fallback_task = None
    if not kafka_manager.is_connected:

        async def fallback_stream():
            from src.glucotwin.synth.personas import get_persona_by_id

            persona = get_persona_by_id(patient_id)
            current_g = persona.fasting_glucose if persona else 125.0
            while True:
                await asyncio.sleep(2.0)
                import random

                delta = random.uniform(-2.5, 3.5)
                current_g = max(55.0, min(360.0, current_g + delta))
                tick = {
                    "patient_id": patient_id,
                    "timestamp": datetime.now(UTC).isoformat(),
                    "glucose_mgdl": round(current_g, 1),
                    "heart_rate_bpm": 74.0,
                    "steps": 15,
                    "mets": 1.1,
                    "carbs_g": 0.0,
                }
                await process_and_broadcast_cgm_reading(tick)

        fallback_task = asyncio.create_task(fallback_stream())

    try:
        # Keep socket open and handle incoming messages (e.g. heartbeat pings)
        while True:
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        await ws_manager.disconnect(patient_id, websocket)
    except Exception:
        await ws_manager.disconnect(patient_id, websocket)
    finally:
        if fallback_task:
            fallback_task.cancel()
