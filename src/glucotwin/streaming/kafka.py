"""Kafka streaming integration for GlucoTwin telemetry.

Manages Kafka producers and consumers for topics: cgm, activity, meals.
Handles duplicate detection, out-of-order ingestion, gap detection,
and automatic reconnects with exponential backoff.
"""

from __future__ import annotations

import asyncio
import json
from collections import defaultdict
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from src.glucotwin.config.settings import settings
from src.glucotwin.serving.metrics import STREAM_LAG
from src.glucotwin.utils.logging import logger

try:
    from aiokafka import AIOKafkaConsumer, AIOKafkaProducer

    HAS_AIOKAFKA = True
except ImportError:
    HAS_AIOKAFKA = False
    AIOKafkaConsumer = None  # type: ignore
    AIOKafkaProducer = None  # type: ignore


class KafkaStreamManager:
    """Manages publishing and consuming telemetry streams over Kafka."""

    def __init__(self, bootstrap_servers: str | None = None) -> None:
        self.bootstrap_servers = bootstrap_servers or settings.KAFKA_BOOTSTRAP_SERVERS
        self.producer: Any = None
        self.consumer: Any = None
        self.is_connected = False
        self._consumer_task: asyncio.Task | None = None
        self._running = False

        # Ingestion state tracking per patient for duplicates, order, and gaps
        self.last_timestamp: dict[str, datetime] = {}
        self.seen_signatures: dict[str, set[str]] = defaultdict(set)
        self.cgm_callback: Callable[[dict[str, Any]], Any] | None = None

    async def connect_producer(self) -> bool:
        """Connect Kafka producer with graceful fallback."""
        if not HAS_AIOKAFKA:
            logger.warning("aiokafka not installed; Kafka stream manager running in offline mode.")
            return False

        try:
            self.producer = AIOKafkaProducer(bootstrap_servers=self.bootstrap_servers)
            await self.producer.start()
            self.is_connected = True
            logger.info(f"Connected Kafka producer to broker at {self.bootstrap_servers}")
            return True
        except Exception as exc:
            logger.warning(f"Kafka producer connection skipped (offline fallback): {exc}")
            self.is_connected = False
            self.producer = None
            return False

    async def publish(self, topic: str, key: str, payload: dict[str, Any]) -> bool:
        """Publish an event to a specified Kafka topic."""
        if not self.is_connected or not self.producer:
            return False
        try:
            raw_payload = json.dumps(payload, default=str).encode("utf-8")
            await self.producer.send_and_wait(
                topic,
                raw_payload,
                key=key.encode("utf-8"),
            )
            return True
        except Exception as exc:
            logger.warning(f"Failed to publish to topic {topic}: {exc}")
            return False

    async def publish_cgm(self, patient_id: str, reading: dict[str, Any]) -> bool:
        return await self.publish(settings.KAFKA_CGM_TOPIC, patient_id, reading)

    async def publish_activity(self, patient_id: str, activity: dict[str, Any]) -> bool:
        return await self.publish(settings.KAFKA_ACTIVITY_TOPIC, patient_id, activity)

    async def publish_meal(self, patient_id: str, meal: dict[str, Any]) -> bool:
        return await self.publish(settings.KAFKA_MEALS_TOPIC, patient_id, meal)

    async def start_consumer(self, callback: Callable[[dict[str, Any]], Any]) -> None:
        """Start background consumer loop across cgm, activity, and meals topics."""
        if not HAS_AIOKAFKA:
            return
        self.cgm_callback = callback
        self._running = True
        self._consumer_task = asyncio.create_task(self._consume_loop())

    async def _consume_loop(self) -> None:
        """Consumer execution loop with automatic reconnect and error resilience."""
        retry_delay = 2.0
        topics = [
            settings.KAFKA_CGM_TOPIC,
            settings.KAFKA_ACTIVITY_TOPIC,
            settings.KAFKA_MEALS_TOPIC,
        ]

        while self._running:
            try:
                self.consumer = AIOKafkaConsumer(
                    *topics,
                    bootstrap_servers=self.bootstrap_servers,
                    group_id=settings.KAFKA_CONSUMER_GROUP,
                    auto_offset_reset="latest",
                    enable_auto_commit=True,
                )
                await self.consumer.start()
                logger.info(f"Kafka consumer started on topics: {topics}")
                retry_delay = 2.0

                async for msg in self.consumer:
                    if not self._running:
                        break
                    try:
                        data = json.loads(msg.value.decode("utf-8"))
                        topic = msg.topic
                        await self._process_incoming_message(topic, data)
                    except Exception as err:
                        logger.error(f"Error processing Kafka message on topic {msg.topic}: {err}")

            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.warning(
                    f"Kafka consumer connection lost: {exc}. Reconnecting in {retry_delay:.1f}s..."
                )
                await asyncio.sleep(retry_delay)
                retry_delay = min(30.0, retry_delay * 1.5)
            finally:
                if self.consumer:
                    try:
                        await self.consumer.stop()
                    except Exception:
                        pass
                    self.consumer = None

    async def _process_incoming_message(self, topic: str, data: dict[str, Any]) -> None:
        """Process incoming message with deduplication, order verification, and gap detection."""
        patient_id = str(data.get("patient_id", "UNKNOWN"))
        ts_raw = data.get("timestamp")
        if not ts_raw:
            return

        # Parse timestamp safely
        if isinstance(ts_raw, str):
            try:
                event_dt = datetime.fromisoformat(ts_raw.replace("Z", "+00:00"))
            except ValueError:
                event_dt = datetime.now(UTC)
        elif isinstance(ts_raw, datetime):
            event_dt = ts_raw
        else:
            event_dt = datetime.now(UTC)

        # 1. Deduplication check
        sig = (
            f"{topic}:{ts_raw}:{data.get('glucose_mgdl', data.get('carbs_g', data.get('mets', 0)))}"
        )
        if sig in self.seen_signatures[patient_id]:
            logger.debug(f"Duplicate reading discarded for {patient_id}: {sig}")
            return
        self.seen_signatures[patient_id].add(sig)
        # Prevent memory unbounded growth
        if len(self.seen_signatures[patient_id]) > 5000:
            self.seen_signatures[patient_id].clear()

        # 2. Out-of-order and gap detection
        last_dt = self.last_timestamp.get(patient_id)
        if last_dt is not None:
            # Check out of order
            if event_dt < last_dt:
                logger.warning(
                    f"Out-of-order event for {patient_id}: current {event_dt} < last {last_dt}"
                )
            # Check gap (>15 minutes between consecutive 5-min readings)
            gap_seconds = (event_dt - last_dt).total_seconds()
            if gap_seconds > 900.0:
                logger.info(
                    f"Telemetry stream gap detected for {patient_id}: {gap_seconds / 60.0:.1f} minutes"
                )

        self.last_timestamp[patient_id] = event_dt

        # 3. Stream lag observation
        try:
            now_utc = datetime.now(UTC)
            if event_dt.tzinfo is None:
                event_dt = event_dt.replace(tzinfo=UTC)
            lag = (now_utc - event_dt).total_seconds()
            STREAM_LAG.labels(patient_id=patient_id).observe(max(0.0, lag))
        except Exception:
            pass

        # 4. Invoke CGM ingestion callback if CGM reading
        if topic == settings.KAFKA_CGM_TOPIC and self.cgm_callback:
            await self.cgm_callback(data)

    async def disconnect(self) -> None:
        """Disconnect producer and consumer."""
        self._running = False
        if self._consumer_task:
            self._consumer_task.cancel()
            try:
                await self._consumer_task
            except asyncio.CancelledError:
                pass
        if self.producer and self.is_connected:
            await self.producer.stop()
            self.is_connected = False
            self.producer = None
        if self.consumer:
            await self.consumer.stop()
            self.consumer = None


kafka_manager = KafkaStreamManager()
