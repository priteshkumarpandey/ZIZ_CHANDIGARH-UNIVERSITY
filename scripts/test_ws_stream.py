"""Helper script to test WebSocket telemetry streaming, count ticks, and measure e2e latency."""

import asyncio
from datetime import datetime, timezone
import json
import sys
import time

try:
    import numpy as np
except ImportError:
    np = None

try:
    import websockets
except ImportError:
    print("websockets library not installed in environment", file=sys.stderr)
    sys.exit(1)


async def test_websocket_stream(
    patient_id: str = "DEMO_PAT_001",
    duration_seconds: int = 45,
    min_ticks: int = 5,
    target_measurements: int = 20,
):
    uri = f"ws://localhost:8000/api/v1/stream/{patient_id}"
    print(f"Connecting to WebSocket: {uri}")

    start_time = time.time()
    received_count = 0
    predictions_received = 0
    e2e_latencies_ms = []

    try:
        async with websockets.connect(uri, ping_interval=10, ping_timeout=10) as ws:
            print("Connected successfully. Listening for telemetry ticks...")
            while (time.time() - start_time < duration_seconds) and (
                len(e2e_latencies_ms) < target_measurements or received_count < min_ticks
            ):
                remaining = duration_seconds - (time.time() - start_time)
                if remaining <= 0 and received_count >= min_ticks:
                    break
                try:
                    msg = await asyncio.wait_for(
                        ws.recv(), timeout=min(6.0, max(0.5, remaining if remaining > 0 else 2.0))
                    )
                    receipt_time = time.time()
                    data = json.loads(msg)
                    received_count += 1

                    g_val = data.get("glucose_mgdl")
                    f_30m = data.get("forecast_30m")
                    f_60m = data.get("forecast_60m")
                    pub_str = data.get("published_at")

                    if pub_str:
                        try:
                            pub_dt = datetime.fromisoformat(pub_str.replace("Z", "+00:00"))
                            lat_ms = (receipt_time - pub_dt.timestamp()) * 1000.0
                            if lat_ms >= 0:
                                e2e_latencies_ms.append(lat_ms)
                        except Exception:
                            pass

                    if f_30m is not None and f_60m is not None:
                        predictions_received += 1

                    lat_display = f"{e2e_latencies_ms[-1]:.1f} ms" if e2e_latencies_ms else "n/a"
                    print(
                        f"[{received_count}] Tick: Glucose={g_val} mg/dL, +30m={f_30m}, +60m={f_60m} | Latency={lat_display}"
                    )
                except asyncio.TimeoutError:
                    pass

    except Exception as exc:
        print(f"WebSocket connection error: {exc}", file=sys.stderr)
        return False

    elapsed = time.time() - start_time
    print(f"\n--- WebSocket Test Summary ---")
    print(f"Total Elapsed: {elapsed:.2f}s")
    print(f"Total Ticks Received: {received_count}")
    print(f"Predictions Received: {predictions_received}")
    print(f"Measured Messages: {len(e2e_latencies_ms)}")

    if len(e2e_latencies_ms) > 0:
        if np is not None:
            med_lat = float(np.median(e2e_latencies_ms))
            p95_lat = float(np.percentile(e2e_latencies_ms, 95))
        else:
            sorted_lat = sorted(e2e_latencies_ms)
            med_lat = sorted_lat[len(sorted_lat) // 2]
            p95_lat = sorted_lat[int(len(sorted_lat) * 0.95)]
        print(f"End-to-End Latency: Median={med_lat:.2f} ms, P95={p95_lat:.2f} ms")

    if received_count >= min_ticks:
        print(f"PASS: Received {received_count} ticks (>= {min_ticks} required).")
        return True
    else:
        print(
            f"FAIL: Only {received_count} ticks received (< {min_ticks} required).", file=sys.stderr
        )
        return False


if __name__ == "__main__":
    patient = sys.argv[1] if len(sys.argv) > 1 else "DEMO_PAT_001"
    duration = int(sys.argv[2]) if len(sys.argv) > 2 else 45
    min_t = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    target_m = int(sys.argv[4]) if len(sys.argv) > 4 else 20
    success = asyncio.run(test_websocket_stream(patient, duration, min_t, target_m))
    sys.exit(0 if success else 1)
