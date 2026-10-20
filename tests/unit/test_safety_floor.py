"""Unit tests for rule-based safety floor alerts.

Verifies deterministic safety floor rules:
1. Observed low (<70 mg/dL)
2. Urgent low (<54 mg/dL)
3. Rapid fall (drop >= 2 mg/dL/min sustained over 15 min while below 120 mg/dL)
All alerts must carry provenance: 'Rule-based alert (observed, not predicted)'.
"""

from src.glucotwin.alerts.safety_floor import (
    evaluate_safety_floor_rules,
    load_safety_floor_params,
)


def test_load_safety_floor_params():
    params = load_safety_floor_params()
    assert isinstance(params, dict)
    assert params["low_threshold_mgdl"] == 70.0
    assert params["urgent_low_threshold_mgdl"] == 54.0
    assert params["rapid_fall_rate_mgdl_per_min"] == 2.0
    assert params["rapid_fall_duration_minutes"] == 15
    assert params["rapid_fall_ceiling_mgdl"] == 120.0


def test_normal_in_range_produces_no_alerts():
    readings = [110.0, 112.0, 115.0, 113.0, 114.0]
    alerts = evaluate_safety_floor_rules(readings)
    assert alerts == []


def test_observed_low_alert():
    readings = [85.0, 80.0, 75.0, 66.0]
    alerts = evaluate_safety_floor_rules(readings)
    assert len(alerts) == 1
    alert = alerts[0]
    assert alert["alert_type"] == "low"
    assert alert["severity"] == "warning"
    assert alert["observed_glucose"] == 66.0
    assert alert["threshold_mgdl"] == 70.0
    assert alert["is_rule_based"] is True
    assert alert["provenance"] == "Rule-based alert (observed, not predicted)"


def test_urgent_low_alert():
    readings = [75.0, 70.0, 60.0, 50.0]
    alerts = evaluate_safety_floor_rules(readings)
    # 50.0 is urgent low; drop is 75->50 = 25 in 15 min (rate 1.67 < 2.0, so no rapid fall)
    assert len(alerts) == 1
    alert = alerts[0]
    assert alert["alert_type"] == "urgent_low"
    assert alert["severity"] == "critical"
    assert alert["observed_glucose"] == 50.0
    assert alert["threshold_mgdl"] == 54.0
    assert alert["is_rule_based"] is True
    assert alert["provenance"] == "Rule-based alert (observed, not predicted)"


def test_rapid_fall_alert():
    # 4 readings spanning 15 minutes: [115, 104, 92, 80]
    # Total drop = 35 mg/dL / 15 min = 2.33 mg/dL/min >= 2.0
    # Current glucose = 80 < 120 ceiling, sustained monotonic drop
    # Current glucose 80 is not low (<70), so only rapid fall fires
    readings = [115.0, 104.0, 92.0, 80.0]
    alerts = evaluate_safety_floor_rules(readings)
    assert len(alerts) == 1
    alert = alerts[0]
    assert alert["alert_type"] == "rapid_fall"
    assert alert["severity"] == "warning"
    assert alert["rate_mgdl_per_min"] == 2.33
    assert alert["is_rule_based"] is True
    assert alert["provenance"] == "Rule-based alert (observed, not predicted)"


def test_rapid_fall_above_ceiling_suppressed():
    # Drop is 40 mg/dL over 15 min (rate 2.67), but ending at 130 mg/dL (> 120)
    readings = [170.0, 155.0, 142.0, 130.0]
    alerts = evaluate_safety_floor_rules(readings)
    assert alerts == []


def test_rapid_fall_non_monotonic_suppressed():
    # Total drop is 35 mg/dL over 15 min below 120, but not sustained (bounces up at step 2)
    readings = [115.0, 95.0, 102.0, 80.0]
    alerts = evaluate_safety_floor_rules(readings)
    assert alerts == []


def test_simultaneous_low_and_rapid_fall():
    # Drop from 110 to 65: drop is 45 mg/dL in 15 min (3.0 mg/dL/min)
    # Monotonic drop, below 120, and current glucose 65 < 70
    readings = [110.0, 95.0, 80.0, 65.0]
    alerts = evaluate_safety_floor_rules(readings)
    assert len(alerts) == 2
    types = [a["alert_type"] for a in alerts]
    assert "low" in types
    assert "rapid_fall" in types


def test_empty_and_short_series_graceful():
    assert evaluate_safety_floor_rules([]) == []
    assert evaluate_safety_floor_rules([None, None]) == []
    assert evaluate_safety_floor_rules([100.0]) == []
