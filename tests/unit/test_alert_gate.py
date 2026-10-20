import json

from src.glucotwin.serving.schemas.fhir import AdverseEventType
from src.glucotwin.serving.schemas.requests import PredictRequest
from src.glucotwin.serving.service import inference_service


def test_operating_thresholds_gate_definition():
    with open("reports/operating_thresholds.json") as fp:
        report = json.load(fp)
    events = report["events"]

    # Verify that each event satisfies the usefulness criteria:
    # alert_usable == True iff mean_recall >= 0.40, mean_precision >= 0.30, and is_reliable == True
    for ev_name, data in events.items():
        usable = data.get("alert_usable")
        rec = data.get("recall_mean", 0.0)
        prec = data.get("precision_mean", 0.0)
        rel = data.get("is_reliable", False)

        expected = bool(rec >= 0.40 and prec >= 0.30 and rel is True)
        assert usable == expected, (
            f"{ev_name}: usable is {usable} but expected {expected} (rec={rec}, prec={prec}, rel={rel})"
        )


def test_inference_service_never_alerts_on_unusable_events():
    inference_service.load()

    # Extreme low glucose scenario to test hypoglycemia gate
    # Hypoglycemia at 30m, 60m, 120m is NOT usable (low precision/count)
    low_series = [50.0] * 72
    req_low = PredictRequest(patient_id="IND_PAT_001", recent_cgm_series=low_series)
    res_low = inference_service.predict(req_low)

    for fp in res_low.forecast_horizons:
        # Hypoglycemia should have alert_usable == False across all horizons
        assert fp.alert_usable["hypoglycemia"] is False
        assert fp.risk_labels["hypoglycemia"] == "risk estimate only"
        # Even with low glucose, unusable event must never trigger an alert
        assert fp.predicted_event not in [
            AdverseEventType.HYPOGLYCEMIA_LEVEL_1,
            AdverseEventType.HYPOGLYCEMIA_LEVEL_2,
        ]

    # Test +60m hyperglycemia (recall is 0.06 < 0.40, unusable)
    # A moderately elevated CGM should NOT trigger hyperglycemia alert at +60m
    high_series = [185.0] * 72
    req_high = PredictRequest(patient_id="IND_PAT_001", recent_cgm_series=high_series)
    res_high = inference_service.predict(req_high)

    fp_60 = next(fp for fp in res_high.forecast_horizons if fp.horizon_minutes == 60)
    assert fp_60.alert_usable["hyperglycemia"] is False
    assert fp_60.risk_labels["hyperglycemia"] == "risk estimate only"

    # But at +30m, hyperglycemia is usable (recall 0.65, prec 0.61)
    fp_30 = next(fp for fp in res_high.forecast_horizons if fp.horizon_minutes == 30)
    assert fp_30.alert_usable["hyperglycemia"] is True
    assert fp_30.risk_labels["hyperglycemia"] == "actionable alert threshold"


def test_all_12_events_alert_usable_parity():
    with open("reports/operating_thresholds.json") as fp:
        report = json.load(fp)
    events = report["events"]
    assert len(events) == 12, f"Expected 12 event-horizon combinations, found {len(events)}"

    expected_events = [
        "hyperglycemia_30m",
        "hyperglycemia_60m",
        "hyperglycemia_120m",
        "severe_hyperglycemia_30m",
        "severe_hyperglycemia_60m",
        "severe_hyperglycemia_120m",
        "hypoglycemia_30m",
        "hypoglycemia_60m",
        "hypoglycemia_120m",
        "spike_30m",
        "spike_60m",
        "spike_120m",
    ]
    for ev in expected_events:
        assert ev in events, f"Missing event: {ev}"
        assert "alert_usable" in events[ev]

    usable_set = {ev for ev, data in events.items() if data.get("alert_usable") is True}
    expected_usable = {
        "hyperglycemia_30m",
        "severe_hyperglycemia_30m",
        "severe_hyperglycemia_60m",
        "spike_30m",
        "spike_60m",
    }
    assert usable_set == expected_usable, (
        f"Usable events mismatch: found {usable_set}, expected {expected_usable}"
    )


def test_recompute_alert_usable_from_printed_rule_for_every_row():
    with open("reports/operating_thresholds.json") as fp:
        report = json.load(fp)

    gate_rule = report["gate_rule"]
    min_rec = gate_rule["min_recall"]
    min_prec = gate_rule["min_precision"]
    req_reliable = gate_rule["require_reliable"]
    fa_budget = gate_rule["false_alert_budget"]

    events = report["events"]
    assert len(events) == 12

    for ev_name, data in events.items():
        rec = data.get("recall_mean", 0.0)
        prec = data.get("precision_mean", 0.0)
        rel = data.get("is_reliable", False)
        fa = data.get("false_alerts_per_patient_day_mean", 999.0)

        rec_pass = rec >= min_rec
        prec_pass = prec >= min_prec
        rel_pass = rel is True or not req_reliable
        fa_pass = fa <= fa_budget

        recomputed = bool(rec_pass and prec_pass and rel_pass and fa_pass)
        assert data.get("alert_usable") == recomputed, (
            f"Row {ev_name}: alert_usable is {data.get('alert_usable')} but recomputed from printed rule is {recomputed} "
            f"(rec={rec} vs {min_rec}, prec={prec} vs {min_prec}, rel={rel}, fa={fa} vs {fa_budget})"
        )

    # Explicitly confirm that Severe Hyperglycemia +60m and Rapid Spike +30m are usable
    assert events["severe_hyperglycemia_60m"]["alert_usable"] is True
    assert events["spike_30m"]["alert_usable"] is True
