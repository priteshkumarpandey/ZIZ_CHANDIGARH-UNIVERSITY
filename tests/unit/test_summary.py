from datetime import datetime, timedelta

import numpy as np

from src.glucotwin.serving.schemas.fhir import CGMReading
from src.glucotwin.summary.generator import ClinicalSummaryGenerator


def test_handbuilt_series_metrics_exact_match():
    # Hand-built series with known analytic quantities: [50.0, 100.0, 200.0, 300.0]
    vals = [50.0, 100.0, 200.0, 300.0]
    cgm_readings = [
        CGMReading(patient_id="IND_PAT_001", timestamp="2026-10-10T12:00:00", glucose_mgdl=v)
        for v in vals
    ]

    summary = ClinicalSummaryGenerator.generate(
        patient_id="IND_PAT_001", window_hours=1, recent_cgm=cgm_readings
    )
    m = summary.metrics

    # 1. Mean: (50 + 100 + 200 + 300) / 4 = 162.5 mg/dL
    assert m.mean_glucose == 162.5

    # 2. Standard deviation: sqrt(9218.75) = 96.01432 -> round 96.0
    expected_sd = float(np.std(vals))
    assert m.standard_deviation == round(expected_sd, 1)

    # 3. Coefficient of Variation: (96.01432 / 162.5) * 100 = 59.0857% -> round 59.1%
    expected_cv = (expected_sd / 162.5) * 100.0
    assert m.coefficient_of_variation == round(expected_cv, 1)

    # 4. GMI: 3.31 + 0.02392 * 162.5 = 7.197 -> round 7.2
    expected_gmi = 3.31 + (0.02392 * 162.5)
    assert m.gmi == round(expected_gmi, 2)

    # 5. Time in Range (70-180 mg/dL): 100.0 is in range -> 1/4 = 25.0%
    assert m.time_in_range_percentage == 25.0

    # 6. Time Above Range (>180 mg/dL): 200.0, 300.0 -> 2/4 = 50.0%
    assert m.time_above_range_percentage == 50.0

    # 7. Time Above Range Level 2 (>250 mg/dL): 300.0 -> 1/4 = 25.0%
    assert m.time_above_range_level2_percentage == 25.0

    # 8. Time Below Range (<70 mg/dL): 50.0 -> 1/4 = 25.0%
    assert m.time_below_range_percentage == 25.0

    # 9. Time Below Range Level 2 (<54 mg/dL): 50.0 -> 1/4 = 25.0%
    assert m.time_below_range_level2_percentage == 25.0


def test_summary_real_window_and_no_em_dash():
    summary = ClinicalSummaryGenerator.generate(patient_id="IND_PAT_001", window_hours=24)
    assert summary.patient_id == "IND_PAT_001"
    assert summary.metrics.time_in_range_percentage >= 0.0
    assert isinstance(summary.actionable_recommendations, list)
    assert "calibrated" in summary.personalization_status
    assert "Decision support only" in summary.disclaimer

    # Absolutely NO em dashes anywhere in narrative or findings
    assert "\u2014" not in summary.plain_language_narrative
    for finding in summary.key_findings:
        assert "\u2014" not in finding
    for rec in summary.actionable_recommendations:
        assert "\u2014" not in rec

    # Verify gated usable alerts
    if summary.adverse_events_usable:
        for _horizon, events in summary.adverse_events_usable.items():
            for _ev_name, info in events.items():
                assert info["actionability"] == "actionable alert threshold"


def test_demo_patient_summary_matches_displayed_series_and_no_matched_participant():
    summary = ClinicalSummaryGenerator.generate(patient_id="DEMO_PAT_001", window_hours=24)
    series, persona = ClinicalSummaryGenerator._fetch_patient_series(
        patient_id="DEMO_PAT_001", window_hours=24
    )

    # 1. Assert series has non-zero variability (not flat baseline)
    assert len(series) > 0
    assert float(np.std(series)) > 0.1, (
        "Summary must be computed on active dynamic series, not flat baseline"
    )

    # 2. Assert metrics exactly equal metrics recomputed directly from the series
    expected_mean = round(float(np.mean(series)), 1)
    expected_sd = round(float(np.std(series)), 1)
    expected_cv = round(float((np.std(series) / np.mean(series)) * 100.0), 1)
    expected_tir = round(float(np.mean((series >= 70.0) & (series <= 180.0)) * 100.0), 1)

    assert summary.metrics.mean_glucose == expected_mean
    assert summary.metrics.standard_deviation == expected_sd
    assert summary.metrics.coefficient_of_variation == expected_cv
    assert summary.metrics.time_in_range_percentage == expected_tir

    # 3. Assert no matched participant phrase or CGMacros reference for synthetic demo patient
    assert "matched participant" not in summary.plain_language_narrative.lower()
    assert "cgmacros" not in summary.plain_language_narrative.lower()
    assert "cgmacros" not in (summary.personalization_status or "").lower()
    assert "24h" in summary.plain_language_narrative


def test_nocturnal_hypoglycemia_rule_trigger():
    # 1. Series with nocturnal low at 03:00 (glucose = 62.0 mg/dL)
    night_readings = [
        CGMReading(patient_id="IND_PAT_001", timestamp="2026-10-10T01:00:00", glucose_mgdl=110.0),
        CGMReading(patient_id="IND_PAT_001", timestamp="2026-10-10T02:00:00", glucose_mgdl=105.0),
        CGMReading(patient_id="IND_PAT_001", timestamp="2026-10-10T03:00:00", glucose_mgdl=62.0),
        CGMReading(patient_id="IND_PAT_001", timestamp="2026-10-10T04:00:00", glucose_mgdl=65.0),
        CGMReading(patient_id="IND_PAT_001", timestamp="2026-10-10T08:00:00", glucose_mgdl=115.0),
    ]
    summary_with_low = ClinicalSummaryGenerator.generate(
        patient_id="IND_PAT_001", window_hours=8, recent_cgm=night_readings
    )
    assert any(
        "nocturnal hypoglycemia" in r.lower() for r in summary_with_low.actionable_recommendations
    )

    # 2. Series without nocturnal low (all in-range between 100 and 120 mg/dL)
    normal_night_readings = [
        CGMReading(patient_id="IND_PAT_001", timestamp="2026-10-10T01:00:00", glucose_mgdl=110.0),
        CGMReading(patient_id="IND_PAT_001", timestamp="2026-10-10T02:00:00", glucose_mgdl=105.0),
        CGMReading(patient_id="IND_PAT_001", timestamp="2026-10-10T03:00:00", glucose_mgdl=112.0),
        CGMReading(patient_id="IND_PAT_001", timestamp="2026-10-10T04:00:00", glucose_mgdl=115.0),
        CGMReading(patient_id="IND_PAT_001", timestamp="2026-10-10T08:00:00", glucose_mgdl=115.0),
    ]
    summary_without_low = ClinicalSummaryGenerator.generate(
        patient_id="IND_PAT_001", window_hours=8, recent_cgm=normal_night_readings
    )
    assert not any(
        "nocturnal hypoglycemia" in r.lower()
        for r in summary_without_low.actionable_recommendations
    )


def test_walk_advice_rule_trigger():
    # 1. Series with significant postprandial rise (120 -> 195 mg/dL) and TAR > 15%
    base_t = datetime(2026, 10, 10, 12, 0)
    high_readings = [
        CGMReading(
            patient_id="IND_PAT_001",
            timestamp=(base_t + timedelta(minutes=5 * i)).isoformat(),
            glucose_mgdl=g,
        )
        for i, g in enumerate([120.0, 130.0, 150.0, 175.0, 195.0, 205.0, 190.0, 170.0])
    ]
    summary_high = ClinicalSummaryGenerator.generate(
        patient_id="IND_PAT_001", window_hours=2, recent_cgm=high_readings
    )
    assert any("brisk walk" in r.lower() for r in summary_high.actionable_recommendations)

    # 2. Series with flat target glucose (110.0 mg/dL throughout)
    flat_readings = [
        CGMReading(
            patient_id="IND_PAT_001",
            timestamp=(base_t + timedelta(minutes=5 * i)).isoformat(),
            glucose_mgdl=110.0,
        )
        for i in range(12)
    ]
    summary_flat = ClinicalSummaryGenerator.generate(
        patient_id="IND_PAT_001", window_hours=1, recent_cgm=flat_readings
    )
    assert not any("brisk walk" in r.lower() for r in summary_flat.actionable_recommendations)


def test_no_data_driven_suggestions_when_all_in_range():
    # Patient with excellent control throughout the window (all 115 mg/dL)
    base_t = datetime(2026, 10, 10, 12, 0)
    perfect_readings = [
        CGMReading(
            patient_id="IND_PAT_001",
            timestamp=(base_t + timedelta(minutes=5 * i)).isoformat(),
            glucose_mgdl=115.0,
        )
        for i in range(24)
    ]
    summary = ClinicalSummaryGenerator.generate(
        patient_id="IND_PAT_001", window_hours=2, recent_cgm=perfect_readings
    )
    assert len(summary.actionable_recommendations) == 0
