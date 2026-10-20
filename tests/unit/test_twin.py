from src.glucotwin.serving.schemas.requests import InterventionItem
from src.glucotwin.twin.simulator import CounterfactualTwinSimulator, run_twin_simulation
from src.glucotwin.twin.state import twin_state_manager


def test_twin_state():
    state = twin_state_manager.get_twin_state("IND_PAT_001")
    assert state.patient.name == "Rajesh Sharma"
    assert state.current_glucose > 0


def test_simulator_zero_meal_invariance():
    # Sanity test: 0 g meal produces exactly 0 mg/dL change across all horizons
    res = run_twin_simulation(
        patient_id="IND_PAT_001",
        initial_glucose=130.0,
        interventions=[InterventionItem(carbs_g=0.0)],
    )
    assert res.peak_change_mgdl == 0.0
    for h in ["+30m", "+60m", "+120m"]:
        assert res.deltas[h] == 0.0
    assert res.baseline_peak_glucose == res.scenario_peak_glucose
    assert res.constraint_applied is False
    assert res.constraint_correction_mgdl == 0.0


def test_simulator_5g_versus_50g():
    # Verify that a 5 g meal produces smaller peak excursion than a 50 g meal, both positive and strictly monotonic
    res_5 = run_twin_simulation(
        patient_id="IND_PAT_001",
        initial_glucose=130.0,
        interventions=[InterventionItem(carbs_g=5.0)],
    )
    res_50 = run_twin_simulation(
        patient_id="IND_PAT_001",
        initial_glucose=130.0,
        interventions=[InterventionItem(carbs_g=50.0)],
    )
    assert res_5.peak_change_mgdl > 0.0
    assert res_50.peak_change_mgdl > res_5.peak_change_mgdl
    assert res_5.peak_change_mgdl < 10.0
    assert res_50.peak_change_mgdl > 20.0

    # With optional calibration, 50 g excursion scales higher (>30 mg/dL)
    res_50_cal = run_twin_simulation(
        patient_id="IND_PAT_001",
        initial_glucose=130.0,
        interventions=[InterventionItem(carbs_g=50.0)],
        calibrate_meal_response=True,
    )
    assert res_50_cal.peak_change_mgdl > 30.0
    assert res_50_cal.peak_change_mgdl > res_50.peak_change_mgdl


def test_isotonic_projection_handbuilt_curves():
    # Hand-built non-monotonic peak series with known dips
    baseline_peak = 120.0
    # Dips at index 1 (130 < 135) and index 3 (140 < 145)
    raw_peaks = [135.0, 130.0, 145.0, 140.0, 150.0]

    iso_peaks, applied, total_corr = CounterfactualTwinSimulator.project_isotonic_trajectory(
        raw_peaks, baseline_peak
    )

    # 1. Correctness: non-decreasing monotonicity strictly restored
    assert applied is True
    assert iso_peaks == [135.0, 135.0, 145.0, 145.0, 150.0]
    for i in range(len(iso_peaks) - 1):
        assert iso_peaks[i + 1] >= iso_peaks[i]

    # 2. Correction size: (135 - 130) + (145 - 140) = 10.0 mg/dL
    assert total_corr == 10.0

    # 3. Already monotonic series produces zero correction
    mono_peaks = [125.0, 135.0, 145.0]
    iso_clean, applied_clean, corr_clean = CounterfactualTwinSimulator.project_isotonic_trajectory(
        mono_peaks, baseline_peak
    )
    assert applied_clean is False
    assert iso_clean == mono_peaks
    assert corr_clean == 0.0


def test_simulator_constraint_flag():
    # Verify single-run simulation reports constraint_applied and constraint_correction_mgdl
    res = run_twin_simulation(
        patient_id="IND_PAT_001",
        initial_glucose=130.0,
        interventions=[InterventionItem(carbs_g=40.0)],
    )
    assert hasattr(res, "constraint_applied")
    assert hasattr(res, "constraint_correction_mgdl")
    assert isinstance(res.constraint_applied, bool)
    assert isinstance(res.constraint_correction_mgdl, float)
    assert res.constraint_correction_mgdl >= 0.0


def test_simulator_clipping_bounds():
    # Extreme inputs are clipped strictly within [40.0, 400.0] mg/dL
    # High test
    res_high = run_twin_simulation(
        patient_id="IND_PAT_001",
        initial_glucose=450.0,
        interventions=[InterventionItem(carbs_g=250.0)],
    )
    for v in res_high.scenario_trajectory:
        assert 40.0 <= v <= 400.0
    for _h, vals in res_high.scenario_horizons.items():
        for _q, val in vals.items():
            assert 40.0 <= val <= 400.0

    # Low test
    res_low = run_twin_simulation(
        patient_id="IND_PAT_001",
        initial_glucose=30.0,
        interventions=[
            InterventionItem(intervention_type="activity", mets=10.0, duration_minutes=120)
        ],
    )
    for v in res_low.scenario_trajectory:
        assert 40.0 <= v <= 400.0
    for _h, vals in res_low.scenario_horizons.items():
        for _q, val in vals.items():
            assert 40.0 <= val <= 400.0


def test_simulator_provenance_and_food_lookup():
    res = run_twin_simulation(
        patient_id="IND_PAT_001",
        initial_glucose=130.0,
        interventions=[
            InterventionItem(food_name="Steamed White Rice (Sona Masoori)"),
            InterventionItem(
                intervention_type="activity", activity_type="walking", mets=3.5, duration_minutes=30
            ),
        ],
    )
    prov = res.provenance
    assert prov["meal_response"] == "model_based"
    assert "assumption_based" in prov["activity_clearance"]
    assert "unvalidated" in prov["activity_assumptions"]["status"]
    assert "origin_of_coefficient" in prov["activity_assumptions"]
    assert prov["activity_assumptions"]["clearance_rate_mgdl_per_min"] == 0.38
    assert prov["activity_assumptions"]["maximum_clearance_mgdl"] == 50.0
    assert "illustrative assumption" in prov["activity_assumptions"]["origin_of_coefficient"]
    assert "Colberg" not in prov["activity_assumptions"]["origin_of_coefficient"]
    assert "Riddell" not in prov["activity_assumptions"]["origin_of_coefficient"]
    assert "calibration" in prov
    assert prov["calibration"]["status"] == "raw_model_default"
    assert prov["calibration"]["calibrate_meal_response"] is False
    assert prov["calibration"]["held_out_rule_satisfied"] is False
    assert len(prov["food_table_lookups"]) == 1
    assert prov["food_table_lookups"][0]["provenance"] == "food_table"
    assert prov["food_table_lookups"][0]["carbs_g"] == 38.2

    # Verify optional calibration flag when explicitly requested
    res_cal = run_twin_simulation(
        patient_id="IND_PAT_001",
        initial_glucose=130.0,
        interventions=[InterventionItem(carbs_g=40.0)],
        calibrate_meal_response=True,
    )
    assert res_cal.provenance["calibration"]["status"] == "applied"
    assert res_cal.provenance["calibration"]["calibrate_meal_response"] is True
