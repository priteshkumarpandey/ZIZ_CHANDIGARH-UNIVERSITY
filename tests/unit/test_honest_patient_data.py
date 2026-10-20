from src.glucotwin.serving.schemas.fhir import Gender, PatientProfile
from src.glucotwin.serving.service import inference_service
from src.glucotwin.synth.personas import get_all_personas, get_demo_personas


def test_honest_patient_data_schema_and_defaults():
    # Constructing a de-identified research participant without invented fields
    real_p = PatientProfile(
        id="CGMacros-004",
        name="CGMacros-004",
        age=45,
        gender=Gender.MALE,
        bmi=24.2,
        hba1c=6.9,
        fasting_glucose=110.0,
        is_synthetic=False,
    )
    assert real_p.region is None
    assert real_p.occupation is None
    assert real_p.dietary_habit is None
    assert real_p.postprandial_glucose is None
    assert real_p.name == "CGMacros-004"
    assert real_p.is_synthetic is False


def test_research_records_do_not_expose_invented_fields():
    inference_service.load()
    patients = inference_service.list_patients()

    research_patients = [p for p in patients if not p.is_synthetic or p.id.startswith("CGMacros")]
    for p in research_patients:
        # Fails if a de-identified research record exposes region, occupation, or diet
        assert p.region is None, f"Research participant {p.id} illegally exposes region: {p.region}"
        assert p.occupation is None, (
            f"Research participant {p.id} illegally exposes occupation: {p.occupation}"
        )
        assert p.dietary_habit is None, (
            f"Research participant {p.id} illegally exposes diet: {p.dietary_habit}"
        )
        assert p.postprandial_glucose is None, (
            f"Research participant {p.id} illegally exposes postprandial: {p.postprandial_glucose}"
        )
        assert p.name == p.id, f"Research participant {p.id} has invented name: {p.name}"


def test_synthetic_personas_and_demo_patients_have_synthetic_flag():
    personas = get_all_personas()
    assert len(personas) == 4
    for p in personas:
        assert p.is_synthetic is True
        assert p.region is not None
        assert p.occupation is not None
        assert p.dietary_habit is not None

    demos = get_demo_personas()
    assert len(demos) == 3
    for p in demos:
        assert p.is_synthetic is True
        assert p.region is not None
        assert p.occupation is not None
        assert p.dietary_habit is not None
