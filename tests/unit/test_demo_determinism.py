from pathlib import Path

import pandas as pd

from scripts.make_demo_data import generate_demo_data
from src.glucotwin.synth.personas import get_demo_personas, get_persona_by_id


def test_demo_patient_profile_canonical_alignment(tmp_path: Path):
    res = generate_demo_data(output_dir=tmp_path)
    patients_file = Path(res["patients_file"])
    assert patients_file.exists()

    df_pats = pd.read_parquet(patients_file)
    assert len(df_pats) == 3

    # Canonical persona for DEMO_PAT_001
    p1 = get_persona_by_id("DEMO_PAT_001")
    assert p1 is not None
    assert p1.age == 54
    assert p1.hba1c == 7.8
    assert p1.bmi == 28.4
    assert p1.fasting_glucose == 142.0
    assert p1.occupation == "Synthetic Engineer"
    assert p1.region is not None and p1.region.value == "West India"
    assert p1.is_synthetic is True

    # Assert attributes in generated demo parquet match canonical persona exactly
    row1 = df_pats[df_pats["patient_id"] == "DEMO_PAT_001"].iloc[0]
    assert int(row1["age"]) == 54
    assert str(row1["sex"]) == "M"
    assert float(row1["hba1c"]) == 7.8
    assert float(row1["bmi"]) == 28.4
    assert float(row1["fasting_glucose"]) == 142.0
    assert str(row1["region"]) == "West India"
    assert str(row1["occupation"]) == "Synthetic Engineer"
    assert bool(row1["is_synthetic"]) is True

    # Immutability check: all demo personas in parquet match get_demo_personas()
    for persona in get_demo_personas():
        row = df_pats[df_pats["patient_id"] == persona.id].iloc[0]
        assert int(row["age"]) == persona.age
        assert float(row["hba1c"]) == persona.hba1c
        assert float(row["fasting_glucose"]) == persona.fasting_glucose
        assert str(row["region"]) == (persona.region.value if persona.region else "")
