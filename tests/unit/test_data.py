import pandas as pd

from src.glucotwin.data.loaders import load_cohort_dataset
from src.glucotwin.data.validation import validate_static_ehr


def test_static_validation():
    df = pd.DataFrame(
        [
            {
                "patient_id": "IND_PAT_001",
                "age": 52,
                "bmi": 27.4,
                "hba1c": 8.2,
                "diabetes_duration_years": 6.5,
                "egfr": 88.0,
                "systolic_bp": 138,
                "diastolic_bp": 86,
            }
        ]
    )
    validated = validate_static_ehr(df)
    assert len(validated) == 1


def test_load_cohort():
    cohort = load_cohort_dataset()
    assert len(cohort) > 0
    assert "patient_id" in cohort.columns
