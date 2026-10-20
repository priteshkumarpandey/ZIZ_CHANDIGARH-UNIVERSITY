from typing import Any

from src.glucotwin.serving.schemas.fhir import (
    DietaryHabit,
    Gender,
    MedicationClass,
    PatientProfile,
    Region,
)

PERSONA_MATCHES: dict[str, dict[str, Any]] = {
    "IND_PAT_001": {
        "matched_participant_id": "CGMacros-039",
        "matched_fold": 1,
        "match_rationale": "Matched on T2D status, baseline HbA1c 8.2% vs 8.3%, and elevated postprandial glucose.",
    },
    "IND_PAT_002": {
        "matched_participant_id": "CGMacros-046",
        "matched_fold": 0,
        "match_rationale": "Matched on T2D status, baseline HbA1c 7.4% vs 7.4%, and morning excursion phenotype.",
    },
    "IND_PAT_003": {
        "matched_participant_id": "CGMacros-035",
        "matched_fold": 1,
        "match_rationale": "Matched on T2D status, baseline HbA1c 8.8% vs 8.5%, and rapid postprandial glycemic excursions.",
    },
    "IND_PAT_004": {
        "matched_participant_id": "CGMacros-030",
        "matched_fold": 3,
        "match_rationale": "Matched on T2D status, baseline HbA1c 7.9% vs 7.6%, and nocturnal glucose variability.",
    },
    "DEMO_PAT_001": {
        "matched_participant_id": "CGMacros-039",
        "matched_fold": 1,
        "match_rationale": "Synthetic demo patient 1 (T2D) mapped to representative fold 1.",
    },
    "DEMO_PAT_002": {
        "matched_participant_id": "CGMacros-046",
        "matched_fold": 0,
        "match_rationale": "Synthetic demo patient 2 (Prediabetes) mapped to representative fold 0.",
    },
    "DEMO_PAT_003": {
        "matched_participant_id": "CGMacros-035",
        "matched_fold": 1,
        "match_rationale": "Synthetic demo patient 3 (Normoglycemic) mapped to representative fold 1.",
    },
}

SYNTHETIC_PERSONAS: list[PatientProfile] = [
    PatientProfile(
        id="IND_PAT_001",
        name="Rajesh Sharma",
        age=52,
        gender=Gender.MALE,
        region=Region.NORTH_INDIA,
        occupation="IT Director",
        dietary_habit=DietaryHabit.VEGETARIAN,
        bmi=27.4,
        hba1c=8.2,
        diabetes_duration_years=6.5,
        egfr=88.0,
        systolic_bp=138,
        diastolic_bp=86,
        fasting_glucose=142.0,
        postprandial_glucose=215.0,
        active_medications=["Metformin 1000mg BD", "Glimepiride 2mg OD"],
        medication_classes=[MedicationClass.METFORMIN, MedicationClass.SULFONYLUREA],
        phenotype_notes="High sedentary screen time, work stress, high-carb dinner leading to frequent late-night hyperglycemia.",
        matched_participant_id="CGMacros-039",
        matched_test_fold=1,
        linkage_type="synthetic_linkage",
        is_synthetic=True,
        typical_meal_templates=[
            {
                "template_name": "North Indian Breakfast",
                "timing": "08:30",
                "items": [
                    {
                        "food": "Whole Wheat Roti (Chapati)",
                        "servings": 2.0,
                        "carbs_g": 64.0,
                        "kcal": 349.0,
                    },
                    {
                        "food": "Yellow Dal Tadka (Toor Dal)",
                        "servings": 1.0,
                        "carbs_g": 18.0,
                        "kcal": 143.8,
                    },
                    {
                        "food": "Masala Chai with 2 tsp Sugar",
                        "servings": 1.0,
                        "carbs_g": 12.0,
                        "kcal": 91.5,
                    },
                ],
                "total_carbs_g": 94.0,
                "notes": "Standard North Indian vegetarian morning meal.",
            },
            {
                "template_name": "North Indian Dinner",
                "timing": "20:30",
                "items": [
                    {
                        "food": "Whole Wheat Roti (Chapati)",
                        "servings": 2.0,
                        "carbs_g": 64.0,
                        "kcal": 349.0,
                    },
                    {
                        "food": "Paneer Tikka / Bhurji",
                        "servings": 1.0,
                        "carbs_g": 5.0,
                        "kcal": 272.0,
                    },
                    {
                        "food": "Yellow Dal Tadka (Toor Dal)",
                        "servings": 1.0,
                        "carbs_g": 18.0,
                        "kcal": 143.8,
                    },
                ],
                "total_carbs_g": 87.0,
                "notes": "High carbohydrate dinner driving nocturnal hyperglycemia.",
            },
            {
                "template_name": "High-Protein Snack",
                "timing": "17:00",
                "items": [
                    {
                        "food": "Paneer Tikka / Bhurji",
                        "servings": 1.0,
                        "carbs_g": 5.0,
                        "kcal": 272.0,
                    },
                ],
                "total_carbs_g": 5.0,
                "notes": "Low glycemic impact protein bridge snack.",
            },
        ],
    ),
    PatientProfile(
        id="IND_PAT_002",
        name="Priya Venkatesh",
        age=46,
        gender=Gender.FEMALE,
        region=Region.SOUTH_INDIA,
        occupation="School Teacher",
        dietary_habit=DietaryHabit.VEGETARIAN,
        bmi=25.8,
        hba1c=7.4,
        diabetes_duration_years=3.0,
        egfr=94.0,
        systolic_bp=126,
        diastolic_bp=80,
        fasting_glucose=128.0,
        postprandial_glucose=185.0,
        active_medications=["Metformin 500mg BD", "Teneligliptin 20mg OD"],
        medication_classes=[MedicationClass.METFORMIN, MedicationClass.DPP4_INHIBITOR],
        phenotype_notes="Traditional South Indian diet with white rice, idli and dosa. Pronounced morning dawn phenomenon with post-breakfast spikes.",
        matched_participant_id="CGMacros-046",
        matched_test_fold=0,
        linkage_type="synthetic_linkage",
        is_synthetic=True,
        typical_meal_templates=[
            {
                "template_name": "South Indian Breakfast",
                "timing": "08:00",
                "items": [
                    {
                        "food": "Steamed Idli (2 pieces)",
                        "servings": 1.0,
                        "carbs_g": 28.0,
                        "kcal": 139.2,
                    },
                    {"food": "Vegetable Sambar", "servings": 1.0, "carbs_g": 14.0, "kcal": 101.0},
                ],
                "total_carbs_g": 42.0,
                "notes": "Fermented rice and lentil preparation with moderate glycemic load.",
            },
            {
                "template_name": "South Indian Lunch",
                "timing": "13:00",
                "items": [
                    {
                        "food": "Steamed White Rice (Sona Masoori)",
                        "servings": 1.0,
                        "carbs_g": 42.0,
                        "kcal": 188.6,
                    },
                    {"food": "Vegetable Sambar", "servings": 1.0, "carbs_g": 14.0, "kcal": 101.0},
                    {
                        "food": "Curd / Dahi (Plain Whole Milk)",
                        "servings": 1.0,
                        "carbs_g": 7.0,
                        "kcal": 104.0,
                    },
                ],
                "total_carbs_g": 63.0,
                "notes": "Polished rice lunch with high glycemic index.",
            },
            {
                "template_name": "Millet Alternative",
                "timing": "13:00",
                "items": [
                    {
                        "food": "Ragi Roti (Finger Millet)",
                        "servings": 1.0,
                        "carbs_g": 30.0,
                        "kcal": 155.4,
                    },
                    {"food": "Vegetable Sambar", "servings": 1.0, "carbs_g": 14.0, "kcal": 101.0},
                    {
                        "food": "Curd / Dahi (Plain Whole Milk)",
                        "servings": 1.0,
                        "carbs_g": 7.0,
                        "kcal": 104.0,
                    },
                ],
                "total_carbs_g": 51.0,
                "notes": "Lower GI finger millet substitution for postprandial blunting.",
            },
        ],
    ),
    PatientProfile(
        id="IND_PAT_003",
        name="Ananya Mukherjee",
        age=38,
        gender=Gender.FEMALE,
        region=Region.EAST_INDIA,
        occupation="Financial Analyst",
        dietary_habit=DietaryHabit.NON_VEGETARIAN,
        bmi=23.6,
        hba1c=8.8,
        diabetes_duration_years=2.0,
        egfr=105.0,
        systolic_bp=130,
        diastolic_bp=84,
        fasting_glucose=155.0,
        postprandial_glucose=235.0,
        active_medications=["Metformin 1000mg BD", "Dapagliflozin 10mg OD"],
        medication_classes=[MedicationClass.METFORMIN, MedicationClass.SGLT2_INHIBITOR],
        phenotype_notes="Early onset T2D with family history. Normal BMI but high visceral fat and rapid glycemic excursions after traditional sweets.",
        matched_participant_id="CGMacros-035",
        matched_test_fold=1,
        linkage_type="synthetic_linkage",
        is_synthetic=True,
        typical_meal_templates=[
            {
                "template_name": "East Indian Lunch",
                "timing": "13:30",
                "items": [
                    {
                        "food": "Steamed White Rice (Sona Masoori)",
                        "servings": 1.0,
                        "carbs_g": 42.0,
                        "kcal": 188.6,
                    },
                    {
                        "food": "Bengali Fish Curry (Macher Jhol)",
                        "servings": 1.0,
                        "carbs_g": 6.0,
                        "kcal": 184.0,
                    },
                    {
                        "food": "Yellow Dal Tadka (Toor Dal)",
                        "servings": 1.0,
                        "carbs_g": 18.0,
                        "kcal": 143.8,
                    },
                ],
                "total_carbs_g": 66.0,
                "notes": "Freshwater fish in light cumin-turmeric broth with steamed rice.",
            },
            {
                "template_name": "Traditional Sweet Occasion",
                "timing": "16:30",
                "items": [
                    {
                        "food": "Gulab Jamun (1 piece)",
                        "servings": 1.0,
                        "carbs_g": 33.0,
                        "kcal": 223.0,
                    },
                ],
                "total_carbs_g": 33.0,
                "notes": "Deep fried khoya dumpling in sugar syrup, rapid glucose spike.",
            },
            {
                "template_name": "Comfort Khichdi Dinner",
                "timing": "20:00",
                "items": [
                    {"food": "Moong Dal Khichdi", "servings": 1.0, "carbs_g": 36.0, "kcal": 225.0},
                    {
                        "food": "Curd / Dahi (Plain Whole Milk)",
                        "servings": 1.0,
                        "carbs_g": 7.0,
                        "kcal": 104.0,
                    },
                ],
                "total_carbs_g": 43.0,
                "notes": "Balanced rice and split moong lentil porridge.",
            },
        ],
    ),
    PatientProfile(
        id="IND_PAT_004",
        name="Vikram Patil",
        age=64,
        gender=Gender.MALE,
        region=Region.WEST_INDIA,
        occupation="Retired Engineer",
        dietary_habit=DietaryHabit.VEGETARIAN,
        bmi=26.2,
        hba1c=7.9,
        diabetes_duration_years=14.0,
        egfr=64.0,
        systolic_bp=142,
        diastolic_bp=88,
        fasting_glucose=115.0,
        postprandial_glucose=190.0,
        active_medications=["Metformin 500mg BD", "Insulin Glargine 16 units at bedtime"],
        medication_classes=[MedicationClass.METFORMIN, MedicationClass.BASAL_INSULIN],
        phenotype_notes="Long-standing diabetes with reduced beta-cell reserve. Vulnerable to nocturnal hypoglycemia between 2 AM and 4 AM.",
        matched_participant_id="CGMacros-030",
        matched_test_fold=3,
        linkage_type="synthetic_linkage",
        is_synthetic=True,
        typical_meal_templates=[
            {
                "template_name": "West Indian Breakfast",
                "timing": "08:30",
                "items": [
                    {
                        "food": "Kanda Poha (Flattened Rice)",
                        "servings": 1.0,
                        "carbs_g": 45.0,
                        "kcal": 264.3,
                    },
                    {
                        "food": "Masala Chai with 2 tsp Sugar",
                        "servings": 1.0,
                        "carbs_g": 12.0,
                        "kcal": 91.5,
                    },
                ],
                "total_carbs_g": 57.0,
                "notes": "Flattened rice flakes tempered with mustard seeds and onions.",
            },
            {
                "template_name": "West Indian Lunch",
                "timing": "13:00",
                "items": [
                    {
                        "food": "Whole Wheat Roti (Chapati)",
                        "servings": 2.0,
                        "carbs_g": 64.0,
                        "kcal": 349.0,
                    },
                    {
                        "food": "Chana Masala (Chickpea Curry)",
                        "servings": 1.0,
                        "carbs_g": 28.0,
                        "kcal": 204.0,
                    },
                    {
                        "food": "Curd / Dahi (Plain Whole Milk)",
                        "servings": 1.0,
                        "carbs_g": 7.0,
                        "kcal": 104.0,
                    },
                ],
                "total_carbs_g": 99.0,
                "notes": "High fiber legume and whole wheat combination.",
            },
            {
                "template_name": "Bedtime Protein Snack",
                "timing": "22:00",
                "items": [
                    {
                        "food": "Besan Chilla (Gram Flour Pancake)",
                        "servings": 1.0,
                        "carbs_g": 22.0,
                        "kcal": 169.5,
                    },
                ],
                "total_carbs_g": 22.0,
                "notes": "Complex carbohydrate and protein snack to protect against nocturnal hypoglycemic dips.",
            },
        ],
    ),
]


DEMO_PERSONAS: list[PatientProfile] = [
    PatientProfile(
        id="DEMO_PAT_001",
        name="Demo Patient 1 (Synthetic T2D)",
        age=54,
        gender=Gender.MALE,
        region=Region.WEST_INDIA,
        occupation="Synthetic Engineer",
        dietary_habit=DietaryHabit.VEGETARIAN,
        bmi=28.4,
        hba1c=7.8,
        diabetes_duration_years=4.5,
        egfr=88.0,
        systolic_bp=132,
        diastolic_bp=84,
        fasting_glucose=142.0,
        postprandial_glucose=185.0,
        active_medications=["Metformin 500mg"],
        medication_classes=[MedicationClass.METFORMIN],
        phenotype_notes="Synthetic T2D patient with postprandial glycemic excursions.",
        matched_participant_id="CGMacros-039",
        matched_test_fold=1,
        linkage_type="synthetic_demo",
        is_synthetic=True,
    ),
    PatientProfile(
        id="DEMO_PAT_002",
        name="Demo Patient 2 (Synthetic Prediabetes)",
        age=46,
        gender=Gender.FEMALE,
        region=Region.NORTH_INDIA,
        occupation="Synthetic Analyst",
        dietary_habit=DietaryHabit.NON_VEGETARIAN,
        bmi=25.2,
        hba1c=5.9,
        diabetes_duration_years=1.0,
        egfr=94.0,
        systolic_bp=122,
        diastolic_bp=78,
        fasting_glucose=108.0,
        postprandial_glucose=138.0,
        active_medications=[],
        medication_classes=[],
        phenotype_notes="Synthetic prediabetes profile with morning excursion tendencies.",
        matched_participant_id="CGMacros-046",
        matched_test_fold=0,
        linkage_type="synthetic_demo",
        is_synthetic=True,
    ),
    PatientProfile(
        id="DEMO_PAT_003",
        name="Demo Patient 3 (Synthetic Normoglycemic)",
        age=33,
        gender=Gender.MALE,
        region=Region.SOUTH_INDIA,
        occupation="Synthetic Developer",
        dietary_habit=DietaryHabit.VEGETARIAN,
        bmi=22.3,
        hba1c=5.1,
        diabetes_duration_years=0.0,
        egfr=102.0,
        systolic_bp=115,
        diastolic_bp=74,
        fasting_glucose=89.0,
        postprandial_glucose=112.0,
        active_medications=[],
        medication_classes=[],
        phenotype_notes="Synthetic healthy reference profile.",
        matched_participant_id="CGMacros-035",
        matched_test_fold=1,
        linkage_type="synthetic_demo",
        is_synthetic=True,
    ),
]


def get_all_personas() -> list[PatientProfile]:
    return SYNTHETIC_PERSONAS


def get_demo_personas() -> list[PatientProfile]:
    return DEMO_PERSONAS


def get_persona_by_id(patient_id: str) -> PatientProfile | None:
    for persona in SYNTHETIC_PERSONAS + DEMO_PERSONAS:
        if persona.id == patient_id:
            return persona
    return None


def get_matched_participant(patient_id: str) -> dict[str, Any] | None:
    return PERSONA_MATCHES.get(patient_id)
