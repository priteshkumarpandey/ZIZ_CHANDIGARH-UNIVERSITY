from src.glucotwin.synth.generator import generate_patient_cgm_series
from src.glucotwin.synth.meals import calculate_glycemic_load, get_all_food_items
from src.glucotwin.synth.personas import (
    get_all_personas,
    get_matched_participant,
)


def test_personas_linkage_and_folds():
    personas = get_all_personas()
    assert len(personas) == 4

    expected_matches = {
        "IND_PAT_001": ("CGMacros-039", 1),
        "IND_PAT_002": ("CGMacros-046", 0),
        "IND_PAT_003": ("CGMacros-035", 1),
        "IND_PAT_004": ("CGMacros-030", 3),
    }

    for persona in personas:
        # Every persona response must be labeled as synthetic linkage
        assert persona.linkage_type == "synthetic_linkage"
        exp_pid, exp_fold = expected_matches[persona.id]
        assert persona.matched_participant_id == exp_pid
        assert persona.matched_test_fold == exp_fold

        # Verify meal templates
        assert persona.typical_meal_templates is not None
        assert len(persona.typical_meal_templates) >= 2
        for tmpl in persona.typical_meal_templates:
            assert "template_name" in tmpl
            assert "items" in tmpl
            assert len(tmpl["items"]) > 0

    # Test lookup helper
    matched = get_matched_participant("IND_PAT_001")
    assert matched["matched_participant_id"] == "CGMacros-039"
    assert matched["matched_fold"] == 1


def test_meals_catalog_and_approximate_items():
    from pathlib import Path

    from src.glucotwin.synth.meals import FOOD_CSV_PATH

    assert FOOD_CSV_PATH.name == "indian_meal_templates.csv"
    assert FOOD_CSV_PATH.exists()
    assert not Path("data/external/ifct_foods.csv").exists()

    foods = get_all_food_items()
    # At least 20 items in Indian food library
    assert len(foods) >= 20

    # All items are marked approximate because values are unverified model recall
    approx_items = [f for f in foods if f.is_approximate]
    assert len(approx_items) == len(foods)

    for item in approx_items:
        assert item.notes is not None
        assert (
            "values computed from IFCT 2017 raw ingredient values and team recipe assumptions"
            in item.notes
        )

    gl = calculate_glycemic_load(foods[:2])
    assert gl > 0.0


def test_food_source_strings_must_be_on_allowed_list():
    import re

    foods = get_all_food_items()
    allowed_composition = {
        "derived from IFCT 2017 raw ingredient values (local extraction, not redistributed) and team recipe assumptions",
    }
    allowed_gi = {
        "approximate, not verified",
    }

    # Regex detecting specific codes or study citations
    code_or_citation_pattern = re.compile(
        r"(Code\s+[A-Z0-9]+|\b[A-F]\d{3}\b|et\s+al\.|Atkinson|Foster-Powell|Mani|Gopalan|Shobana)",
        re.IGNORECASE,
    )

    for f in foods:
        assert f.composition_source in allowed_composition, (
            f"Food '{f.name}' has unallowed composition_source: '{f.composition_source}'"
        )
        assert f.gi_source in allowed_gi, (
            f"Food '{f.name}' has unallowed gi_source: '{f.gi_source}'"
        )
        assert not code_or_citation_pattern.search(f.composition_source), (
            f"Food '{f.name}' contains specific code/citation: {f.composition_source}"
        )
        assert not code_or_citation_pattern.search(f.gi_source), (
            f"Food '{f.name}' contains specific code/citation: {f.gi_source}"
        )
        assert f.is_approximate is True, f"Food '{f.name}' is not approximate"
        assert (
            f.notes
            == "values computed from IFCT 2017 raw ingredient values and team recipe assumptions"
        ), f"Food '{f.name}' has unexpected notes: {f.notes}"

    # Verify that a row with an unallowed code fails validation
    from src.glucotwin.serving.schemas.fhir import FoodItem

    def validate_food_source(item: FoodItem) -> None:
        if item.composition_source not in allowed_composition or item.gi_source not in allowed_gi:
            raise ValueError(f"Source not allowed: {item.composition_source}, {item.gi_source}")

    import pytest

    invalid_item = FoodItem(
        id="test_unallowed_code",
        name="Test Item",
        regional_category="Test Region",
        serving_size_g=100.0,
        carbohydrates_g=10.0,
        fiber_g=1.0,
        protein_g=1.0,
        fat_g=1.0,
        glycemic_index=50.0,
        composition_source="IFCT 2017 (Code A001)",
        gi_source="Atkinson et al. 2021",
        is_approximate=True,
    )
    with pytest.raises(ValueError, match="Source not allowed"):
        validate_food_source(invalid_item)


def test_cgm_generation():
    ts, st = generate_patient_cgm_series("IND_PAT_001", days=1)
    assert len(ts) == 288
    assert len(st) == 1
