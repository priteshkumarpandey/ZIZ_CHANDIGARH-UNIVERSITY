import csv
from pathlib import Path
from typing import Any

from src.glucotwin.serving.schemas.fhir import FoodItem

FOOD_CSV_PATH = Path("data/external/indian_meal_templates.csv")


def _find_csv_path() -> Path | None:
    candidates = [
        FOOD_CSV_PATH,
        Path("/app") / FOOD_CSV_PATH,
        Path(__file__).resolve().parent.parent.parent.parent / FOOD_CSV_PATH,
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def _load_foods_from_csv() -> list[FoodItem]:
    foods: list[FoodItem] = []
    csv_file = _find_csv_path()
    if not csv_file or not csv_file.exists():
        return foods

    with open(csv_file, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader, start=1):
            is_approx = str(row.get("is_approximate", "")).strip().lower() == "true"
            item = FoodItem(
                id=f"ifct_{idx:03d}",
                name=row["food"].strip(),
                regional_category="Indian Cuisine",
                serving_size_g=float(row.get("serving_g", 100.0)),
                carbohydrates_g=float(row.get("carbs_g", 0.0)),
                fiber_g=float(row.get("fiber_g", 0.0)),
                protein_g=float(row.get("protein_g", 0.0)),
                fat_g=float(row.get("fat_g", 0.0)),
                glycemic_index=float(row.get("gi_value", 55.0)),
                source=row.get(
                    "composition_source",
                    "derived from IFCT 2017 raw ingredient values (local extraction, not redistributed) and team recipe assumptions",
                ).strip(),
                composition_source=row.get(
                    "composition_source",
                    "derived from IFCT 2017 raw ingredient values (local extraction, not redistributed) and team recipe assumptions",
                ).strip(),
                gi_source=row.get("gi_source", "approximate, not verified").strip(),
                is_approximate=is_approx,
                notes=row.get("notes", "").strip() or None,
            )
            foods.append(item)
    return foods


_CACHED_FOODS: list[FoodItem] = _load_foods_from_csv()


def get_all_food_items() -> list[FoodItem]:
    global _CACHED_FOODS
    if not _CACHED_FOODS:
        _CACHED_FOODS = _load_foods_from_csv()
    return _CACHED_FOODS


INDIAN_MEAL_CATALOG: list[dict[str, Any]] = [item.model_dump() for item in _CACHED_FOODS]
IFCT_FOOD_CATALOG: list[dict[str, Any]] = INDIAN_MEAL_CATALOG


def get_food_by_id(food_id: str) -> FoodItem | None:
    for item in get_all_food_items():
        if item.id == food_id:
            return item
    return None


def get_food_by_name(food_name: str) -> FoodItem | None:
    name_clean = food_name.strip().lower()
    for item in get_all_food_items():
        if item.name.lower() == name_clean or name_clean in item.name.lower():
            return item
    return None


def calculate_glycemic_load(items: list[FoodItem]) -> float:
    # High fat and dietary fiber blunt glucose absorption velocity
    total_gl = 0.0
    for item in items:
        available_carbs = max(0.0, item.carbohydrates_g - (item.fiber_g * 0.5))
        gl = (available_carbs * item.glycemic_index) / 100.0
        total_gl += gl
    return round(total_gl, 2)
