import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException

router = APIRouter(tags=["Evidence & Validation"])


def _find_file(rel_path: str) -> Path | None:
    candidates = [
        Path(rel_path),
        Path("/app") / rel_path,
        Path(__file__).resolve().parent.parent.parent.parent / rel_path,
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


@router.get("/evidence")
def get_evidence_report() -> dict[str, Any]:
    # 1. Load metrics_cv.json
    metrics_cv_path = _find_file("reports/metrics_cv.json")
    if not metrics_cv_path:
        raise HTTPException(status_code=500, detail="reports/metrics_cv.json not found")

    with open(metrics_cv_path, encoding="utf-8") as f:
        metrics_cv = json.load(f)

    # 2. Load operating_thresholds.json
    thresholds_path = _find_file("reports/operating_thresholds.json")
    if not thresholds_path:
        raise HTTPException(status_code=500, detail="reports/operating_thresholds.json not found")

    with open(thresholds_path, encoding="utf-8") as f:
        operating_thresholds = json.load(f)

    # 3. Load whatif_validation.json
    whatif_path = _find_file("reports/whatif_validation.json")
    if not whatif_path:
        raise HTTPException(status_code=500, detail="reports/whatif_validation.json not found")

    with open(whatif_path, encoding="utf-8") as f:
        whatif_val = json.load(f)

    # 3b. Load paired_stats.json
    paired_stats: dict[str, Any] = {}
    paired_stats_path = _find_file("reports/paired_stats.json")
    if paired_stats_path:
        with open(paired_stats_path, encoding="utf-8") as f:
            paired_stats = json.load(f)

    # 4. Limitations
    limitations = [
        "Small research cohort: only 14 type 2 diabetes participants in CGMacros evaluation.",
        "EHR-to-wearable linkage is synthetic for demonstration and calibration.",
        "Hypoglycemia alerts: predicted hypoglycemia alerts are not clinically reliable due to very low event incidence in the training cohort; acute hypoglycemia (<70 mg/dL, <54 mg/dL) and rapid fall are covered by deterministic observed-value safety floor rules rather than predictive model inference.",
        "Meal glucose rises are directionally consistent but underestimated by approximately 50 percent in raw predictions.",
        "Meal values are derived from IFCT 2017 raw ingredient values and team recipe assumptions, and glycemic index values are unverified estimates.",
        "Post-meal physical activity effect is an assumption-based unvalidated heuristic (0.38 mg/dL per net MET-minute).",
        "No prospective clinical trial or real-time patient-facing trial has been conducted.",
    ]

    # 5. Data and licenses
    licenses = [
        {
            "dataset": "CGMacros",
            "license": "CC BY-NC-SA 4.0",
            "redistributed": False,
            "citation": "Gutierrez-Osuna, R., Kerr, D., Mortazavi, B., & Das, A. (2025). CGMacros: a scientific dataset for personalized nutrition and diet monitoring (version 1.0.0). PhysioNet. https://doi.org/10.13026/3z8q-x658.",
        },
        {
            "dataset": "HUPA-UCM",
            "license": "CC BY 4.0",
            "redistributed": False,
            "citation": "Hidalgo JI, Alvarado J, Botella M, et al. HUPA-UCM diabetes dataset. Data in Brief 55 (2024) 110559, doi 10.1016/j.dib.2024.110559.",
        },
        {
            "dataset": "Synthetic AI-READI",
            "license": "ADVANCE Center synthetic dataset, terms not published",
            "redistributed": False,
            "citation": "ADVANCE Center, Vanderbilt University Medical Center and Washington University. Derived from NIH Bridge2AI AI-READI cohort.",
        },
        {
            "dataset": "IFCT 2017",
            "license": "National Institute of Nutrition (ICMR)",
            "redistributed": False,
            "citation": "Longvah, T., Ananthan, R., Bhaskarachary, K., & Venkaiah, K. (2017). Indian Food Composition Tables. National Institute of Nutrition, ICMR, Hyderabad.",
        },
        {
            "dataset": "Model Artifacts & Reports",
            "license": "CC BY-NC-SA 4.0",
            "redistributed": True,
            "citation": "GlucoTwin project models and evaluation reports derived from CGMacros under CC BY-NC-SA 4.0.",
        },
        {
            "dataset": "Codebase",
            "license": "MIT",
            "redistributed": True,
            "citation": "GlucoTwin open-source software implementation.",
        },
    ]

    # 6. Read DATA_LICENSES.md text
    data_licenses_text = ""
    lic_path = _find_file("DATA_LICENSES.md")
    if lic_path and lic_path.exists():
        with open(lic_path, encoding="utf-8") as f:
            data_licenses_text = f.read()

    return {
        "metrics_cv": metrics_cv,
        "cv_metrics": metrics_cv,
        "operating_thresholds": operating_thresholds,
        "whatif_validation": whatif_val,
        "limitations": limitations,
        "licenses": licenses,
        "data_licenses": data_licenses_text,
        "paired_stats": paired_stats,
    }
