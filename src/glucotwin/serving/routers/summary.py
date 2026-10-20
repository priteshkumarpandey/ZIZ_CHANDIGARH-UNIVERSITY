from fastapi import APIRouter, HTTPException, Query

from src.glucotwin.serving.schemas.responses import ClinicalSummaryResponse
from src.glucotwin.serving.service import inference_service
from src.glucotwin.summary.generator import ClinicalSummaryGenerator
from src.glucotwin.synth.personas import get_persona_by_id

router = APIRouter(tags=["Clinical Summary"])


@router.get("/summary/{patient_id}", response_model=ClinicalSummaryResponse)
def get_clinical_summary(
    patient_id: str,
    window: int = Query(
        default=24, ge=1, le=168, description="Monitoring window in hours (default: 24)"
    ),
    as_of: str | None = Query(
        default=None, description="ISO timestamp cutoff for historical replay"
    ),
):
    inference_service.load()
    persona = get_persona_by_id(patient_id)
    if not persona and patient_id not in inference_service.patient_profiles:
        raise HTTPException(status_code=404, detail=f"Patient {patient_id} not found")

    try:
        return ClinicalSummaryGenerator.generate(
            patient_id=patient_id, window_hours=window, as_of=as_of
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
