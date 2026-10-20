from fastapi import APIRouter, HTTPException, Query

from src.glucotwin.serving.schemas.responses import ExplainabilityResponse
from src.glucotwin.serving.service import inference_service

router = APIRouter(tags=["Explainability"])


@router.get("/explain/{patient_id}", response_model=ExplainabilityResponse)
def get_model_explanation(
    patient_id: str,
    horizon: int = Query(default=60, ge=30, le=120),
    as_of: str | None = Query(
        default=None, description="ISO timestamp cutoff for historical replay"
    ),
):
    try:
        return inference_service.explain(patient_id=patient_id, horizon=horizon, as_of=as_of)
    except Exception as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
