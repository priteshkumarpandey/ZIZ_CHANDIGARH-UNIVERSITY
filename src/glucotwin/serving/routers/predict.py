from fastapi import APIRouter, HTTPException, Query

from src.glucotwin.serving.schemas.requests import PredictRequest
from src.glucotwin.serving.schemas.responses import ForecastResponse
from src.glucotwin.serving.service import inference_service

router = APIRouter(tags=["Glucose Forecasting"])


@router.get("/operating-thresholds")
def get_operating_thresholds():
    inference_service.load()
    return inference_service.operating_summary


@router.post("/predict", response_model=ForecastResponse)
def predict_glucose_forecast(
    request: PredictRequest,
    fallback: bool = Query(default=False, description="Use LightGBM fallback model if true"),
):
    try:
        return inference_service.predict(request, use_fallback=fallback)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
