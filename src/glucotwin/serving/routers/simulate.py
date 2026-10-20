from fastapi import APIRouter, HTTPException

from src.glucotwin.serving.schemas.fhir import FoodItem
from src.glucotwin.serving.schemas.requests import SimulateRequest
from src.glucotwin.serving.schemas.responses import SimulationResponse
from src.glucotwin.synth.meals import get_all_food_items
from src.glucotwin.twin.simulator import run_twin_simulation

router = APIRouter(tags=["What-If Simulation"])


@router.get("/simulate/foods", response_model=list[FoodItem])
def list_food_catalog():
    return get_all_food_items()


@router.post("/simulate", response_model=SimulationResponse)
def execute_what_if_simulation(request: SimulateRequest):
    try:
        init_g = None
        if request.baseline_cgm and len(request.baseline_cgm) > 0:
            init_g = request.baseline_cgm[-1].glucose_mgdl
        elif request.recent_cgm_series and len(request.recent_cgm_series) > 0:
            init_g = request.recent_cgm_series[-1]

        return run_twin_simulation(
            patient_id=request.patient_id,
            as_of=request.as_of,
            recent_cgm_series=request.recent_cgm_series,
            interventions=request.interventions,
            initial_glucose=init_g,
            meal=request.planned_meal,
            activity=request.planned_activity,
            duration_minutes=request.simulation_duration_minutes,
            calibrate_meal_response=request.calibrate_meal_response,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
