from src.glucotwin.serving.routers.explain import router as explain_router
from src.glucotwin.serving.routers.health import router as health_router
from src.glucotwin.serving.routers.predict import router as predict_router
from src.glucotwin.serving.routers.simulate import router as simulate_router
from src.glucotwin.serving.routers.summary import router as summary_router
from src.glucotwin.serving.routers.twin import router as twin_router
from src.glucotwin.serving.routers.websocket import router as websocket_router

__all__ = [
    "health_router",
    "twin_router",
    "predict_router",
    "simulate_router",
    "explain_router",
    "summary_router",
    "websocket_router",
]
