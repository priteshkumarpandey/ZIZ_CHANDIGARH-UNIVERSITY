import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import make_asgi_app

from src.glucotwin.config.settings import settings
from src.glucotwin.serving.metrics import REQUEST_COUNT, REQUEST_LATENCY
from src.glucotwin.serving.routers.evidence import router as evidence_router
from src.glucotwin.serving.routers.explain import router as explain_router
from src.glucotwin.serving.routers.health import router as health_router
from src.glucotwin.serving.routers.predict import router as predict_router
from src.glucotwin.serving.routers.simulate import router as simulate_router
from src.glucotwin.serving.routers.summary import router as summary_router
from src.glucotwin.serving.routers.twin import router as twin_router
from src.glucotwin.serving.routers.websocket import router as websocket_router
from src.glucotwin.utils.logging import logger


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"Starting {settings.PROJECT_NAME} API service [{settings.ENVIRONMENT}]")
    try:
        from src.glucotwin.serving.service import inference_service

        inference_service.load()
        logger.info("InferenceService pre-loaded successfully during app lifespan.")
    except Exception as e:
        logger.warning(f"InferenceService pre-load warning: {e}")

    try:
        from src.glucotwin.serving.routers.websocket import process_and_broadcast_cgm_reading
        from src.glucotwin.streaming.kafka import kafka_manager

        await kafka_manager.start_consumer(process_and_broadcast_cgm_reading)
        logger.info("Kafka consumer initialized successfully.")
    except Exception as exc:
        logger.warning(f"Kafka consumer initialization warning: {exc}")

    yield

    try:
        from src.glucotwin.streaming.kafka import kafka_manager

        await kafka_manager.disconnect()
    except Exception:
        pass
    logger.info(f"Stopping {settings.PROJECT_NAME} API service")


app = FastAPI(
    title=settings.PROJECT_NAME,
    version="0.1.0",
    description="India-aware Digital Twin for proactive Type 2 Diabetes management",
    lifespan=lifespan,
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request ID and Structured Logging Middleware
@app.middleware("http")
async def request_middleware(request: Request, call_next):
    req_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    start_time = time.time()

    response = await call_next(request)

    duration = time.time() - start_time
    response.headers["X-Request-ID"] = req_id

    endpoint = request.url.path
    REQUEST_COUNT.labels(
        method=request.method, endpoint=endpoint, status_code=response.status_code
    ).inc()
    REQUEST_LATENCY.labels(endpoint=endpoint).observe(duration)

    logger.info(
        f"{request.method} {endpoint} completed with status {response.status_code} in {duration:.4f}s",
        extra={"request_id": req_id},
    )
    return response


# Prometheus metrics mount
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# Root Health Route
app.include_router(health_router)

# Versioned API Router at /api/v1
v1_app = FastAPI(title=f"{settings.PROJECT_NAME} v1 API")
v1_app.include_router(health_router)
v1_app.include_router(twin_router)
v1_app.include_router(predict_router)
v1_app.include_router(simulate_router)
v1_app.include_router(explain_router)
v1_app.include_router(summary_router)
v1_app.include_router(evidence_router)
v1_app.include_router(websocket_router)

app.mount(settings.API_V1_PREFIX, v1_app)
