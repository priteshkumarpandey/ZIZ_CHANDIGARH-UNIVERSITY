from fastapi import APIRouter

from src.glucotwin.config.settings import settings

router = APIRouter(tags=["System"])


@router.get("/health")
def health_check():
    return {
        "status": "healthy",
        "project": settings.PROJECT_NAME,
        "environment": settings.ENVIRONMENT,
        "version": "0.1.0",
    }
