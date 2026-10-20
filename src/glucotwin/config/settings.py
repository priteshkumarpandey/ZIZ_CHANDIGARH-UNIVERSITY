from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    PROJECT_NAME: str = "GlucoTwin"
    ENVIRONMENT: str = "development"
    LOG_LEVEL: str = "INFO"
    DEBUG: bool = True

    POSTGRES_USER: str = "glucouser"
    POSTGRES_PASSWORD: str = "glucopass"
    POSTGRES_DB: str = "glucotwin"
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432
    DATABASE_URL: str = "postgresql+asyncpg://glucouser:glucopass@localhost:5432/glucotwin"

    DEMO_MODE: bool = False

    KAFKA_BOOTSTRAP_SERVERS: str = "localhost:9092"
    KAFKA_CGM_TOPIC: str = "cgm"
    KAFKA_ACTIVITY_TOPIC: str = "activity"
    KAFKA_MEALS_TOPIC: str = "meals"
    KAFKA_ALERT_TOPIC: str = "glucotwin.alerts"
    KAFKA_CONSUMER_GROUP: str = "glucotwin-serving-group"

    REPLAY_SPEED: float = 30.0
    REPLAY_PATIENT_ID: str = "DEMO_PAT_001"

    MLFLOW_TRACKING_URI: str = "http://localhost:5000"
    MLFLOW_EXPERIMENT_NAME: str = "glucotwin-forecasting"

    API_V1_PREFIX: str = "/api/v1"
    CORS_ORIGINS: list[str] = ["http://localhost:3000", "http://127.0.0.1:3000"]
    RANDOM_SEED: int = 42

    MODEL_CHECKPOINT_DIR: str = "models/checkpoints"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


settings = Settings()
