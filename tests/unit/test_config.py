from src.glucotwin.config.settings import settings


def test_settings_defaults():
    assert settings.PROJECT_NAME == "GlucoTwin"
    assert settings.API_V1_PREFIX == "/api/v1"
    assert settings.RANDOM_SEED == 42
