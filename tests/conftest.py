import os

import pytest
from fastapi.testclient import TestClient

from src.glucotwin.serving.app import app


@pytest.fixture(autouse=True)
def setup_test_env():
    os.environ["USE_SQLITE_FALLBACK"] = "true"
    yield


@pytest.fixture
def client():
    return TestClient(app)
