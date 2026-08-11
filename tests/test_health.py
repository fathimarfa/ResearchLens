import os

# Prevent Groq initialization from failing during test collection
os.environ.setdefault("GROQ_API_KEY", "test-key")

from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200