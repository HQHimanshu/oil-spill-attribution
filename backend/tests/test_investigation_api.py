import sys
from pathlib import Path
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.main import app

client = TestClient(app)


def test_create_investigation_returns_structure():
    payload = {
        "latitude": 20.45,
        "longitude": 68.32,
        "timestamp": "2024-02-11T10:30:00Z",
        "metadata": {"status": "valid"},
    }

    response = client.post("/api/investigations", json=payload)

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["id"]
    assert body["status"] in {"draft", "ready", "analyzing"}
    assert abs(body["spill"]["centroid"]["latitude"] - payload["latitude"]) < 1.0
    assert abs(body["spill"]["centroid"]["longitude"] - payload["longitude"]) < 1.0


def test_get_investigation_returns_expected_sections():
    response = client.get("/api/investigations/demo-investigation")

    assert response.status_code == 200, response.text
    body = response.json()
    assert "spill" in body
    assert "origin" in body
    assert "environment" in body
    assert "vessels" in body
    assert "uncertainty" in body
    assert "dataAvailability" in body


def test_get_investigation_unknown_id_returns_404():
    response = client.get("/api/investigations/does-not-exist")

    assert response.status_code == 404, response.text
    assert "not found" in response.json()["detail"].lower()


def test_location_search():
    response = client.get("/api/location/search?query=Mumbai")
    assert response.status_code == 200
    data = response.json()
    assert len(data["results"]) > 0
    assert "latitude" in data["results"][0]


def test_live_environmental():
    response = client.get("/api/environmental/live?latitude=18.94&longitude=72.86")
    assert response.status_code == 200
    data = response.json()
    assert "wind" in data
    assert "current" in data
    assert data["wind"]["speed"] > 0

