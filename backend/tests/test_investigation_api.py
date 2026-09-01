from fastapi.testclient import TestClient

from app.main import app

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
    assert body["spill"]["centroid"]["latitude"] == payload["latitude"]
    assert body["spill"]["centroid"]["longitude"] == payload["longitude"]


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
