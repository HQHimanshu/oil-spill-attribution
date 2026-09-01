import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "backend") not in sys.path:
    sys.path.insert(0, str(ROOT / "backend"))

from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)


def test_create_investigation_contract():
    response = client.post(
        "/api/investigations",
        json={
            "latitude": 28.582,
            "longitude": -94.925,
            "timestamp": "2020-12-31T11:30:25Z",
            "metadata": {"status": "valid"},
        },
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["id"]
    assert payload["status"] in {"ready", "pending"}
    assert "spill" in payload
    assert "vessels" in payload
    assert payload["selectedVessel"] is not None


def test_analyze_investigation_contract():
    response = client.post("/api/investigations/demo-investigation/analyze")

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["id"] == "demo-investigation"
    assert payload["spill"]["centroid"]["latitude"]
    assert payload["vessels"]
    assert payload["selectedVessel"]


def test_rag_copilot_endpoint_contract():
    response = client.post(
        "/api/investigations/demo-investigation/ask",
        json={"question": "Why was this vessel ranked first?"}
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert "answer" in payload
    assert "sources" in payload
    assert "Observed Data" in payload["answer"]
