import sys
from pathlib import Path
import json
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.main import app

client = TestClient(app)


def test_endpoints():
    # 1. Health
    h = client.get("/health")
    assert h.status_code == 200
    
    # 2. Create Investigation
    inv = client.post("/api/investigations", json={
        "latitude": 28.582,
        "longitude": -94.925,
        "timestamp": "2020-12-31T11:30:25Z"
    })
    assert inv.status_code == 200
    data = inv.json()
    inv_id = data["id"]
    assert data["spill"]["detected"] is True
    assert len(data["vessels"]) > 0
    
    # 3. Ask RAG Copilot
    rag_q1 = client.post(f"/api/investigations/{inv_id}/ask", json={
        "question": "Why was this vessel ranked first?"
    })
    assert rag_q1.status_code == 200
    rag_res1 = rag_q1.json()
    assert len(rag_res1["sources"]) > 0
    assert "Observed Data" in rag_res1["answer"]
    
    # 4. Model Evaluation Metrics
    m = client.get("/api/model/metrics")
    assert m.status_code == 200
    metrics = m.json()
    assert metrics["test_mean_iou"] > 0
    
    # 5. Provenance
    p = client.get("/api/provenance")
    assert p.status_code == 200
    prov = p.json()
    assert prov["system_data_mode"] == "REAL DATA"


if __name__ == "__main__":
    test_endpoints()
    print("ALL API ENDPOINTS OPERATIONAL AND FULLY VERIFIED!")

