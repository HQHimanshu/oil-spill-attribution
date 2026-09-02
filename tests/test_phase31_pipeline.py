"""
Comprehensive Unit and Integration Tests for Phase 31 Pipeline:
1. Real SAR dataset and ML inference
2. Environmental data caching and fetching
3. Real historical AIS service & gap detection
4. Physics-based backtracking drift model
5. Multi-factor vessel ranking
6. RAG knowledge vector store and grounded investigation copilot
7. Backend API contract validation
"""
import sys
from pathlib import Path
import pytest
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "backend") not in sys.path:
    sys.path.insert(0, str(ROOT / "backend"))

from fastapi.testclient import TestClient
from backend.app.main import app
from backend.app.environmental_service import get_environmental_data
from backend.app.ais_service import get_ais_data, calculate_haversine_distance
from backend.app.backtracking_service import compute_drift_vector, reconstruct_probable_origin
from backend.app.vessel_ranking import rank_candidate_vessels
from backend.app.rag.assistant import get_investigation_copilot
from backend.app.rag.ingestion import get_vector_store
from ml.characterization.inference import get_sar_inference_engine
from ml.characterization.preprocessing import extract_sar_features, apply_lee_filter

client = TestClient(app)


def test_api_health():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["data_mode"] == "REAL DATA"
    assert data["ml_model_loaded"] is True
    assert data["rag_kb_loaded"] is True


def test_sar_feature_extraction():
    dummy_img = np.random.randint(50, 180, size=(128, 128), dtype=np.uint8)
    features = extract_sar_features(dummy_img)
    assert features.shape == (128, 128, 10)
    assert not np.isnan(features).any()

    lee = apply_lee_filter(dummy_img, size=5)
    assert lee.shape == (128, 128)


def test_sar_ml_inference():
    engine = get_sar_inference_engine()
    assert engine.model is not None

    sample_sar = ROOT / "backend" / "database" / "sar_scenes" / "S1A_IW_GRDH_1SDV_20201231T113025_GOM_GALVESTON.png"
    if sample_sar.exists():
        result = engine.run_inference(str(sample_sar), center_lat=28.582, center_lon=-94.925)
        assert result["detected"] is True
        assert result["confidence"] >= 0.80
        assert result["area_km2"] > 0
        assert "lat" in result["centroid"]
        assert "lon" in result["centroid"]
        assert len(result["boundary"]) > 0


def test_environmental_data_service():
    env = get_environmental_data(28.582, -94.925, "2020-12-31T11:30:25Z")
    assert "wind" in env
    assert "current" in env
    assert env["wind"]["speed"] > 0
    assert 0 <= env["wind"]["direction"] <= 360
    assert env["current"]["speed"] >= 0
    assert "source" in env
    assert "REAL DATA" in env["data_mode"]


def test_backtracking_drift_physics():
    u, v, spd, dir_deg = compute_drift_vector(
        wind_speed_knots=15.0,
        wind_direction_deg=135.0,
        current_speed_ms=0.40,
        current_direction_deg=70.0,
        latitude=28.582
    )
    assert spd > 0
    assert 0 <= dir_deg <= 360

    backtrack = reconstruct_probable_origin(
        spill_latitude=28.582,
        spill_longitude=-94.925,
        observation_time_iso="2020-12-31T11:30:25Z",
        wind_speed_knots=15.0,
        wind_direction_deg=135.0,
        current_speed_ms=0.40,
        current_direction_deg=70.0,
        estimated_drift_hours=3.0
    )
    assert "probableOrigin" in backtrack
    assert "latitude" in backtrack["probableOrigin"]
    assert "longitude" in backtrack["probableOrigin"]
    assert len(backtrack["route"]) > 2
    assert backtrack["backtrackingConfidence"] >= 70


def test_real_ais_service_and_gap_detection():
    ais_res = get_ais_data(
        origin_region={"latitude": 28.582, "longitude": -94.925},
        max_distance_km=60.0
    )
    assert "vessels" in ais_res
    assert len(ais_res["vessels"]) > 0
    assert "data_mode" in ais_res
    assert "REAL DATA" in ais_res["data_mode"]

    v0 = ais_res["vessels"][0]
    assert v0["mmsi"]
    assert len(v0["positions"]) > 0
    assert "lat" in v0["positions"][0]
    assert "lon" in v0["positions"][0]


def test_multi_factor_vessel_ranking():
    ais_res = get_ais_data(origin_region={"latitude": 28.582, "longitude": -94.925})
    ranked = rank_candidate_vessels(
        probable_origin={"latitude": 28.25, "longitude": -95.35},
        origin_time_window={"start": "2020-12-31T08:00:00Z", "end": "2020-12-31T11:00:00Z"},
        candidate_vessels=ais_res["vessels"]
    )
    assert len(ranked) > 0
    assert ranked[0]["rank"] == 1
    assert ranked[0]["evidenceScore"] >= ranked[-1]["evidenceScore"]
    assert ranked[0]["riskLevel"] in ["HIGH", "MEDIUM", "LOW"]
    assert "evidence" in ranked[0]
    assert "proximity" in ranked[0]["evidence"]


def test_rag_knowledge_base_and_copilot():
    store = get_vector_store()
    assert len(store.chunks) >= 5

    res = store.search("What wind speed is required for SAR oil detection?", top_k=2)
    assert len(res) > 0
    assert res[0][0].title

    copilot = get_investigation_copilot()
    mock_context = {
        "spill": {"detected": True, "confidence": 94, "area": 14.85, "centroid": {"latitude": 28.582, "longitude": -94.925}},
        "environment": {"wind": {"speed": 14.2, "unit": "knots", "direction": 135}, "current": {"speed": 0.42, "unit": "m/s", "direction": 72}},
        "origin": {
            "probableOrigin": {"latitude": 28.22, "longitude": -95.40},
            "timeWindow": {"start": "2020-12-31T08:00:00Z", "end": "2020-12-31T10:30:00Z"},
            "backtrackingConfidence": 85,
            "uncertaintyRadiusKm": 3.8
        },
        "vessels": [
            {
                "name": "HOEGH SHANGHAI",
                "mmsi": "258758000",
                "latitude": 28.58,
                "longitude": -94.92,
                "distanceKm": 4.2,
                "evidenceScore": 94,
                "proximity": 92,
                "timeMatch": 95,
                "trajectoryMatch": 92,
                "vesselType": "Vehicle Carrier / Cargo",
                "aisGapDetected": False
            }
        ]
    }

    ans = copilot.query("Why was this vessel ranked first?", mock_context)
    assert "Observed Data" in ans["answer"]
    assert "Model Output" in ans["answer"]
    assert "Retrieved Knowledge" in ans["answer"]
    assert "Inference" in ans["answer"]
    assert len(ans["sources"]) > 0
    assert "definitely caused" not in ans["answer"]


def test_api_investigation_flow():
    # 1. Create Investigation
    res = client.post(
        "/api/investigations",
        json={
            "latitude": 28.582,
            "longitude": -94.925,
            "timestamp": "2020-12-31T11:30:25Z"
        }
    )
    assert res.status_code == 200
    inv = res.json()
    assert inv["id"]
    assert inv["spill"]["detected"] is True
    assert len(inv["vessels"]) > 0
    assert inv["selectedVessel"] is not None

    inv_id = inv["id"]

    # 2. Copilot Query Endpoint
    copilot_res = client.post(
        f"/api/investigations/{inv_id}/ask",
        json={"question": "What evidence supports this candidate?"}
    )
    assert copilot_res.status_code == 200
    copilot_data = copilot_res.json()
    assert "answer" in copilot_data
    assert len(copilot_data["sources"]) > 0
    assert "grounding" in copilot_data

    # 3. Model Metrics Endpoint
    metrics_res = client.get("/api/model/metrics")
    assert metrics_res.status_code == 200
    metrics = metrics_res.json()
    assert metrics["test_mean_iou"] >= 0.50
    assert metrics["test_mean_dice"] >= 0.60
    assert "distinction_note" in metrics

    # 4. Provenance Endpoint
    prov_res = client.get("/api/provenance")
    assert prov_res.status_code == 200
    prov = prov_res.json()
    assert prov["system_data_mode"] == "REAL DATA"
    assert "data_sources" in prov
