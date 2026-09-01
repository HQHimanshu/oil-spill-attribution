from typing import Any, Dict

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.detection import detect_oil_spill
from app.vessel_matching import rank_vessels
from app.ais_data import get_vessels_from_ais_csv, get_trajectories_from_ais_csv
from app.investigation_service import (
    analyze_investigation,
    build_demo_investigation,
    create_investigation,
    get_investigation,
    get_vessel_for_investigation,
    get_vessels_for_investigation,
)

app = FastAPI(
    title="OceanGuard AI",
    description="SIH26143 Oil Spill Detection and Vessel Correlation Prototype",
    version="2.0"
)

# =========================
# CORS
# =========================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5500",
    "http://127.0.0.1:5500",],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# HOME
# =========================

@app.get("/")
def home():
    return {
        "project": "OceanGuard AI",
        "problem_statement": "SIH26143",
        "status": "running",
        "version": "2.0"
    }


# =========================
# HEALTH CHECK
# =========================

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "system": "Operational"
    }


# =========================
# OIL SPILL DETECTION
# =========================

@app.post("/detect-spill")
async def detect_spill(file: UploadFile = File(...)):

    try:
        image_bytes = await file.read()

        if not image_bytes:
            raise ValueError("Empty image file")

        result = detect_oil_spill(image_bytes)

        return result

    except Exception as e:

        raise HTTPException(
            status_code=400,
            detail=str(e)
        )


# =========================
# GET VESSELS
# =========================

@app.get("/vessels")
def get_vessels():

    VESSELS = get_vessels_from_ais_csv()

    return {
        "count": len(VESSELS),
        "vessels": VESSELS
    }


# =========================
# MATCH VESSELS
# =========================

@app.post("/match-vessels")
def match_vessels(
    latitude: float,
    longitude: float
):

    VESSELS = get_vessels_from_ais_csv()

    ranked = rank_vessels(
        latitude,
        longitude,
        VESSELS
    )

    return {
        "spill_location": {
            "latitude": latitude,
            "longitude": longitude
        },
        "suspected_vessels": ranked
    }


# =========================
# AIS TRAJECTORIES
# =========================

@app.get("/trajectories")
def get_trajectories():

    TRAJECTORIES = get_trajectories_from_ais_csv()
    return {
        "data_mode": "Prototype / Simulated AIS",
        "trajectories": TRAJECTORIES
    }


# =========================
# DASHBOARD
# =========================

@app.get("/dashboard")
def dashboard():

    VESSELS = get_vessels_from_ais_csv()

    return {
        "total_vessels": len(VESSELS),
        "system_status": "Operational",
        "module": "Oil Spill Detection + AIS Correlation",
        "data_mode": "Prototype / Simulated AIS",
        "features": [
            "Satellite image analysis",
            "Spill area estimation",
            "AIS vessel correlation",
            "Vessel risk ranking",
            "Trajectory visualization",
            "Spill drift visualization"
        ]
    }


# =========================
# INVESTIGATION API
# =========================

@app.post("/api/investigations")
def create_investigation_api(payload: Dict[str, Any]):
    try:
        return create_investigation(payload)
    except Exception as exc:  # pragma: no cover - defensive fallback
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/api/investigations/{investigation_id}")
def get_investigation_api(investigation_id: str):
    try:
        return get_investigation(investigation_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - defensive fallback
        raise HTTPException(status_code=404, detail=str(exc))


@app.post("/api/investigations/{investigation_id}/analyze")
def analyze_investigation_api(investigation_id: str):
    try:
        return analyze_investigation(investigation_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - defensive fallback
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/api/investigations/{investigation_id}/vessels")
def get_vessels_api(investigation_id: str):
    try:
        return get_vessels_for_investigation(investigation_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/investigations/{investigation_id}/vessels/{vessel_id}")
def get_vessel_api(investigation_id: str, vessel_id: str):
    try:
        return get_vessel_for_investigation(investigation_id, vessel_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc