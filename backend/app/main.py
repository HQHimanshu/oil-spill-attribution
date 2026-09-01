from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path

from detection import detect_oil_spill
from vessel_matching import rank_vessels

from ais_data import get_vessels_from_ais_csv, get_trajectories_from_ais_csv
from get_metadata import get_metadata_for_image

import random

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

        image_id = Path(file.filename).stem

        # Find metadata
        metadata = get_metadata_for_image(image_id)

        if metadata is None:

    # Fallback coordinates
            latitude = random.uniform(-60, 60)
            longitude = random.uniform(-180, 180)

            result["coordinate_source"] = "Random fallback"

        else:

            latitude = metadata["latitude"]
            longitude = metadata["longitude"]

            result["coordinate_source"] = "Image metadata"


        result["image_id"] = image_id
        result["latitude"] = latitude
        result["longitude"] = longitude

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