from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from detection import detect_oil_spill
from vessel_matching import rank_vessels


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
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================
# DEMO AIS DATA
# =========================

VESSELS = [
    {
        "mmsi": "419001234",
        "name": "MV Ocean Star",
        "lat": 20.45,
        "lon": 68.32
    },
    {
        "mmsi": "419005678",
        "name": "MV Sea Trader",
        "lat": 20.80,
        "lon": 68.70
    },
    {
        "mmsi": "419009876",
        "name": "MV Blue Horizon",
        "lat": 21.20,
        "lon": 69.10
    },
    {
        "mmsi": "419004321",
        "name": "MV Indian Pearl",
        "lat": 19.90,
        "lon": 68.00
    }
]


# =========================
# DEMO AIS TRAJECTORIES
# =========================

TRAJECTORIES = {
    "419001234": [
        [20.10, 68.10],
        [20.20, 68.18],
        [20.30, 68.25],
        [20.40, 68.30],
        [20.45, 68.32]
    ],

    "419005678": [
        [21.10, 68.90],
        [21.00, 68.85],
        [20.90, 68.78],
        [20.80, 68.70]
    ],

    "419009876": [
        [21.60, 69.40],
        [21.45, 69.30],
        [21.30, 69.20],
        [21.20, 69.10]
    ],

    "419004321": [
        [19.50, 67.60],
        [19.65, 67.75],
        [19.80, 67.90],
        [19.90, 68.00]
    ]
}


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

    return {
        "data_mode": "Prototype / Simulated AIS",
        "trajectories": TRAJECTORIES
    }


# =========================
# DASHBOARD
# =========================

@app.get("/dashboard")
def dashboard():

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