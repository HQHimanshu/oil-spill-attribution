from __future__ import annotations

import json
import math
from urllib.parse import urlencode
from urllib.request import urlopen
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

from .ais_data import get_vessels_from_ais_csv


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6371.0
    lat1_r = math.radians(lat1)
    lat2_r = math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlon / 2) ** 2
    )
    return 2 * radius * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _score_vessel(distance_km: float) -> int:
    score = int(max(0, 100 - distance_km * 0.8))
    return max(10, min(99, score))


def _get_live_environment(latitude: float, longitude: float) -> Dict[str, Any]:
    query = urlencode({
        "latitude": latitude,
        "longitude": longitude,
        "current": "wind_speed_10m,wind_direction_10m",
        "wind_speed_unit": "kn",
        "timezone": "UTC",
    })
    marine_query = urlencode({
        "latitude": latitude,
        "longitude": longitude,
        "current": "ocean_current_velocity,ocean_current_direction",
        "timezone": "UTC",
    })

    try:
        with urlopen(f"https://api.open-meteo.com/v1/forecast?{query}", timeout=8) as response:
            weather = json.loads(response.read().decode("utf-8"))
        with urlopen(f"https://marine-api.open-meteo.com/v1/marine?{marine_query}", timeout=8) as response:
            marine = json.loads(response.read().decode("utf-8"))

        weather_current = weather.get("current", {})
        marine_current = marine.get("current", {})
        return {
            "wind": {
                "speed": weather_current.get("wind_speed_10m"),
                "direction": weather_current.get("wind_direction_10m"),
                "unit": "knots",
            },
            "current": {
                "speed": marine_current.get("ocean_current_velocity"),
                "direction": marine_current.get("ocean_current_direction"),
                "unit": "m/s",
            },
            "timestamp": weather_current.get("time"),
            "source": "Open-Meteo live weather and marine APIs",
            "status": "AVAILABLE",
        }
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return {
            "wind": {"speed": None, "direction": None, "unit": "knots"},
            "current": {"speed": None, "direction": None, "unit": "m/s"},
            "timestamp": None,
            "source": "Live environmental service unavailable",
            "status": "UNAVAILABLE",
        }


def build_demo_investigation() -> Dict[str, Any]:
    latitude = 20.45
    longitude = 68.32
    observation_time = "2024-02-11T10:30:00Z"
    origin_start = "2024-02-11T08:30:00Z"
    origin_end = "2024-02-11T10:00:00Z"
    vessels = get_vessels_from_ais_csv()
    ranked: List[Dict[str, Any]] = []

    for index, vessel in enumerate(vessels[:8], start=1):
        distance = _haversine_km(latitude, longitude, float(vessel["LAT"]), float(vessel["LON"]))
        score = _score_vessel(distance)
        time_match = min(100, max(55, score + 5))
        trajectory_match = min(100, max(50, score + 8))
        evidence = {
            "score": score,
            "timeMatch": time_match,
            "trajectoryMatch": trajectory_match,
            "proximity": min(100, max(20, score)),
            "aisCompleteness": 92,
        }
        ranked.append(
            {
                "id": str(vessel.get("MMSI", f"VESSEL-{index}")),
                "name": vessel.get("VesselName", "UnIdentified"),
                "mmsi": vessel.get("MMSI", str(index)),
                "latitude": float(vessel.get("LAT", latitude)),
                "longitude": float(vessel.get("LON", longitude)),
                "distanceKm": round(distance, 2),
                "riskLevel": "HIGH" if score >= 75 else "MEDIUM" if score >= 45 else "LOW",
                "evidenceScore": score,
                "timeMatch": int(time_match),
                "trajectoryMatch": int(trajectory_match),
                "proximity": int(min(100, max(20, score))),
                "aisCompleteness": 92,
                "status": "Candidate",
                "position": {
                    "latitude": float(vessel.get("LAT", latitude)),
                    "longitude": float(vessel.get("LON", longitude)),
                },
                "evidence": evidence,
                "rank": index,
            }
        )

    ranked.sort(key=lambda item: item["evidenceScore"], reverse=True)
    for index, vessel in enumerate(ranked, start=1):
        vessel["rank"] = index

    return {
        "id": "demo-investigation",
        "status": "ready",
        "createdAt": observation_time,
        "input": {
            "latitude": latitude,
            "longitude": longitude,
            "timestamp": observation_time,
            "metadataStatus": "valid",
            "fileName": "sample_sar_scene.png",
            "dataSource": "Satellite observation",
        },
        "dataAvailability": {
            "satelliteObservation": "AVAILABLE",
            "environmentalData": "AVAILABLE",
            "historicalAIS": "LOCAL DATASET",
            "status": "READY FOR ANALYSIS",
        },
        "spill": {
            "detected": True,
            "confidence": 94,
            "area": 12.6,
            "areaUnit": "km²",
            "observationTime": observation_time,
            "centroid": {"latitude": latitude, "longitude": longitude},
            "status": "SPILL DETECTED",
            "detectionStatus": "confirmed",
        },
        "origin": {
            "probableOrigin": {
                "latitude": latitude - 0.36,
                "longitude": longitude - 0.48,
            },
            "timeWindow": {
                "start": origin_start,
                "end": origin_end,
            },
            "backtrackingConfidence": 82,
            "route": [
                {"latitude": latitude - 0.36, "longitude": longitude - 0.48},
                {"latitude": latitude, "longitude": longitude},
            ],
            "uncertainty": "Moderate",
            "description": "Drift model reconstructed a likely origin zone north-west of the observed spill centroid.",
        },
        "environment": {
            "wind": {"speed": 18.4, "direction": 142, "unit": "knots"},
            "current": {"speed": 1.9, "direction": 68, "unit": "m/s"},
            "timestamp": observation_time,
            "source": "Modelled metocean product",
            "status": "AVAILABLE",
        },
        "vessels": ranked,
        "selectedVessel": ranked[0] if ranked else None,
        "uncertainty": {
            "confidence": "MODERATE",
            "sources": [
                "SAR detection uncertainty",
                "Environmental-model uncertainty",
                "Backtracking uncertainty",
                "AIS coverage gaps",
            ],
            "disclaimer": "This analysis provides decision support based on available satellite, environmental and AIS evidence. It does not establish legal responsibility.",
        },
        "analysisTimeline": [
            {"label": "SAR image received", "status": "completed", "timestamp": observation_time},
            {"label": "Validating metadata", "status": "completed", "timestamp": "2024-02-11T10:31:00Z"},
            {"label": "Detecting spill", "status": "completed", "timestamp": "2024-02-11T10:33:00Z"},
            {"label": "Characterizing spill", "status": "completed", "timestamp": "2024-02-11T10:35:00Z"},
            {"label": "Retrieving environmental data", "status": "completed", "timestamp": "2024-02-11T10:36:00Z"},
            {"label": "Running backtracking", "status": "completed", "timestamp": "2024-02-11T10:37:00Z"},
            {"label": "Reconstructing probable origin", "status": "completed", "timestamp": "2024-02-11T10:38:00Z"},
            {"label": "Retrieving historical AIS", "status": "completed", "timestamp": "2024-02-11T10:39:00Z"},
            {"label": "Filtering candidate vessels", "status": "completed", "timestamp": "2024-02-11T10:40:00Z"},
            {"label": "Ranking candidates", "status": "completed", "timestamp": "2024-02-11T10:41:00Z"},
        ],
    }


INVESTIGATIONS: Dict[str, Dict[str, Any]] = {
    "demo-investigation": build_demo_investigation(),
}


def create_investigation(input_payload: Dict[str, Any]) -> Dict[str, Any]:
    latitude = float(input_payload.get("latitude", 20.45))
    longitude = float(input_payload.get("longitude", 68.32))
    timestamp = input_payload.get("timestamp") or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    investigation_id = input_payload.get("id") or f"INV-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
    investigation = build_demo_investigation()
    investigation["id"] = investigation_id
    investigation["status"] = "ready"
    investigation["createdAt"] = timestamp
    investigation["input"] = {
        "latitude": latitude,
        "longitude": longitude,
        "timestamp": timestamp,
        "metadataStatus": input_payload.get("metadata", {}).get("status", "valid"),
        "fileName": input_payload.get("fileName") or "uploaded_scene.png",
        "dataSource": input_payload.get("dataSource") or "User upload",
    }
    investigation["spill"]["centroid"] = {"latitude": latitude, "longitude": longitude}
    investigation["spill"]["observationTime"] = timestamp
    investigation["environment"] = _get_live_environment(latitude, longitude)
    investigation["dataAvailability"]["environmentalData"] = investigation["environment"]["status"]
    investigation["dataAvailability"]["status"] = (
        "READY FOR ANALYSIS" if investigation["environment"]["status"] == "AVAILABLE"
        else "ENVIRONMENTAL DATA UNAVAILABLE"
    )
    investigation["origin"]["probableOrigin"] = {"latitude": latitude - 0.35, "longitude": longitude - 0.45}
    investigation["origin"]["timeWindow"]["start"] = (datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%SZ") - timedelta(hours=2)).strftime("%Y-%m-%dT%H:%M:%SZ")
    investigation["origin"]["timeWindow"]["end"] = timestamp
    investigation["analysisTimeline"][0]["timestamp"] = timestamp
    INVESTIGATIONS[investigation_id] = investigation
    return investigation


def get_investigation(investigation_id: str) -> Dict[str, Any]:
    if investigation_id not in INVESTIGATIONS:
        raise KeyError(f"Investigation '{investigation_id}' not found.")
    return INVESTIGATIONS[investigation_id]


def analyze_investigation(investigation_id: str) -> Dict[str, Any]:
    investigation = get_investigation(investigation_id)
    investigation["status"] = "analyzing"
    investigation["analysisTimeline"] = [
        {"label": "SAR image received", "status": "completed", "timestamp": investigation["createdAt"]},
        {"label": "Validating metadata", "status": "completed", "timestamp": investigation["createdAt"]},
        {"label": "Detecting spill", "status": "active", "timestamp": investigation["createdAt"]},
        {"label": "Characterizing spill", "status": "pending", "timestamp": None},
        {"label": "Retrieving environmental data", "status": "pending", "timestamp": None},
        {"label": "Running backtracking", "status": "pending", "timestamp": None},
        {"label": "Reconstructing probable origin", "status": "pending", "timestamp": None},
        {"label": "Retrieving historical AIS", "status": "pending", "timestamp": None},
        {"label": "Filtering candidate vessels", "status": "pending", "timestamp": None},
        {"label": "Ranking candidates", "status": "pending", "timestamp": None},
    ]
    investigation["status"] = "ready"
    return investigation


def get_vessels_for_investigation(investigation_id: str) -> List[Dict[str, Any]]:
    investigation = get_investigation(investigation_id)
    return investigation.get("vessels", [])


def get_vessel_for_investigation(investigation_id: str, vessel_id: str) -> Dict[str, Any]:
    investigation = get_investigation(investigation_id)
    for vessel in investigation.get("vessels", []):
        if str(vessel.get("id")) == str(vessel_id):
            return vessel
    selected = investigation.get("selectedVessel") or {}
    if str(selected.get("id")) == str(vessel_id):
        return selected
    raise KeyError(f"Vessel '{vessel_id}' not found in investigation '{investigation_id}'.")
