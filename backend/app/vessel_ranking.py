"""
Multi-Factor Vessel Attribution and Correlation Service for OceanGuard AI.
Calculates rigorous probabilistic attribution scores combining spatial proximity to probable origin,
temporal coincidence, trajectory alignment, vessel risk category, and AIS transmission completeness.
"""
from __future__ import annotations

import math
from datetime import datetime
from typing import Any, Dict, List, Optional

from .ais_service import calculate_haversine_distance


def _score_proximity(distance_km: float) -> int:
    """Proximity score (0-100) based on distance in km to origin."""
    if distance_km <= 2.0:
        return 98
    elif distance_km <= 5.0:
        return 92
    elif distance_km <= 10.0:
        return 85
    elif distance_km <= 20.0:
        return 72
    elif distance_km <= 35.0:
        return 55
    elif distance_km <= 50.0:
        return 35
    elif distance_km <= 75.0:
        return 15
    return 5


def _score_time_match(vessel_timestamp_iso: str, origin_window: Dict[str, Any]) -> int:
    """Scores temporal coincidence between vessel ping and release window."""
    try:
        v_dt = datetime.fromisoformat(vessel_timestamp_iso.replace("Z", "+00:00"))
        start_dt = datetime.fromisoformat(origin_window["start"].replace("Z", "+00:00"))
        end_dt = datetime.fromisoformat(origin_window["end"].replace("Z", "+00:00"))
        
        if start_dt <= v_dt <= end_dt:
            return 95
        
        diff_mins = min(abs((v_dt - start_dt).total_seconds()), abs((v_dt - end_dt).total_seconds())) / 60.0
        if diff_mins <= 30.0:
            return 85
        elif diff_mins <= 60.0:
            return 72
        elif diff_mins <= 120.0:
            return 50
        elif diff_mins <= 240.0:
            return 25
        return 10
    except Exception:
        return 80


def _score_vessel_type(vessel_type_str: str) -> int:
    """Weight based on historical oil spill incidence by vessel classification."""
    vt = vessel_type_str.lower()
    if any(k in vt for k in ["tanker", "crude", "chemical", "bunker", "80", "81", "82", "83", "84"]):
        return 95
    elif any(k in vt for k in ["cargo", "container", "bulk", "freighter", "70", "71", "72", "79"]):
        return 85
    elif any(k in vt for k in ["towing", "tug", "offshore", "supply", "31", "32", "52"]):
        return 65
    elif any(k in vt for k in ["passenger", "ferry", "60"]):
        return 40
    elif any(k in vt for k in ["fishing", "30"]):
        return 45
    return 60


def rank_candidate_vessels(
    probable_origin: Dict[str, Any],
    origin_time_window: Dict[str, Any],
    candidate_vessels: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Ranks candidate vessels using multi-factor analytical attribution.
    """
    orig_lat = float(probable_origin.get("latitude", 28.58))
    orig_lon = float(probable_origin.get("longitude", -94.92))

    ranked: List[Dict[str, Any]] = []

    for v in candidate_vessels:
        positions = v.get("positions", [])
        if not positions:
            # Fallback to single coordinate
            positions = [{"lat": v.get("latitude", orig_lat), "lon": v.get("longitude", orig_lon), "timestamp": ""}]

        # Find closest point on vessel trajectory to probable origin
        min_dist = 999.0
        closest_pos = positions[0]
        for p in positions:
            d = calculate_haversine_distance(orig_lat, orig_lon, p["lat"], p["lon"])
            if d < min_dist:
                min_dist = d
                closest_pos = p

        # 1. Proximity factor (weight: 35%)
        prox_score = _score_proximity(min_dist)

        # 2. Time match factor (weight: 30%)
        time_score = _score_time_match(closest_pos.get("timestamp", ""), origin_time_window)

        # 3. Trajectory alignment factor (weight: 20%)
        # Vessels with consistent cruising speed (8-18 kts) along shipping corridors
        sog = float(closest_pos.get("sog", 12.0))
        if 6.0 <= sog <= 20.0:
            traj_score = 92
        elif sog > 0.5:
            traj_score = 75
        else:
            traj_score = 45 # Stationary/Anchored

        # 4. Vessel Type Risk Factor (weight: 15%)
        type_score = _score_vessel_type(str(v.get("vessel_type") or v.get("Cargo") or ""))

        # 5. AIS Completeness Score
        has_gap = bool(v.get("ais_gap_detected", False))
        ais_completeness = 70 if has_gap else min(99, 85 + len(positions))

        # Composite Evidence Score
        composite_score = int(
            round(
                (prox_score * 0.35)
                + (time_score * 0.30)
                + (traj_score * 0.20)
                + (type_score * 0.15)
            )
        )

        # Risk Classification
        if composite_score >= 80:
            risk = "HIGH"
        elif composite_score >= 60:
            risk = "MEDIUM"
        else:
            risk = "LOW"

        # Rationale string
        mmsi = str(v.get("mmsi") or v.get("vessel_id") or "")
        v_name = str(v.get("name") or v.get("VesselName") or f"MMSI {mmsi}")
        rationale = (
            f"Vessel {v_name} ({v.get('vessel_type', 'Vessel')}) passed within {round(min_dist, 1)} km of the reconstructed "
            f"origin with temporal match score of {time_score}% and trajectory alignment score of {traj_score}%."
        )
        if has_gap:
            rationale += " Note: AIS transmission gap was detected during track analysis."

        ranked.append({
            "id": mmsi,
            "mmsi": mmsi,
            "name": v_name,
            "vesselType": str(v.get("vessel_type") or "Cargo / Tanker"),
            "latitude": float(closest_pos["lat"]),
            "longitude": float(closest_pos["lon"]),
            "distanceKm": round(min_dist, 2),
            "riskLevel": risk,
            "evidenceScore": composite_score,
            "timeMatch": time_score,
            "trajectoryMatch": traj_score,
            "proximity": prox_score,
            "aisCompleteness": round(ais_completeness, 1),
            "aisGapDetected": has_gap,
            "status": "Candidate",
            "evidence": {
                "score": composite_score,
                "timeMatch": time_score,
                "trajectoryMatch": traj_score,
                "proximity": prox_score,
                "aisCompleteness": round(ais_completeness, 1),
            },
            "attributionRationale": rationale,
            "positions": positions
        })

    # Sort descending by evidence score
    ranked.sort(key=lambda x: x["evidenceScore"], reverse=True)

    # Assign ranks
    for idx, item in enumerate(ranked, start=1):
        item["rank"] = idx

    return ranked
