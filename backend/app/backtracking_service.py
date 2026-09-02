"""
Physics-based Metocean Backtracking Service for OceanGuard AI.
Implements Lagrangian surface oil spill drift modeling (3% wind leeway factor + Coriolis deflection + 100% surface ocean current)
to reconstruct probable spill discharge origins and confidence zones backward in time.
"""
from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Tuple


def compute_drift_vector(
    wind_speed_knots: float,
    wind_direction_deg: float,
    current_speed_ms: float,
    current_direction_deg: float,
    latitude: float
) -> Tuple[float, float, float, float]:
    """
    Computes resultant surface drift velocity vector (u_east, v_north in m/s) and (speed, direction).
    Formula: v_drift = v_current + (0.03 * v_wind with Coriolis leeway deflection).
    """
    # 1. Wind component: 1 knot = 0.514444 m/s. Leeway factor = 3% (0.03)
    wind_speed_ms = wind_speed_knots * 0.514444
    leeway_speed_ms = wind_speed_ms * 0.03

    # Coriolis deflection angle: ~10 degrees clockwise in Northern Hemisphere, counter-clockwise in Southern Hemisphere
    coriolis_deflection = 10.0 if latitude >= 0 else -10.0
    # Meteorological wind direction is "coming from" (blowing towards = wind_dir + 180)
    wind_blow_towards_deg = (wind_direction_deg + 180.0 + coriolis_deflection) % 360.0
    wind_rad = math.radians(wind_blow_towards_deg)

    u_wind = leeway_speed_ms * math.sin(wind_rad)
    v_wind = leeway_speed_ms * math.cos(wind_rad)

    # 2. Ocean current component: ocean current direction is standard "flowing towards"
    curr_rad = math.radians(current_direction_deg)
    u_curr = current_speed_ms * math.sin(curr_rad)
    v_curr = current_speed_ms * math.cos(curr_rad)

    # 3. Superposition
    u_total = u_curr + u_wind
    v_total = v_curr + v_wind

    total_speed_ms = math.sqrt(u_total**2 + v_total**2)
    total_direction_deg = (math.degrees(math.atan2(u_total, v_total)) + 360.0) % 360.0

    return u_total, v_total, total_speed_ms, total_direction_deg


def reconstruct_probable_origin(
    spill_latitude: float,
    spill_longitude: float,
    observation_time_iso: str,
    wind_speed_knots: float,
    wind_direction_deg: float,
    current_speed_ms: float,
    current_direction_deg: float,
    estimated_drift_hours: float = 3.5,
    num_steps: int = 7
) -> Dict[str, Any]:
    """
    Integrates the Lagrangian drift equation backward in time to determine the origin zone and trajectory.
    """
    u_ms, v_ms, speed_ms, direction_deg = compute_drift_vector(
        wind_speed_knots=wind_speed_knots,
        wind_direction_deg=wind_direction_deg,
        current_speed_ms=current_speed_ms,
        current_direction_deg=current_direction_deg,
        latitude=spill_latitude
    )

    # Parse observation timestamp
    try:
        obs_dt = datetime.fromisoformat(observation_time_iso.replace("Z", "+00:00"))
    except Exception:
        obs_dt = datetime.now(timezone.utc)

    # Conversion factors for meters to degrees
    lat_deg_per_meter = 1.0 / 111_139.0
    lon_deg_per_meter = 1.0 / (111_139.0 * math.cos(math.radians(spill_latitude)))

    # Backtrack in steps
    step_duration_hours = estimated_drift_hours / num_steps
    step_duration_sec = step_duration_hours * 3600.0

    route_points: List[Dict[str, Any]] = []
    curr_lat = spill_latitude
    curr_lon = spill_longitude
    curr_dt = obs_dt

    # Add observed centroid (t = 0)
    route_points.append({
        "latitude": round(curr_lat, 6),
        "longitude": round(curr_lon, 6),
        "timestamp": curr_dt.isoformat(),
        "step_hours_ago": 0.0
    })

    # Integrate backwards: displacement = - velocity * dt
    for step in range(1, num_steps + 1):
        dx_m = -(u_ms * step_duration_sec)
        dy_m = -(v_ms * step_duration_sec)

        curr_lat += dy_m * lat_deg_per_meter
        curr_lon += dx_m * lon_deg_per_meter
        curr_dt -= timedelta(seconds=step_duration_sec)

        route_points.append({
            "latitude": round(curr_lat, 6),
            "longitude": round(curr_lon, 6),
            "timestamp": curr_dt.isoformat(),
            "step_hours_ago": round(step * step_duration_hours, 2)
        })

    # Reorder route points from earliest (probable origin) to latest (observed spill)
    route_chronological = list(reversed(route_points))
    probable_origin = route_chronological[0]

    # Calculate uncertainty radius based on drift hours and wind speed variance
    # Empirical drift diffusion coefficient: ~0.8 km per drift hour + wind variance factor
    uncertainty_radius_km = round(1.2 + (estimated_drift_hours * 0.95) + (wind_speed_knots * 0.04), 2)
    backtracking_confidence = max(65, min(96, int(95 - (estimated_drift_hours * 3.5))))

    # Time window for origin
    start_time_iso = (obs_dt - timedelta(hours=estimated_drift_hours + 0.75)).isoformat()
    end_time_iso = (obs_dt - timedelta(hours=estimated_drift_hours - 0.75)).isoformat()

    return {
        "probableOrigin": {
            "latitude": probable_origin["latitude"],
            "longitude": probable_origin["longitude"]
        },
        "timeWindow": {
            "start": start_time_iso,
            "end": end_time_iso,
            "estimatedDriftHours": estimated_drift_hours
        },
        "backtrackingConfidence": backtracking_confidence,
        "uncertaintyRadiusKm": uncertainty_radius_km,
        "uncertainty": "Moderate" if backtracking_confidence >= 75 else "High",
        "route": route_chronological,
        "driftParameters": {
            "resultantSpeedMs": round(speed_ms, 3),
            "resultantDirectionDeg": round(direction_deg, 1),
            "windLeewayFactor": 0.03,
            "coriolisDeflectionDeg": 10.0 if spill_latitude >= 0 else -10.0,
            "governingPhysics": "Eulerian-Lagrangian Superposition (v_drift = v_current + 0.03 * v_wind)"
        },
        "description": (
            f"Lagrangian drift model reconstructed probable origin {round(probable_origin['latitude'], 4)}° N, "
            f"{round(probable_origin['longitude'], 4)}° E (~{round(speed_ms * 3.6 * estimated_drift_hours, 1)} km up-drift) "
            f"based on {wind_speed_knots} kt winds and {current_speed_ms} m/s surface currents."
        )
    }
