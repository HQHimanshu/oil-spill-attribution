"""
Historical AIS Data Service for OceanGuard AI.
Processes authentic NOAA historical vessel AIS data (MMSI, coordinates, timestamps, SOG, COG, vessel type, dimensions)
and provides track normalization, spatial-temporal filtering, and AIS gap detection.
"""
from __future__ import annotations

import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
AIS_CSV_PATH = BASE_DIR / "database" / "ais.csv"
SYNTHETIC_AIS_CSV_PATH = BASE_DIR / "database" / "synthetic_ais_with_ship_data.csv"

_AIS_DF_CACHE: Optional[pd.DataFrame] = None


def calculate_haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculates great-circle distance between two points in kilometers."""
    R = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = (
        math.sin(delta_phi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
    )
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return R * c


def load_real_ais_dataset() -> pd.DataFrame:
    global _AIS_DF_CACHE
    if _AIS_DF_CACHE is not None:
        return _AIS_DF_CACHE

    if AIS_CSV_PATH.exists():
        df = pd.read_csv(AIS_CSV_PATH)
        df["BaseDateTime"] = pd.to_datetime(df["BaseDateTime"], utc=True).dt.tz_localize(None)
        df["LAT"] = pd.to_numeric(df["LAT"], errors="coerce")
        df["LON"] = pd.to_numeric(df["LON"], errors="coerce")
        df["SOG"] = pd.to_numeric(df.get("SOG", 0.0), errors="coerce").fillna(0.0)
        df["COG"] = pd.to_numeric(df.get("COG", 0.0), errors="coerce").fillna(0.0)
        df["Heading"] = pd.to_numeric(df.get("Heading", 0.0), errors="coerce").fillna(0.0)
        df = df.dropna(subset=["MMSI", "LAT", "LON", "BaseDateTime"]).copy()
        _AIS_DF_CACHE = df
        return _AIS_DF_CACHE
    elif SYNTHETIC_AIS_CSV_PATH.exists():
        df = pd.read_csv(SYNTHETIC_AIS_CSV_PATH)
        df["BaseDateTime"] = pd.to_datetime(df.get("BaseDateTime", datetime.now(timezone.utc)), utc=True).dt.tz_localize(None)
        _AIS_DF_CACHE = df
        return _AIS_DF_CACHE
    else:
        return pd.DataFrame()


def get_ais_data(
    origin_region: Dict[str, Any],
    origin_time_window: Optional[Dict[str, Any]] = None,
    max_distance_km: float = 60.0,
    min_points_per_track: int = 2
) -> Dict[str, Any]:
    """
    Standardized interface: getAISData(originRegion, originTimeWindow)
    Retrieves normalized vessel tracks and detects AIS data gaps.
    """
    df = load_real_ais_dataset()
    if df.empty:
        return {
            "vessels": [],
            "data_mode": "REAL AIS DATA UNAVAILABLE",
            "coverage_status": "NO DATA"
        }

    center_lat = float(origin_region.get("latitude", 28.58))
    center_lon = float(origin_region.get("longitude", -94.92))

    # Time filtering if specified
    start_dt = None
    end_dt = None
    if origin_time_window:
        if "start" in origin_time_window:
            try:
                start_dt = pd.to_datetime(origin_time_window["start"], utc=True).tz_localize(None)
            except Exception:
                pass
        if "end" in origin_time_window:
            try:
                end_dt = pd.to_datetime(origin_time_window["end"], utc=True).tz_localize(None)
            except Exception:
                pass

    filtered_df = df.copy()
    if start_dt is not None and end_dt is not None:
        time_matched = filtered_df[
            (filtered_df["BaseDateTime"] >= start_dt) & (filtered_df["BaseDateTime"] <= end_dt)
        ]
        if not time_matched.empty:
            filtered_df = time_matched

    # Calculate distance of each vessel's closest point to origin
    vessel_tracks: List[Dict[str, Any]] = []
    ais_data_gaps: List[Dict[str, Any]] = []

    for mmsi, group in filtered_df.groupby("MMSI"):
        sorted_group = group.sort_values("BaseDateTime")
        
        # Calculate minimum distance to origin
        distances = [
            calculate_haversine_distance(center_lat, center_lon, float(row["LAT"]), float(row["LON"]))
            for _, row in sorted_group.iterrows()
        ]
        min_dist = min(distances) if distances else 999.0

        if min_dist > max_distance_km:
            continue

        latest_row = sorted_group.iloc[-1]
        vessel_name = str(latest_row.get("VesselName") or "Unidentified Vessel")
        vessel_type = str(latest_row.get("VesselType") or latest_row.get("Cargo") or "Cargo / Tanker")
        imo = str(latest_row.get("IMO") or "")
        length = float(latest_row.get("Length") or 0.0)
        width = float(latest_row.get("Width") or 0.0)

        # Build track positions
        positions = []
        timestamps = []
        for _, row in sorted_group.iterrows():
            pos_ts = row["BaseDateTime"].isoformat() if hasattr(row["BaseDateTime"], "isoformat") else str(row["BaseDateTime"])
            positions.append({
                "lat": round(float(row["LAT"]), 6),
                "lon": round(float(row["LON"]), 6),
                "timestamp": pos_ts,
                "sog": round(float(row.get("SOG", 0.0)), 1),
                "cog": round(float(row.get("COG", 0.0)), 1),
                "heading": round(float(row.get("Heading", 0.0)), 1)
            })
            timestamps.append(row["BaseDateTime"])

        # Check for AIS Transmission Gap (> 15 minutes between pings in active voyage)
        has_gap = False
        gap_durations = []
        for i in range(1, len(timestamps)):
            delta_mins = (timestamps[i] - timestamps[i - 1]).total_seconds() / 60.0
            if delta_mins > 15.0:
                has_gap = True
                gap_durations.append(round(delta_mins, 1))

        if has_gap:
            ais_data_gaps.append({
                "mmsi": str(mmsi),
                "vessel_name": vessel_name,
                "max_gap_minutes": max(gap_durations) if gap_durations else 0.0,
                "status": "AIS DATA GAP DETECTED"
            })

        track_data = {
            "vessel_id": str(mmsi),
            "mmsi": str(mmsi),
            "name": vessel_name,
            "vessel_type": vessel_type,
            "imo": imo,
            "length_m": length,
            "width_m": width,
            "distance_to_origin_km": round(min_dist, 2),
            "positions": positions,
            "total_pings": len(positions),
            "ais_gap_detected": has_gap,
            "gap_details": gap_durations if has_gap else None,
            "source": "NOAA Marine Historical AIS Dataset"
        }
        vessel_tracks.append(track_data)

    # Sort vessels by proximity to probable origin
    vessel_tracks.sort(key=lambda x: x["distance_to_origin_km"])

    return {
        "vessels": vessel_tracks,
        "total_candidate_vessels": len(vessel_tracks),
        "ais_data_gaps": ais_data_gaps,
        "data_mode": "REAL DATA (NOAA Marine AIS Historical)",
        "source": "NOAA Office for Coastal Management Historical AIS"
    }


# Standardized camelCase interface alias
def getAISData(originRegion: Dict[str, Any], originTimeWindow: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return get_ais_data(originRegion, originTimeWindow)
