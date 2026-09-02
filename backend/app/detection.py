"""
SAR Oil Spill Detection Engine powered by trained Machine Learning Segmentation Model.
"""
from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path
from typing import Any, Dict

from ml.characterization.inference import get_sar_inference_engine


def detect_oil_spill(
    image_bytes: bytes,
    center_lat: float = 28.582,
    center_lon: float = -94.925
) -> Dict[str, Any]:
    """
    Runs ML segmentation model inference on uploaded SAR image bytes.
    """
    engine = get_sar_inference_engine()
    inference_result = engine.run_inference(
        image_input=image_bytes,
        center_lat=center_lat,
        center_lon=center_lon
    )

    return {
        "spill_detected": inference_result["detected"],
        "confidence": inference_result["confidence"],
        "area_km2": inference_result["area_km2"],
        "centroid": inference_result["centroid"],
        "boundary": inference_result["boundary"],
        "mask": inference_result["mask"],
        "detection_method": "Trained SAR Multi-Scale Texture-Backscatter Random Forest",
        "model_status": "Operational (v2.1)",
        "model_provenance": inference_result.get("model_provenance", {})
    }