"""
Unified ML Training Pipeline for OceanGuard AI.
Trains both:
1. SAR Sentinel-1 / PALSAR Oil Spill Segmentation Model (RandomForest v2.1)
2. AIS Spill Location Regression Model (RandomForest Regressor)
Evaluates metrics and exports metadata.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.training.train import train_model
from ml.training.evaluate import evaluate_model
from ml.train_spill_location_model import main as train_spill_location


def run_full_training():
    start_time = time.time()
    print("=" * 70)
    print("[START] OCEANGUARD AI UNIFIED ML TRAINING PIPELINE")
    print("=" * 70)

    # 1. Train SAR Segmentation Model on PALSAR Dataset
    print("\n[STEP 1/3] Training SAR Oil Spill Segmentation Model...")
    seg_model, seg_meta = train_model()

    # 2. Evaluate Segmentation Model
    print("\n[STEP 2/3] Evaluating Segmentation Model on Held-out Scenes...")
    eval_metrics = evaluate_model()

    # 3. Train AIS Spill Location Regressor
    print("\n[STEP 3/3] Training AIS Probable Spill Location Regressor...")
    train_spill_location()

    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"[SUCCESS] ML TRAINING PIPELINE COMPLETED IN {elapsed:.1f}s")
    print("=" * 70)
    return {
        "status": "success",
        "elapsed_seconds": round(elapsed, 2),
        "segmentation_metrics": eval_metrics,
        "model_version": seg_meta.get("model_version", "v2.1"),
        "dataset_version": seg_meta.get("dataset_version", "PALSAR / SAR Benchmark v2.1")
    }


if __name__ == "__main__":
    run_full_training()
