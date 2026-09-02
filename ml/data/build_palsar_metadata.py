"""
Builds a comprehensive metadata catalog for the authentic PALSAR SAR oil spill dataset
containing 8,070 scenes (6,455 train / 1,615 val).
Computes pixel statistics, slick presence, slick area ratio, and image dimensions.
"""
from __future__ import annotations

import os
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
IMAGES_DIR = PROJECT_ROOT / "images"
MASKS_DIR = PROJECT_ROOT / "masks"
METADATA_OUT = PROJECT_ROOT / "ml" / "data" / "palsar_metadata.csv"
RAW_METADATA_OUT = PROJECT_ROOT / "ml" / "data" / "raw" / "metadata.csv"


def process_scene(scene_tuple):
    split, filename = scene_tuple
    img_rel = f"images/{split}/{filename}"
    mask_rel = f"masks/{split}/{filename}"
    
    img_path = PROJECT_ROOT / img_rel
    mask_path = PROJECT_ROOT / mask_rel
    
    if not img_path.exists() or not mask_path.exists():
        return None
        
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
        
    h, w = mask.shape[:2]
    total_pixels = h * w
    spill_pixels = int(np.count_nonzero(mask > 127))
    spill_ratio = round(float(spill_pixels / total_pixels), 5)
    has_spill = bool(spill_pixels >= 50)
    
    # Estimate approximate real-world area (assuming 10m PALSAR pixel resolution)
    area_km2 = round(float(spill_pixels * 100.0 / 1_000_000.0), 4)

    return {
        "scene_id": Path(filename).stem,
        "split": split,
        "filename": filename,
        "image_path": img_rel,
        "mask_path": mask_rel,
        "height": h,
        "width": w,
        "has_spill": has_spill,
        "spill_pixels": spill_pixels,
        "spill_area_ratio": spill_ratio,
        "estimated_area_km2": area_km2,
        "sensor": "ALOS PALSAR / SAR C-Band"
    }


def build_metadata(max_workers: int = 16) -> pd.DataFrame:
    print("=" * 65)
    print("Building PALSAR SAR Dataset Catalog (Train & Val)...")
    print("=" * 65)
    
    tasks = []
    for split in ["train", "val"]:
        split_img_dir = IMAGES_DIR / split
        if split_img_dir.exists():
            for fname in os.listdir(split_img_dir):
                if fname.lower().endswith((".png", ".jpg", ".jpeg", ".tif")):
                    tasks.append((split, fname))
                    
    print(f"Discovered {len(tasks)} candidate SAR scene pairs.")
    
    records = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for res in executor.map(process_scene, tasks):
            if res is not None:
                records.append(res)
                
    df = pd.DataFrame(records)
    # Sort by split and scene_id
    df = df.sort_values(["split", "scene_id"]).reset_index(drop=True)
    
    METADATA_OUT.parent.mkdir(parents=True, exist_ok=True)
    RAW_METADATA_OUT.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(METADATA_OUT, index=False)
    df.to_csv(RAW_METADATA_OUT, index=False)
    
    train_count = len(df[df["split"] == "train"])
    val_count = len(df[df["split"] == "val"])
    spill_count = len(df[df["has_spill"]])
    clean_count = len(df[~df["has_spill"]])
    
    print(f"\nMetadata Catalog successfully generated:")
    print(f"  • Total scenes:     {len(df):,}")
    print(f"  • Train scenes:     {train_count:,}")
    print(f"  • Val scenes:       {val_count:,}")
    print(f"  • Oil spill scenes: {spill_count:,} ({spill_count / len(df) * 100:.1f}%)")
    print(f"  • Clean/Neg scenes: {clean_count:,} ({clean_count / len(df) * 100:.1f}%)")
    print(f"  • Saved catalog to: {METADATA_OUT}")
    
    return df


if __name__ == "__main__":
    build_metadata()
