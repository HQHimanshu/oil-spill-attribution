"""
Training Pipeline for SAR Sentinel-1 Oil Spill Segmentation Model.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
import joblib
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.characterization.model import SarOilSpillSegmenter
from ml.training.dataset import SarOilSpillDataset

CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"


def train_model():
    print("=" * 65)
    print("OceanGuard AI - SAR Oil Spill Segmentation Training Pipeline")
    print("=" * 65)

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    metadata_path = PROJECT_ROOT / config["paths"]["metadata_path"]
    model_save_path = PROJECT_ROOT / config["paths"]["model_save_path"]
    metadata_save_path = PROJECT_ROOT / config["paths"]["metadata_save_path"]

    model_save_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset from: {metadata_path}")
    dataset = SarOilSpillDataset(
        metadata_csv=metadata_path,
        random_seed=config["dataset"]["random_seed"],
        train_ratio=config["dataset"]["train_split"],
        val_ratio=config["dataset"]["val_split"],
        test_ratio=config["dataset"]["test_split"],
    )

    print(f"Train scenes: {len(dataset.train_df)} | Val scenes: {len(dataset.val_df)} | Test scenes: {len(dataset.test_df)}")

    max_train_scenes = config["dataset"].get("max_train_scenes", 60)
    max_val_scenes = config["dataset"].get("max_val_scenes", 20)
    samples_per_image = config["dataset"].get("samples_per_image", 4000)

    print(f"\nExtracting multi-scale SAR textural & backscatter features from up to {max_train_scenes} train scenes...")
    X_train, y_train = dataset.extract_training_samples(
        split="train",
        max_scenes=max_train_scenes,
        samples_per_image=samples_per_image,
        positive_ratio=config["dataset"]["positive_ratio"],
        augment=True,
    )
    print(f"Extracted {X_train.shape[0]:,} training samples across {X_train.shape[1]} feature channels.")
    print(f"Oil spill pixel samples: {int(y_train.sum()):,} ({y_train.sum()/len(y_train)*100:.1f}%) | Clean sea: {int((y_train == 0).sum()):,}")

    X_val, y_val = dataset.extract_training_samples(
        split="val",
        max_scenes=max_val_scenes,
        samples_per_image=min(2000, samples_per_image),
        positive_ratio=config["dataset"]["positive_ratio"],
        augment=False,
    )

    print("\nTraining Random Forest SAR Segmentation Model...")
    model = SarOilSpillSegmenter(
        n_estimators=config["model"]["n_estimators"],
        max_depth=config["model"]["max_depth"],
        random_state=config["dataset"]["random_seed"],
    )
    model.fit(X_train, y_train)

    train_acc = float((model.model.predict(X_train) == y_train).mean())
    val_acc = float((model.model.predict(X_val) == y_val).mean())
    print(f"Training Pixel Accuracy:   {train_acc * 100:.2f}%")
    print(f"Validation Pixel Accuracy: {val_acc * 100:.2f}%")

    # Save model with joblib compression (<40MB for GitHub compatibility)
    joblib.dump(model, model_save_path, compress=5)
    print(f"\nSaved model checkpoint to: {model_save_path}")

    # Feature importances
    feature_importances = {
        name: round(float(imp), 4)
        for name, imp in zip(model.feature_names, model.model.feature_importances_)
    }
    print("Feature Importances:")
    for name, imp in sorted(feature_importances.items(), key=lambda x: x[1], reverse=True):
        print(f"  - {name:<20}: {imp:.4f}")

    # Save metadata
    training_metadata = {
        "project": config["project_name"],
        "model_version": config["model_version"],
        "dataset_version": config["dataset_version"],
        "training_timestamp": datetime.now(timezone.utc).isoformat(),
        "model_type": config["model"]["type"],
        "parameters": config["model"],
        "feature_names": model.feature_names,
        "feature_importances": feature_importances,
        "training_samples": int(X_train.shape[0]),
        "train_pixel_accuracy": round(train_acc, 4),
        "val_pixel_accuracy": round(val_acc, 4),
        "split_counts": {
            "train_scenes": len(dataset.train_df),
            "val_scenes": len(dataset.val_df),
            "test_scenes": len(dataset.test_df),
        },
    }

    with open(metadata_save_path, "w", encoding="utf-8") as f:
        json.dump(training_metadata, f, indent=2)
    print(f"Saved training metadata to: {metadata_save_path}")

    return model, training_metadata


if __name__ == "__main__":
    train_model()
