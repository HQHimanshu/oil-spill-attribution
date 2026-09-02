"""
SAR Oil Spill Dataset Loader, Augmentation, and Split Pipeline.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple
import cv2
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.characterization.preprocessing import extract_sar_features


class SarOilSpillDataset:
    def __init__(
        self,
        metadata_csv: Path,
        random_seed: int = 42,
        train_ratio: float = 0.80,
        val_ratio: float = 0.10,
        test_ratio: float = 0.10,
    ):
        self.metadata_csv = Path(metadata_csv)
        self.random_seed = random_seed
        self.df = pd.read_csv(self.metadata_csv)
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

        # Check if pre-defined split exists in dataframe
        if "split" in self.df.columns and set(self.df["split"].unique()).issubset({"train", "val", "test"}):
            self.train_df = self.df[self.df["split"] == "train"].reset_index(drop=True)
            val_candidates = self.df[self.df["split"] == "val"].reset_index(drop=True)
            if len(val_candidates) > 0:
                half = max(1, len(val_candidates) // 2)
                self.val_df = val_candidates.iloc[:half].reset_index(drop=True)
                self.test_df = val_candidates.iloc[half:].reset_index(drop=True)
            else:
                self.val_df = val_candidates
                self.test_df = val_candidates.copy()
        else:
            # Split scenes reproducibly
            np.random.seed(self.random_seed)
            shuffled = self.df.sample(frac=1.0, random_state=self.random_seed).reset_index(drop=True)
            
            n_total = len(shuffled)
            n_train = max(1, int(n_total * self.train_ratio))
            n_val = max(1, int(n_total * self.val_ratio))
            
            self.train_df = shuffled.iloc[:n_train].reset_index(drop=True)
            self.val_df = shuffled.iloc[n_train:n_train + n_val].reset_index(drop=True)
            self.test_df = shuffled.iloc[n_train + n_val:].reset_index(drop=True)
            
        if len(self.test_df) == 0:
            self.test_df = self.val_df.copy()

    def resolve_path(self, path_str: str) -> Path:
        p = Path(path_str)
        if p.is_absolute() and p.exists():
            return p
        if (PROJECT_ROOT / p).exists():
            return PROJECT_ROOT / p
        if (self.metadata_csv.parent / p).exists():
            return self.metadata_csv.parent / p
        return PROJECT_ROOT / p

    def load_scene(self, row: pd.Series) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        img_path = self.resolve_path(str(row["image_path"]))
        mask_path = self.resolve_path(str(row["mask_path"]))
        
        image = cv2.imread(str(img_path))
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        
        if image is None:
            raise FileNotFoundError(f"Image not found at {img_path}")
        if mask is None:
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            
        info = row.to_dict()
        return image, mask, info

    def extract_training_samples(
        self,
        split: str = "train",
        max_scenes: Optional[int] = None,
        samples_per_image: int = 4000,
        positive_ratio: float = 0.40,
        augment: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Samples pixel feature vectors for training / validation across multiple scenes.
        """
        if split == "train":
            target_df = self.train_df
        elif split == "val":
            target_df = self.val_df
        else:
            target_df = self.test_df
            
        if max_scenes and len(target_df) > max_scenes:
            target_df = target_df.sample(n=max_scenes, random_state=self.random_seed).reset_index(drop=True)
            
        X_all: List[np.ndarray] = []
        y_all: List[np.ndarray] = []
        
        for _, row in target_df.iterrows():
            try:
                img, mask, _ = self.load_scene(row)
            except Exception:
                continue
            
            # Augmentations for train split
            if augment and split == "train":
                if np.random.rand() > 0.5:
                    img = cv2.flip(img, 1)
                    mask = cv2.flip(mask, 1)
                if np.random.rand() > 0.5:
                    img = cv2.flip(img, 0)
                    mask = cv2.flip(mask, 0)
                    
            features = extract_sar_features(img) # H x W x C
            h, w, c = features.shape
            flat_features = features.reshape(-1, c)
            flat_mask = (mask.reshape(-1) > 127).astype(np.int32)
            
            pos_indices = np.where(flat_mask == 1)[0]
            neg_indices = np.where(flat_mask == 0)[0]
            
            if len(pos_indices) > 0:
                n_pos = int(samples_per_image * positive_ratio)
                n_pos = min(n_pos, len(pos_indices))
                chosen_pos = np.random.choice(pos_indices, size=n_pos, replace=False)
            else:
                n_pos = 0
                chosen_pos = np.array([], dtype=int)
                
            n_neg = samples_per_image - n_pos
            n_neg = min(n_neg, len(neg_indices))
            if len(neg_indices) > 0 and n_neg > 0:
                chosen_neg = np.random.choice(neg_indices, size=n_neg, replace=False)
            else:
                chosen_neg = np.array([], dtype=int)
                
            chosen = np.concatenate([chosen_pos, chosen_neg])
            if len(chosen) == 0:
                continue
            np.random.shuffle(chosen)
            
            X_all.append(flat_features[chosen])
            y_all.append(flat_mask[chosen])
            
        if not X_all:
            raise ValueError(f"No valid training samples extracted from {split} split")
            
        X = np.vstack(X_all)
        y = np.concatenate(y_all)
        return X, y

