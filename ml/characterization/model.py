"""
SAR Oil Spill Segmentation Model based on multi-scale radar backscatter
feature extraction, ensemble decision trees, and morphological contour refinement.
"""
from __future__ import annotations

import cv2
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from typing import Dict, Any, Tuple, List

try:
    from .preprocessing import extract_sar_features
except ImportError:
    from preprocessing import extract_sar_features


class SarOilSpillSegmenter:
    """
    Supervised Machine Learning Model for SAR Sentinel-1 Oil Spill Detection and Segmentation.
    """
    def __init__(self, n_estimators: int = 150, max_depth: int = 16, random_state: int = 42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            n_jobs=-1,
            random_state=self.random_state,
            class_weight="balanced",
            min_samples_leaf=2
        )
        self.is_fitted = False
        self.feature_names = [
            "raw_intensity",
            "log_backscatter",
            "lee_filtered",
            "mean_3x3",
            "mean_9x9",
            "std_9x9_damping",
            "mean_21x21_bg",
            "sobel_gradient",
            "dark_contrast",
            "laplacian_blob"
        ]

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fits the random forest classifier on sampled pixel feature vectors.
        """
        self.model.fit(X, y)
        self.is_fitted = True
        return self

    def predict_mask(self, image: np.ndarray, threshold: float = 0.50) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predicts pixel-level probability map and binary segmentation mask for an input SAR image.
        """
        if not self.is_fitted:
            raise ValueError("Model is not trained. Call fit() or load a trained checkpoint first.")
            
        h, w = image.shape[:2]
        features = extract_sar_features(image)
        flat_features = features.reshape(-1, features.shape[-1])
        
        # Predict probabilities of oil spill (class 1)
        prob_flat = self.model.predict_proba(flat_features)[:, 1]
        prob_map = prob_flat.reshape(h, w)
        
        # Raw binary mask
        binary_mask = (prob_map >= threshold).astype(np.uint8) * 255
        
        # Morphological post-processing
        # 1. Opening to remove isolated false positive pixels
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cleaned_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel_open)
        
        # 2. Closing to fill small holes within continuous oil slicks
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel_close)
        
        # 3. Filter tiny spurious regions (< 50 pixels)
        contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        final_mask = np.zeros_like(cleaned_mask)
        for cnt in contours:
            if cv2.contourArea(cnt) >= 50:
                cv2.drawContours(final_mask, [cnt], -1, 255, -1)
                
        return final_mask, prob_map
