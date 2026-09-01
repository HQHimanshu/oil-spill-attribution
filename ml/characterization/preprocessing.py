"""
SAR Image Preprocessing and Multi-Scale Feature Extraction for Oil Spill Segmentation.
Implements C-band SAR backscatter calibration, speckle filtering, and texture analysis.
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy.ndimage import uniform_filter


def apply_lee_filter(img: np.ndarray, size: int = 5) -> np.ndarray:
    """
    Applies Lee Speckle Filter on SAR amplitude image.
    Preserves sharp edges while smoothing multiplicative speckle noise in homogeneous sea areas.
    """
    img_f = img.astype(np.float32)
    img_mean = uniform_filter(img_f, (size, size))
    img_sqr_mean = uniform_filter(img_f**2, (size, size))
    img_variance = np.maximum(0, img_sqr_mean - img_mean**2)
    
    overall_variance = np.var(img_f)
    if overall_variance == 0:
        return img_f
        
    weights = img_variance / (img_variance + overall_variance + 1e-6)
    filtered = img_mean + weights * (img_f - img_mean)
    return np.clip(filtered, 0, 255).astype(np.float32)


def extract_sar_features(image: np.ndarray) -> np.ndarray:
    """
    Extracts multi-scale radar backscatter, textural, and edge features for each pixel.
    
    Feature Channels:
    0: Normalized intensity (0 to 1)
    1: Log-scaled radar cross section
    2: Lee filtered intensity
    3: Local mean (3x3 window)
    4: Local mean (9x9 window)
    5: Local standard deviation (9x9 window) - detects wave damping
    6: Local mean (21x21 window)
    7: Sobel gradient magnitude - detects slick boundary
    8: Relative dark contrast (pixel intensity - local background)
    9: Laplacian of Gaussian (scale-space blob detector)
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
        
    gray_f = gray.astype(np.float32)
    
    # 0. Normalized intensity
    f0 = gray_f / 255.0
    
    # 1. Log-scaled backscatter
    f1 = np.log1p(gray_f) / np.log1p(255.0)
    
    # 2. Lee filtered intensity
    lee = apply_lee_filter(gray, size=5)
    f2 = lee / 255.0
    
    # 3 & 4. Multi-scale local means
    mean_3 = cv2.blur(gray_f, (3, 3)) / 255.0
    mean_9 = cv2.blur(gray_f, (9, 9)) / 255.0
    
    # 5. Local standard deviation (9x9)
    sqr_mean_9 = cv2.blur(gray_f**2, (9, 9))
    var_9 = np.maximum(0, sqr_mean_9 - (cv2.blur(gray_f, (9, 9))**2))
    std_9 = np.sqrt(var_9) / 255.0
    
    # 6. Broad background mean (21x21)
    mean_21 = cv2.blur(gray_f, (21, 21)) / 255.0
    
    # 7. Sobel edge magnitude
    sobel_x = cv2.Sobel(gray_f, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray_f, cv2.CV_32F, 0, 1, ksize=3)
    sobel_mag = np.sqrt(sobel_x**2 + sobel_y**2)
    sobel_mag = sobel_mag / (np.max(sobel_mag) + 1e-6)
    
    # 8. Dark contrast relative to 21x21 background
    dark_contrast = (mean_21 - f0)
    
    # 9. Laplacian of Gaussian (blob & ridge detection)
    blurred = cv2.GaussianBlur(gray_f, (5, 5), 1.0)
    laplacian = cv2.Laplacian(blurred, cv2.CV_32F)
    laplacian = np.clip(laplacian, -50, 50) / 50.0
    
    features = np.stack([
        f0, f1, f2, mean_3, mean_9, std_9, mean_21, sobel_mag, dark_contrast, laplacian
    ], axis=-1)
    
    return features
