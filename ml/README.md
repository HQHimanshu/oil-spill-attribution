# OceanGuard AI — Machine Learning Pipeline & Dataset Documentation

The **OceanGuard AI Machine Learning Pipeline** implements multi-scale Synthetic Aperture Radar (SAR) texture characterization, oil spill segmentation, and coordinate regression for probable spill origin localization.

---

## 📊 Dataset Ingestion: 8,070 PALSAR SAR Scenes

The pipeline is trained and validated on **8,070 authentic PALSAR SAR satellite images and corresponding ground-truth masks**:

| Split | Scenes | Description |
| :--- | :--- | :--- |
| **Train Set** (`images/train/`, `masks/train/`) | **6,455** | Multi-scene training images with pixel annotations |
| **Validation Set** (`images/val/`, `masks/val/`) | **1,615** | Held-out validation and evaluation scenes |
| **Total** | **8,070** | 7,710 Spill Cases / 360 Clean Negative Scenes |

Dataset indexing is managed by [`ml/data/build_palsar_metadata.py`](file:///c:/Users/Ashutosh%20Pandey/Downloads/oil-spill-attribution/ml/data/build_palsar_metadata.py), producing the metadata catalog in [`ml/data/palsar_metadata.csv`](file:///c:/Users/Ashutosh%20Pandey/Downloads/oil-spill-attribution/ml/data/palsar_metadata.csv).

---

## 🧠 Feature Engineering (10 Multi-Scale Channels)

Oil slicks appear in SAR imagery as dark regions with suppressed radar backscatter due to the dampening of wind-generated capillary waves. The feature extraction engine in [`ml/characterization/model.py`](file:///c:/Users/Ashutosh%20Pandey/Downloads/oil-spill-attribution/ml/characterization/model.py) computes 10 complementary features for each pixel:

1. **`raw_intensity`**: Normalized raw pixel intensity $[0, 1]$.
2. **`log_backscatter`**: $\log_{10}(I + 10^{-4})$ backscatter coefficient in decibels.
3. **`lee_filtered`**: Adaptive Lee speckle filter with a $7\times 7$ window to remove multiplicative speckle noise while preserving sharp slick boundaries:
   $$\hat{I} = \bar{I} + W \cdot (I - \bar{I}), \quad W = \frac{\sigma^2}{\sigma^2 + \sigma_{\text{noise}}^2}$$
4. **`mean_3x3`**: Micro-texture local neighborhood mean.
5. **`mean_9x9`**: Meso-scale spatial mean.
6. **`mean_21x21_bg`**: Macro-scale background reference level for adaptive contrast calculation.
7. **`std_9x9_damping`**: Local variance / standard deviation quantifying capillary wave damping.
8. **`dark_contrast`**: Relative darkness ratio $\frac{I_{\text{local}} - I_{\text{bg}}}{I_{\text{bg}} + 10^{-3}}$.
9. **`sobel_gradient`**: Sobel edge magnitude $\sqrt{G_x^2 + G_y^2}$ detecting slick boundaries.
10. **`laplacian_blob`**: Laplacian of Gaussian (LoG) operator detecting localized dark discharge blobs.

---

## 🏗️ Model Architectures

### 1. SAR Oil Spill Segmenter (`SarOilSpillSegmenter`)
- **Classifier**: Random Forest Classifier (150 estimators, max depth 16, balanced class weights).
- **Training Samples**: 240,000 multi-scale feature vectors sampled across the dataset.
- **Post-Processing**: Morphological opening and closing kernels to remove spurious noise and fill internal pinholes.
- **Artifact**: Saved to [`ml/models/sar_spill_segmentation_model.joblib`](file:///c:/Users/Ashutosh%20Pandey/Downloads/oil-spill-attribution/ml/models/sar_spill_segmentation_model.joblib).

### 2. AIS Probable Spill Location Regressor
- **Architecture**: Multi-Output Random Forest Regressor mapping observed centroid coordinates, wind leeway, ocean drift vectors, and drift duration to probable release coordinates $(\lambda_{\text{orig}}, \phi_{\text{orig}})$.
- **Evaluation**: $R^2 = 0.999999$, $\text{MAE} = 0.0038^\circ$ (~400 meters spatial accuracy).
- **Artifact**: Saved to [`ml/models/spill_location_model.joblib`](file:///c:/Users/Ashutosh%20Pandey/Downloads/oil-spill-attribution/ml/models/spill_location_model.joblib).

---

## 📈 Evaluation Metrics & Benchmark Performance

Evaluated across the held-out validation scenes in [`ml/evaluation/metrics.json`](file:///c:/Users/Ashutosh%20Pandey/Downloads/oil-spill-attribution/ml/evaluation/metrics.json):

```json
{
  "model_version": "v2.1",
  "dataset_version": "PALSAR SAR Benchmark (8,070 scenes)",
  "test_mean_iou": 0.5054,
  "test_mean_dice": 0.6263,
  "test_mean_precision": 0.6393,
  "test_mean_recall": 0.7481,
  "test_pixel_accuracy": 0.7995
}
```

Qualitative 4-panel visual composites (`SAR Scene | Ground Truth Mask | ML Prediction | Segmented Overlay`) are generated and stored in [`ml/evaluation/qualitative_results/`](file:///c:/Users/Ashutosh%20Pandey/Downloads/oil-spill-attribution/ml/evaluation/qualitative_results/).

---

## 🚀 Running the Full ML Training Pipeline

To execute dataset indexing, feature extraction, model fitting, and evaluation in a single command:
```bash
python ml/train_all.py
```
