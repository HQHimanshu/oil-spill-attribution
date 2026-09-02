# OceanGuard AI — SAR Spill Characterization & Inference Engine

The **SAR Spill Characterization Module** (`ml/characterization/`) processes raw or preprocessed Synthetic Aperture Radar (SAR) imagery, extracts multi-scale backscatter and texture features, segments oil spill slicks, and extracts geographic polygon boundaries and physical area metrics.

---

## 🛰️ SAR Oil Spill Physics

SAR sensors transmit microwave radar pulses and record the backscattered signal from the Earth's surface:
- **Clean Sea Surface**: Capillary waves (wavelengths of a few centimeters) interact resonantly with radar waves via Bragg scattering, producing strong backscatter (bright return).
- **Oil Slick Surface**: Viscous oil dampens capillary and short gravity waves, resulting in specular reflection away from the radar sensor (dark return / low backscatter).

---

## ⚙️ Feature Extraction Engine (`model.py`)

The feature extraction routine extracts 10 normalized channels for each pixel:

| Feature Name | Kernel / Parameters | Physical Interpretation |
| :--- | :--- | :--- |
| `raw_intensity` | $1\times 1$ | Normalized raw radar amplitude $[0, 1]$ |
| `log_backscatter` | $1\times 1$ | Backscatter coefficient in dB ($\log_{10}$) |
| `lee_filtered` | $7\times 7$ adaptive | Multiplicative speckle noise suppression |
| `mean_3x3` | $3\times 3$ box | Micro-texture averaging |
| `mean_9x9` | $9\times 9$ box | Meso-scale contextual intensity |
| `mean_21x21_bg` | $21\times 21$ box | Macro-scale ocean background reference |
| `std_9x9_damping` | $9\times 9$ std dev | Capillary wave damping quantification |
| `dark_contrast` | $(I_{\text{local}} - I_{\text{bg}}) / I_{\text{bg}}$ | Local-to-background contrast ratio |
| `sobel_gradient` | $3\times 3$ Sobel $G_x, G_y$ | Slick boundary gradient detection |
| `laplacian_blob` | $\sigma = 2.0$ LoG | Localized dark discharge blob detector |

---

## 🔬 Contour & Geospatial Boundary Extraction

Once the Random Forest model outputs a binary pixel mask $\hat{Y} \in \{0, 1\}^{H \times W}$, [`SarInferenceEngine`](file:///c:/Users/Ashutosh%20Pandey/Downloads/oil-spill-attribution/ml/characterization/inference.py) performs:
1. **Morphological Filtering**: $3\times 3$ opening and closing to eliminate single-pixel false alarms.
2. **OpenCV Contour Tracing**: Identifies external polygon contours of the largest detected slick blobs.
3. **Geospatial GeoJSON Mapping**: Projects image pixel coordinates $(x, y)$ to geographic latitude and longitude based on the scene metadata:
   $$\text{Lat}(y) = \text{CenterLat} + \left(\frac{H/2 - y}{H}\right) \cdot \Delta\text{Lat}$$
   $$\text{Lon}(x) = \text{CenterLon} + \left(\frac{x - W/2}{W}\right) \cdot \Delta\text{Lon}$$
4. **Physical Area Calculation**: Calculates slick area in $\text{km}^2$ from ground sampling distance (GSD):
   $$\text{Area}_{\text{km}^2} = N_{\text{oil pixels}} \times \left(\frac{\text{GSD}_{\text{meters}}}{1000}\right)^2$$

---

## 💻 Python Usage Example

```python
from ml.characterization.inference import get_sar_inference_engine

engine = get_sar_inference_engine()

# Run inference on an image file path or bytes
result = engine.run_inference("images/train/palsar_0.png", center_lat=28.582, center_lon=-94.925)

print(f"Spill Detected: {result['detected']}")
print(f"Confidence: {result['confidence'] * 100:.1f}%")
print(f"Area: {result['area_km2']:.2f} km²")
print(f"Centroid: {result['centroid']}")
print(f"Polygon Boundary Vertices: {len(result['boundary'])}")
```
