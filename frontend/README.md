# OceanGuard AI — Frontend Investigator Dashboard

The **OceanGuard AI Frontend** is a responsive, high-performance maritime intelligence dashboard designed for environmental authorities, coast guards, and port inspectors. Built using modern **Vanilla CSS, ES6+ JavaScript, and Leaflet GIS**, it provides real-time spatial analysis, interactive satellite scene exploration, and an AI Investigation Copilot without heavy framework bloat.

---

## 🎨 Design System & Aesthetics

- **Typography**: Modern pairing using [Outfit](https://fonts.google.com/specimen/Outfit) for headings/brand identity and [Inter](https://fonts.google.com/specimen/Inter) for dense telemetry data.
- **Color Palette**:
  - Deep Oceanic Dark Background: `#030c12` / `#071722`
  - Neon Cyan Accent & Primary Glow: `#00e5ff` (`rgba(0, 229, 255, 0.28)`)
  - Success Indicator: `#10b981` (Verified Real Data / Safe Vessels)
  - Warning Amber: `#f59e0b` (Metocean Drift & Medium Risk)
  - Alert Danger: `#ef4444` (High Risk Vessels & Probable Origin Zone)
- **Glassmorphism**: Translucent backdrop-filter panels (`blur(16px)`) with subtle neon borders.
- **Responsiveness**: CSS Grid layout adapting seamlessly from 4K ultrawide monitors down to mobile viewports.

---

## 🗺️ Interactive GIS & Leaflet Implementation

The center map utilizes **Leaflet 1.9.4** on top of the keyless **Esri World Dark Gray Canvas** base layer:
1. **Observed Spill Centroid & Slick Polygon**: Cyan circle/polygon with computed area in $\text{km}^2$ and ML backscatter confidence.
2. **Lagrangian Backtracked Drift Vector**: Amber dashed line showing the backwards trajectory over the calculated drift duration.
3. **Probable Discharge Origin Zone**: Red translucent ellipse with the spatial uncertainty radius ($\pm 3.8\text{ km}$).
4. **AIS Candidate Vessel Trajectories**: Color-coded trajectories (Red: High Risk, Amber: Medium, Green: Low) with clickable markers revealing vessel metadata (MMSI, SOG, COG, heading, distance, and AIS gap warnings).
5. **Click-to-Pick Inspection**: Clicking anywhere on the ocean map immediately updates coordinates, fetches real-time Open-Meteo weather and currents, and runs the entire attribution pipeline for that point.

---

## 🔍 Key Dashboard Features

### 1. Global Location & Port Search
- Autocomplete search bar querying curated maritime ports (Mumbai, Singapore, Galveston, Suez, Rotterdam, Dubai, Dover, Malacca, Paradip, Shanghai, Tokyo, etc.) and worldwide Open-Meteo geocoding.
- Selecting any suggestion navigates the map and initiates attribution analysis.

### 2. PALSAR Dataset Scene Explorer
- Allows selecting real satellite scenes from the **8,070 PALSAR dataset** (e.g. `palsar_0`, `palsar_10`, `palsar_100`, `palsar_1009`) or Sentinel-1 preset incidents.
- Custom SAR file drag-and-drop zone sending images to `POST /detect-spill` for instant ML segmentation.

### 3. Live Metocean Feed
- Displays real-time wind speed (knots), wind direction, surface ocean current velocity ($m/s$), direction, and significant wave height ($m$).
- Live Real-Time Feed badge toggles dynamically based on the timestamp.

### 4. Ranked Suspect Vessels & Evidence Breakdown
- Real-time ranked list of candidate vessels.
- Selecting a vessel visualizes its evidence breakdown bars:
  - Spatial Proximity ($35\%$)
  - Temporal Release Overlap ($30\%$)
  - Trajectory & Heading Alignment ($20\%$)
  - AIS Track Completeness ($15\%$)

### 5. Grounded AI Investigation Copilot (RAG Drawer)
- Sliding right-hand drawer with quick-prompt chips:
  - *"Why ranked #1?"*
  - *"Candidate Evidence"*
  - *"Metocean Conditions"*
  - *"Probable Origin"*
  - *"Uncertainties & Gaps"*
- Answers formatted with 4 distinct color-coded grounding tags and authoritative citations (MARPOL Annex I, IMO Guidelines).

### 6. Model Retraining & Evaluation Modals
- **Model Metrics Modal**: Displays empirical evaluation metrics on benchmark scenes (Mean IoU: 50.54%, Dice: 62.63%, Recall: 74.81%, Precision: 63.93%).
- **Retraining Modal**: Triggers ML training across the 8,070 PALSAR dataset via `POST /api/model/train` with live progress spinner.

---

## 📁 File Structure

```
frontend/
├── index.html                    # Single-page HTML application and stylesheet
├── services/
│   └── investigationApi.js       # Asynchronous API service client for FastAPI backend
└── README.md                     # Frontend documentation
```

---

## 🚀 Serving Frontend Locally

To run the frontend locally:
```bash
python -m http.server 3000 --directory frontend
```
Open **[http://127.0.0.1:3000](http://127.0.0.1:3000)** in your browser.
