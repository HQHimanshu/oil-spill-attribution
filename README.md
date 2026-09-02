# OceanGuard AI — Real-Data Maritime Oil Spill Attribution & RAG System

[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://python.org)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4+-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Leaflet](https://img.shields.io/badge/Leaflet-1.9.4-199900?logo=leaflet&logoColor=white)](https://leafletjs.com)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

> **Autonomous End-to-End Maritime Intelligence Platform (SIH26143)**: Detects marine oil spills from authentic SAR satellite imagery, models backward Lagrangian metocean drift, correlates vessel trajectories with AIS transmission gap analysis, performs multi-factor probabilistic attribution, and provides a grounded RAG Investigation Copilot with legal & regulatory citations.

---

## 🌟 Key Capabilities

1. **Authentic SAR Satellite Oil Spill Segmentation (8,070 PALSAR Scenes)**
   - Random Forest segmentation model trained on **8,070 authentic PALSAR SAR images and masks** (6,455 Train / 1,615 Val scenes).
   - 10 multi-scale feature channels: Log-Backscatter, Lee Speckle Filtering, Multi-Scale Spatial Means (3x3, 9x9, 21x21), Local Standard Deviation (Wave Damping), Sobel Edge Gradient, Dark Contrast, and Laplacian of Gaussian.
   - Evaluated benchmark performance: **50.54% Mean IoU**, **62.63% Dice Score**, **74.81% Recall**.

2. **Real-Time Live Metocean Feed & Historical Reanalysis**
   - Seamlessly queries **Open-Meteo Live Forecast APIs** (10m wind speed, direction, gusts, 0.5m surface ocean current vector, wave height) for current timestamps.
   - Queries **ECMWF ERA5 & Copernicus Marine (CMEMS)** high-resolution reanalysis for historical investigation dates.

3. **Physics-Based Backward Lagrangian Drift Simulation**
   - Reconstructs probable discharge origin zone $(\lambda_{orig}, \phi_{orig})$ and release time window using Euler-Lagrangian integration:
     $$\vec{v}_{drift} = \vec{v}_{current} + 0.03 \cdot \mathbf{R}(\theta_{Coriolis}) \vec{v}_{wind}$$
   - Calculates uncertainty dispersion envelope $(\pm 3.8\text{ km})$ based on wind gust variance and metocean temporal resolution.

4. **Historical AIS Correlation & Gap Analysis**
   - Matches NOAA & Global Marine AIS transponder data (IMO SOLAS Class-A) against the reconstructed origin zone.
   - Automatically detects suspicious **AIS transmission gaps (>15 min)** indicating intentional transponder deactivation.

5. **Multi-Factor Probabilistic Attribution Scoring**
   - Weighted multi-factor attribution model:
     - Spatial Proximity: **35%**
     - Release Time Window Overlap: **30%**
     - Trajectory & Heading Alignment: **20%**
     - Vessel Risk Profile (Crude/Chemical Tanker vs Cargo): **15%**
     - AIS Transmission Gap Penalty: **-15% confidence reduction**

6. **Grounded RAG Investigation Copilot**
   - Interactive investigative assistant strictly grounded in observed telemetry, ML outputs, and ingested maritime knowledge bases (**MARPOL 73/78 Annex I**, **IMO Resolution A.1106(29)**, **ESA Copernicus SAR Technical Guidelines**).
   - Returns structured evidence sections: `Observed Data`, `Model Output`, `Retrieved Knowledge`, and `Probabilistic Inference` with source citations.

7. **Interactive Global GIS Investigator Dashboard**
   - Worldwide location geocoding search (e.g. Mumbai Port, Singapore Strait, Suez Canal, Galveston, Rotterdam, Dubai, English Channel, Malacca Strait).
   - Interactive Leaflet map with **click-to-pick coordinate inspection**, candidate vessel track overlays, drift vectors, and live retraining modals.

---

## 🏗️ Architecture & Pipeline Overview

```
                          ┌───────────────────────────┐
                          │   SAR Satellite Imagery   │
                          │  (Sentinel-1 / PALSAR)    │
                          └─────────────┬─────────────┘
                                        │
                                        ▼
                          ┌───────────────────────────┐
                          │  SAR Feature Extraction   │
                          │ (Lee, Damping, Multi-Mean)│
                          └─────────────┬─────────────┘
                                        │
                                        ▼
                          ┌───────────────────────────┐
                          │  Random Forest Segmenter  │
                          │ (Area km², Centroid, Mask)│
                          └─────────────┬─────────────┘
                                        │
             ┌──────────────────────────┴──────────────────────────┐
             ▼                                                     ▼
┌───────────────────────────┐                         ┌───────────────────────────┐
│ ECMWF ERA5 / CMEMS Marine │                         │ NOAA / Global Marine AIS  │
│ Live Wind & Ocean Currents│                         │ Track & Gap Analysis      │
└────────────┬──────────────┘                         └─────────────┬─────────────┘
             │                                                      │
             ▼                                                      ▼
┌───────────────────────────┐                         ┌───────────────────────────┐
│   Lagrangian Backtrack    │                         │    Multi-Factor Vessel    │
│ Origin Zone & Time Window │ ──────────────────────> │    Attribution Ranking    │
└───────────────────────────┘                         └─────────────┬─────────────┘
                                                                    │
                                        ┌───────────────────────────┴──────────────────────────┐
                                        ▼                                                      ▼
                          ┌───────────────────────────┐                          ┌───────────────────────────┐
                          │ Interactive GIS Dashboard │                          │ Grounded RAG Copilot      │
                          │ (Leaflet, Global Search)  │                          │ (Citations, MARPOL Rules) │
                          └───────────────────────────┘                          └───────────────────────────┘
```

---

## 📂 Repository Structure

```
oil-spill-attribution/
├── backend/                  # FastAPI backend server & intelligence services
│   ├── app/
│   │   ├── main.py                   # FastAPI REST router & application entrypoint
│   │   ├── environmental_service.py  # Live Open-Meteo & ERA5/CMEMS metocean integration
│   │   ├── ais_service.py            # NOAA & Global AIS tracking and gap detector
│   │   ├── backtracking_service.py   # Euler-Lagrangian backward drift physics model
│   │   ├── vessel_ranking.py         # Multi-factor probabilistic attribution scorer
│   │   ├── rag_service.py            # Grounded RAG Investigation Copilot with citations
│   │   └── provenance_service.py     # End-to-end data audit trail & provenance
│   ├── database/                     # Caches, knowledge items, sample scenes, and AIS CSVs
│   ├── tests/                        # Backend unit & endpoint contract tests
│   └── README.md                     # Backend architecture documentation
├── frontend/                 # High-performance Investigator Dashboard
│   ├── index.html                    # Single-page GIS dashboard (Vanilla CSS & Leaflet)
│   ├── services/
│   │   └── investigationApi.js       # Asynchronous backend API client
│   └── README.md                     # Frontend UI & design system documentation
├── ml/                       # Machine learning training & evaluation pipeline
│   ├── data/                         # Metadata indexer (8,070 PALSAR scenes)
│   ├── training/                     # Dataset loader, feature sampler, training scripts
│   ├── characterization/             # 10-channel SAR texture feature engine & inference
│   ├── backtracking/                 # Drift physics formulations & location regressor
│   ├── evaluation/                   # Evaluation metrics JSON & 4-panel visual plots
│   ├── models/                       # Serialized Joblib model checkpoints & metadata
│   ├── train_all.py                  # Unified pipeline script to train & evaluate all models
│   └── README.md                     # ML model architecture & dataset documentation
├── images/                   # Authentic PALSAR SAR satellite imagery (train: 6,455, val: 1,615)
├── masks/                    # Ground truth segmentation masks (train: 6,455, val: 1,615)
├── docs/                     # Regulatory guidelines, system manuals, API specs
├── tests/                    # Integration, contract, and live API test suite
├── requirements.txt          # Python dependencies
└── README.md                 # Root project documentation
```

---

## 🚀 Quick Start

### 1. Prerequisites
- Python 3.10+ (Tested on Python 3.11, 3.12, 3.14)
- Web browser (Chrome, Edge, Firefox, Safari)

### 2. Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/HQHimanshu/oil-spill-attribution.git
cd oil-spill-attribution
python -m pip install -r requirements.txt
```

### 3. Run Machine Learning Pipeline (Optional — Pretrained Models Included)
To retrain the SAR segmentation model on the 8,070 PALSAR scenes and train the AIS location regressor:
```bash
python ml/train_all.py
```

### 4. Start Backend Server
```bash
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```
Backend API interactive Swagger docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### 5. Launch Frontend Dashboard
In a separate terminal, serve the frontend static files:
```bash
python -m http.server 3000 --directory frontend
```
Open your browser and navigate to: **[http://127.0.0.1:3000](http://127.0.0.1:3000)**

---

## 🧪 Testing

Run the comprehensive 18-test automated suite covering API contracts, Lagrangian physics, RAG grounding, and ML pipeline verification:
```bash
python -m pytest
```

---

## 📊 Evaluation & Benchmark Results

Evaluated across the held-out validation and test splits of the PALSAR SAR Satellite Benchmark dataset:

| Evaluation Metric | Score | Description |
| :--- | :--- | :--- |
| **Mean IoU (Jaccard)** | **50.54%** | Overlap between predicted slick boundary and ground truth |
| **Dice Coefficient (F1)** | **62.63%** | Harmonic mean of pixel precision and recall |
| **Precision** | **63.93%** | True positive oil pixels / all predicted oil pixels |
| **Recall** | **74.81%** | True positive oil pixels / actual ground truth oil pixels |
| **Pixel Accuracy** | **79.95%** | Overall multi-class classification accuracy across SAR scenes |
| **Location MAE** | **0.0038°** | AIS location regressor error (~400m spatial accuracy) |

---

## 📜 Regulatory & Legal References
- **MARPOL 73/78 Annex I (Regulation 15 & 34)**: Mandatory control of operational discharge of oil into sea.
- **IMO Resolution A.1106(29)**: Revised guidelines for the onboard operational use of shipborne AIS.
- **UNCLOS Article 211**: Pollution from vessels & coastal state enforcement authority.
- **U.S. Clean Water Act (Section 311)**: Prohibition of oil discharges causing sheen.

---

## 👥 Contributors & Acknowledgements
- **Team SamadhanLabs** (Smart India Hackathon SIH26143)
- Satellite Data: Japan Aerospace Exploration Agency (JAXA) PALSAR & ESA Copernicus Sentinel-1.
- Metocean Data: European Centre for Medium-Range Weather Forecasts (ECMWF) & Copernicus Marine Environment Monitoring Service (CMEMS).
- AIS Data: National Oceanic and Atmospheric Administration (NOAA) Office for Coastal Management.