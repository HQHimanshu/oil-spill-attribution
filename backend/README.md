# OceanGuard AI — Backend Service Architecture

The **OceanGuard AI Backend** is a high-performance asynchronous REST API powered by **FastAPI**. It coordinates the analytical pipelines required for maritime oil spill detection, metocean drift physics, historical AIS vessel correlation, multi-factor probabilistic attribution, and grounded RAG investigation assistant.

---

## 🏛️ Architecture Overview

```
                      FastAPI Application (backend/app/main.py)
                                      │
  ┌───────────────────┬───────────────┴───────────────┬───────────────────┐
  ▼                   ▼                               ▼                   ▼
SAR Detection    Environmental Service           AIS Service        RAG Copilot
(Inference)     (Open-Meteo & ERA5/CMEMS)    (NOAA & Global Marine) (Grounded Knowledge)
  │                   │                               │                   │
  ▼                   ▼                               ▼                   ▼
Oil Slick Mask    10m Wind & Surface Currents   Candidate Tracks     MARPOL Citations
Area & Centroid       Physics Vectors            & Gap Detection      & Telemetry Grounding
  └───────────────────┼───────────────────────────────┘
                      ▼
             Lagrangian Backtracking
             (Probable Origin & Time)
                      │
                      ▼
         Multi-Factor Vessel Ranking
         (35% Prox, 30% Time, 20% Traj, 15% Type)
```

---

## 📦 Service Components

### 1. `app/main.py`
The central FastAPI router providing HTTP REST endpoints, CORS middleware, data normalization, and request routing:
- Manages active investigations in memory.
- Provides fallback and dynamic scene resolution for Sentinel-1 and PALSAR SAR images.
- Implements global location search & Open-Meteo geocoding.

### 2. `app/environmental_service.py`
Retrieves and formats real-world atmospheric and oceanographic metocean vectors:
- **Live Real-Time Forecast**: Queries `api.open-meteo.com/v1/forecast` and `marine-api.open-meteo.com/v1/marine` for current wind speed (knots), wind direction (deg), wind gusts, surface ocean current velocity ($m/s$), direction ($deg$), and significant wave height ($m$).
- **Historical ERA5/CMEMS Reanalysis**: Queries `archive-api.open-meteo.com/v1/archive` for historical incident dates.
- **Disk Caching**: Local hashed disk cache in `backend/database/environmental_cache/` prevents duplicate network queries and ensures offline repeatability.

### 3. `app/ais_service.py`
Ingests, filters, and analyzes vessel transponder tracking records:
- **NOAA Marine AIS Dataset**: Ingests high-density historical CSV records from `backend/database/ais.csv`.
- **Global Marine Telemetry Simulation**: Provides realistic candidate merchant vessel tracks along standard shipping corridors for international ocean coordinates outside North American coastal waters.
- **Spatial-Temporal Filtering**: Haversine great-circle distance filtering within search radius (e.g. 60 km).
- **AIS Gap Detection**: Identifies voyage segments with >15 minutes between consecutive transponder pings, flagging suspicious transmitter deactivation.

### 4. `app/backtracking_service.py`
Executes backward Euler-Lagrangian physics integration to determine where and when the oil slick was discharged:
- **Wind Leeway ($3\%$)**: Oil slick moves at $3\%$ of the $10\text{m}$ wind speed.
- **Coriolis Deflection**: Deflects drift vector $10^\circ$ to the right in the Northern Hemisphere (left in Southern Hemisphere).
- **Surface Current ($100\%$)**: Advects directly with the top $0.5\text{m}$ ocean current layer.
- Computes the uncertainty dispersion ellipse $(\pm 3.8\text{ km})$ and discharge time window.

### 5. `app/vessel_ranking.py`
Applies weighted probabilistic attribution scoring to candidate vessels:
$$\text{Score} = 0.35 \cdot S_{\text{prox}} + 0.30 \cdot S_{\text{time}} + 0.20 \cdot S_{\text{traj}} + 0.15 \cdot S_{\text{type}} - P_{\text{gap}}$$
- Classifies candidate vessels into `HIGH RISK` ($\ge 75$), `MEDIUM RISK` ($45-74$), or `LOW RISK` ($< 45$).

### 6. `app/rag_service.py`
Grounded Retrieval-Augmented Generation Copilot:
- Ingests structured maritime knowledge:
  - MARPOL 73/78 Annex I (Regulations 15 & 34)
  - IMO Resolution A.1106(29) AIS Guidelines
  - ESA Copernicus SAR Oil Spill Observation Guide
- Generates structured answers strictly citing telemetry with 4 grounding sections:
  1. `Observed Data`
  2. `Model Output`
  3. `Retrieved Knowledge`
  4. `Probabilistic Inference`

### 7. `app/provenance_service.py`
Maintains immutable audit records of all data sources, model versions, GSD resolutions, and processing timestamps for maritime regulatory compliance.

---

## 📡 REST API Reference

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/health` | Server health check and uptime status |
| `POST` | `/detect-spill` | Multipart upload for SAR image segmentation |
| `POST` | `/api/investigations` | Creates a new investigation from lat/lon/time/scene |
| `GET` | `/api/investigations/{id}` | Retrieves investigation details and candidate rankings |
| `POST` | `/api/investigations/{id}/analyze` | Triggers re-analysis of an existing investigation |
| `POST` | `/api/investigations/{id}/ask` | Queries the grounded RAG Investigation Copilot |
| `GET` | `/api/location/search?query=...` | Worldwide port and coastal location geocoding search |
| `GET` | `/api/environmental/live` | Queries real-time wind and ocean current parameters |
| `POST` | `/api/model/train` | Triggers retraining of the SAR segmenter and location regressor |
| `GET` | `/api/model/metrics` | Returns empirical validation IoU, Dice, Precision, Recall |
| `GET` | `/api/palsar/scenes` | Returns sample scenes from the 8,070 PALSAR dataset |
| `GET` | `/api/provenance` | Returns full data provenance audit trail |

---

## 💻 Running & Testing Backend

### Start Backend Development Server:
```bash
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
```

### Run Backend Unit Tests:
```bash
python -m pytest backend/tests/ -v
```
