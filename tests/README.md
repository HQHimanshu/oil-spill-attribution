# OceanGuard AI — Automated Test Suite & Verification

The **OceanGuard AI Test Suite** validates all computational modules, backend API contracts, physics simulations, ML models, and RAG copilot grounded responses.

---

## 🧪 Test Suite Structure

```
tests/
├── test_backend_contract.py    # Validates REST API schemas and camelCase/snake_case contract compliance
├── test_live_api.py            # End-to-end integration test against FastAPI test client
├── test_phase31_pipeline.py    # Comprehensive verification of the 10-step analytical pipeline
└── README.md                   # Test documentation

backend/tests/
└── test_investigation_api.py   # Backend investigation CRUD, 404 handling, geocoding, live environmental tests
```

---

## 🚀 Running the Tests

To run the full test suite using `pytest`:

```bash
python -m pytest
```

To run with verbose output and per-test timing:

```bash
python -m pytest -v --durations=10
```

To run a specific test module:

```bash
python -m pytest tests/test_phase31_pipeline.py -v
```

---

## 📋 Test Coverage Breakdown

### 1. `test_backend_contract.py`
- Verifies that all backend responses strictly match the standardized JSON contracts required by frontend consumers.
- Ensures all numeric values (`latitude`, `longitude`, `confidence`, `area`, `evidenceScore`) are within valid physical ranges.

### 2. `test_investigation_api.py`
- Tests `POST /api/investigations` creation and parameter parsing.
- Tests `GET /api/investigations/{id}` for valid and invalid IDs (ensuring proper HTTP 404 responses).
- Tests `GET /api/location/search?query=...` geocoding search.
- Tests `GET /api/environmental/live` real-time weather and current fetching.

### 3. `test_phase31_pipeline.py`
- **SAR Feature Extraction**: Verifies 10-channel multi-scale feature array generation on sample SAR scenes.
- **Inference Engine**: Tests `SarInferenceEngine.run_inference()` output dictionary structure, confidence computation, and polygon boundary extraction.
- **Metocean Backtracking**: Tests backward Lagrangian drift numerical integration, Coriolis deflection, and uncertainty radius calculation.
- **AIS Correlation**: Tests great-circle Haversine filtering and AIS transmission gap (>15 min) detection.
- **Probabilistic Attribution**: Tests weighted candidate scoring and ranking logic.
- **RAG Copilot**: Tests grounded retrieval, citation linking, and grounding section validation (`Observed Data`, `Model Output`, `Retrieved Knowledge`, `Probabilistic Inference`).
- **Empirical Model Validation**: Verifies that model metrics meet benchmark accuracy standards (Mean IoU $\ge 50\%$, Dice Score $\ge 60\%$).
- **Data Provenance**: Verifies audit trail integrity across all pipeline stages.
