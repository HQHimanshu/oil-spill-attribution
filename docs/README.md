# OceanGuard AI — Technical Documentation & Regulatory Manual

This directory contains architectural specifications, regulatory compliance manuals, mathematical formulations, and operational runbooks for the **OceanGuard AI Autonomous Maritime Attribution System**.

---

## 📚 Documentation Index

1. **[System Architecture & Data Flow](file:///c:/Users/Ashutosh%20Pandey/Downloads/oil-spill-attribution/docs/architecture.md)**: End-to-end data pipelines from SAR satellite ingestion to RAG copilot.
2. **[Regulatory Compliance Manual](file:///c:/Users/Ashutosh%20Pandey/Downloads/oil-spill-attribution/docs/regulatory_frameworks.md)**: Legal frameworks governing maritime oil pollution, MARPOL Annex I discharge thresholds, and evidentiary standards.
3. **[API Specifications & Schemas](file:///c:/Users/Ashutosh%20Pandey/Downloads/oil-spill-attribution/backend/README.md)**: Complete OpenAPI 3.1 REST contracts.
4. **[Machine Learning & Dataset Guide](file:///c:/Users/Ashutosh%20Pandey/Downloads/oil-spill-attribution/ml/README.md)**: Details on the 8,070 PALSAR dataset, 10 feature channels, and validation metrics.

---

## ⚖️ Maritime Legal & Regulatory Frameworks

### 1. MARPOL 73/78 Annex I — Prevention of Pollution by Oil
- **Regulation 15 (Control of Operational Discharge of Oil from Cargo Spaces of Oil Tankers)**:
  - Discharge of oily mixtures is strictly prohibited unless:
    1. The tanker is not within a **Special Area** (e.g. Mediterranean, Baltic, Red Sea, Arabian Gulf).
    2. The tanker is more than **50 nautical miles** from the nearest land.
    3. The tanker is proceeding en route.
    4. The instantaneous rate of discharge of oil content does not exceed **30 liters per nautical mile**.
    5. The total quantity of oil discharged into the sea does not exceed $\frac{1}{30,000}$ of the total quantity of cargo carried.
    6. The tanker has in operation an approved **Oil Discharge Monitoring and Control System (ODMCS)** and a slop tank arrangement.
- **Regulation 34 (Machinery Space Bilges for All Ships)**:
  - Oil content of the effluent without dilution must not exceed **15 parts per million (15 ppm)**.
  - Oily-water separator (OWS) and 15 ppm bilge alarm must be operational.

### 2. IMO Resolution A.1106(29) — Operational Use of Shipborne AIS
- Mandates that Class-A Automatic Identification Systems (AIS) must remain operational at all times while underway or at anchor.
- Deactivating AIS (*"going dark"*) during ocean transit without master's safety justification constitutes an intentional violation and creates a rebuttable presumption of non-compliance during oil pollution investigations.

### 3. UNCLOS (United Nations Convention on the Law of the Sea)
- **Article 211**: Gives coastal states jurisdictional authority to establish laws and regulations to prevent, reduce, and control pollution of the marine environment from vessels in their Exclusive Economic Zone (EEZ).
- **Article 218 & 220**: Provides port states authority to investigate and institute proceedings against vessels for discharge violations committed outside port waters.

### 4. U.S. Clean Water Act (33 U.S.C. § 1321 - Section 311)
- Prohibits the discharge of oil or hazardous substances into navigable waters in quantities that cause a film or sheen upon or discoloration of the surface of the water or cause a sludge or emulsion.

---

## 🛡️ Chain of Custody & Evidentiary Provenance

For maritime investigations to withstand judicial scrutiny in admiralty courts or port state control hearings, OceanGuard AI enforces:
- **Immutable Timestamping**: UTC acquisition timestamps recorded for all SAR scenes, metocean records, and AIS positions.
- **Reproducible Determinism**: Physics calculations and ML inference pipelines utilize fixed seeds and versioned checkpoints (`v2.1`).
- **Data Source Verification**: Telemetry origin explicitly tagged (`Copernicus Sentinel-1`, `NOAA AIS`, `ERA5 High-Resolution Reanalysis`).
- **Grounded AI Assistant**: The RAG Copilot cannot hallucinate facts; responses are strictly grounded in verified observed telemetry and official maritime regulations.
