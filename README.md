# SIH Oil Spill Attribution

AI-powered system for detecting marine oil spills, estimating their
probable origin, and ranking potential source vessels using satellite
imagery, environmental conditions, and historical AIS data.

## Problem

Marine oil spills can be detected from satellite imagery, but identifying
the vessel most likely responsible requires correlating the spill's
location and movement with environmental conditions and historical vessel
trajectories.

## Proposed Pipeline

Satellite Imagery
→ Spill Characterization
→ Environmental Data
→ Drift / Backtracking
→ Probable Origin Region
→ Historical AIS
→ Candidate Vessel Filtering
→ Evidence-Based Ranking
→ Investigator Dashboard

## Project Structure

- `ml/` — spill characterization and backtracking models
- `backend/` — API and application logic
- `frontend/` — investigator dashboard
- `data/` — local/sample datasets
- `docs/` — architecture and project documentation
- `tests/` — testing

## Status

🚧 Prototype under development for Smart India Hackathon.