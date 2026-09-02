# Sentinel-1 SAR Oil Spill Detection Technical Reference

**Source**: European Space Agency (ESA) Copernicus Sentinel-1 Technical Guide  
**Title**: Principles of Synthetic Aperture Radar (SAR) Marine Oil Spill Detection and Characterization  
**Date**: 2024-01-15  
**Document Type**: Technical Reference Manual  
**URL**: https://sentinels.copernicus.eu/web/sentinel/user-guides/sentinel-1-sar  

## 1. Physical Detection Mechanism
C-band Synthetic Aperture Radar (5.405 GHz, wavelength ~5.6 cm) detects marine oil slicks via the suppression of short gravity-capillary ocean surface waves (Bragg scattering waves with wavelength $\lambda_{Bragg} = \frac{\lambda_{radar}}{2 \sin \theta}$). 

Mineral oils and petroleum hydrocarbons form a thin surface viscoelastic film that dampens resonant capillary waves, causing specular reflection away from the radar antenna. This produces a characteristic low-backscatter dark signature (damping ratio typically 8 dB to 15 dB below surrounding sea clutter).

## 2. Environmental Operating Limits (Wind Speed Thresholds)
* **Under 3 m/s (~6 knots)**: Insufficient sea surface roughness exists to generate background Bragg backscatter; the entire sea surface appears dark, leading to high false-positive look-alikes.
* **Between 3 m/s and 12 m/s (6 to 24 knots)**: Optimal operating window for SAR oil spill detection. Sea clutter provides high contrast against dampened slicks.
* **Above 12-14 m/s (>25 knots)**: Intense wind turbulence and wave breaking rapidly disperse oil films into the water column through emulsification and down-welling, making surface films undetectable.

## 3. Look-Alikes and False Positive Differentiation
Natural look-alikes must be distinguished from mineral oil spills:
1. **Biogenic Slicks (Algal blooms / Plankton)**: Produce soft, fuzzy boundaries with lower damping contrast (<6 dB) and often align with internal ocean wave patterns.
2. **Low-Wind Areas**: Feature broad, diffuse transitions rather than sharp curvilinear boundaries.
3. **Rain Cells & Grease Ice**: Have distinct polarimetric texture and atmospheric attenuation signatures.
4. **Mineral Oil Discharges**: Characteristic sharp, high-contrast boundaries, elongated plume morphology matching prevailing vessel tracks or current drift lines, and localized origins.
