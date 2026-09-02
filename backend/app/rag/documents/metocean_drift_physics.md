# Metocean Dynamics & Lagrangian Oil Spill Drift Modeling

**Source**: NOAA National Ocean Service & Copernicus Marine Environment Monitoring Service (CMEMS)  
**Title**: Hydrodynamic Transport, Weathering and Backward Trajectory Modeling of Marine Hydrocarbon Slicks  
**Date**: 2024-02-10  
**Document Type**: Scientific Methodology Guide  
**URL**: https://response.restoration.noaa.gov/oil-and-chemical-spills/oil-spills/response-tools/gnome-suite.html  

## 1. Governing Lagrangian Transport Equation
The displacement velocity vector $\vec{v}_{spill}$ of a surface oil slick is modeled by the Eulerian-Lagrangian superposition of the sea surface current velocity $\vec{v}_{current}$ and the 10-meter wind velocity $\vec{v}_{wind}$:

$$\vec{v}_{spill} = \vec{v}_{current} + \alpha_{wind} \cdot \mathbf{R}(\theta_{Coriolis}) \cdot \vec{v}_{wind} + \vec{v}'_{diffusion}$$

Where:
* $\vec{v}_{current}$: Surface ocean current vector (top 0.5m layer).
* $\alpha_{wind}$: Wind leeway drift factor, empirically established between 0.030 and 0.035 (nominally 3% of wind speed at 10m height).
* $\mathbf{R}(\theta_{Coriolis})$: Coriolis deflection rotation matrix (typically deflection of 5° to 15° to the right of downwind in the Northern Hemisphere due to Ekman spiral dynamics).
* $\vec{v}'_{diffusion}$: Stochastic turbulent diffusion component.

## 2. Backward Trajectory (Backtracking) Methodology
To estimate the probable discharge origin location $(x_0, y_0)$ at discharge time $t_{release} = t_{obs} - \Delta t$:
$$\vec{x}(t_{release}) = \vec{x}(t_{obs}) - \int_{t_{obs} - \Delta t}^{t_{obs}} \vec{v}_{spill}(\tau) d\tau$$

Uncertainty grows non-linearly with backtracking duration $\Delta t$, driven by spatial-temporal wind shear and sub-grid ocean eddies. An uncertainty ellipse with major semi-axis $r_{unc} = r_0 + \kappa \cdot \Delta t$ delineates the search zone for candidate AIS vessels.
