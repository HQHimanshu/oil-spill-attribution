# Automatic Identification System (AIS) Maritime Tracking & Gap Analysis Standards

**Source**: International Maritime Organization (IMO) & US Coast Guard Navigation Center  
**Title**: Carriage Requirements, Signal Integrity, and Intentional Dark Activity Detection in Terrestrial and Satellite AIS  
**Date**: 2023-09-20  
**Document Type**: Maritime Technical Standard  
**URL**: https://www.navcen.uscg.gov/automatic-identification-system-overview  

## 1. IMO Carriage Requirements (SOLAS Chapter V, Regulation 19)
All ships of 300 gross tonnage (GT) and upwards engaged on international voyages, cargo ships of 500 GT and upwards not engaged on international voyages, and all passenger ships irrespective of size are mandated to maintain an operational Class-A AIS transponder at all times.

## 2. AIS Gap Analysis & "Going Dark"
Vessels intentionally disabling or manipulating AIS transmitters create data transmission anomalies:
* **Normal Reporting Interval**: Class A transponders broadcast position every 2 to 10 seconds while underway (>3 knots), and every 3 minutes while at anchor.
* **Transmission Gap (>15 minutes)**: A blackout period where a vessel transits through monitored coastal or satellite footprints without broadcasting pings.
* **Correlated Discharges**: Vessels turning off AIS prior to an illegal discharge and resuming transmission afterwards represent high-risk suspect behavior. The system explicitly flags `AIS DATA GAP DETECTED` rather than interpolating or inventing missing vessel positions.
