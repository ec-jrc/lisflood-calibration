# CAL_5c_FORCING_STATS.py

## Overview

This script computes forcing statistics for one or more stations. It reads meteorological forcing data (precipitation, potential evapotranspiration, and average temperature) from the NetCDF files referenced in the LISFLOOD settings template and computes catchment-averaged quantities used for Budyko analysis and calibration parameter enabling checks.

The script writes the following fields into the station's `station_data.csv`:
- `precip_budyko`: total catchment-averaged precipitation over the calibration period (mm).
- `PET_budyko`: total catchment-averaged potential evapotranspiration over the calibration period (mm).
- `min_AridIdx`: minimum pixel-wise aridity index (P/PET) across the catchment mask.
- `min_TAvgS`: minimum snow-adjusted temperature across the catchment mask, used to decide whether to enable the SnowMelt calibration parameter.

These statistics are used by the calibration framework (CAL_6) to conditionally enable or disable certain parameters (e.g. TransSub for arid catchments, SnowMelt for warm catchments) and to compute the Budyko evaporative index during calibration.

## Usage

To use this script, you need to provide the settings file `SETTINGS` and a station identifier `STATION`:

```bash
CAL_5c_FORCING_STATS.py SETTINGS STATION
```

- `SETTINGS`: path to the calibration settings file.
- `STATION`: either a single station ID (integer) or a path to a text file listing station IDs (one per line).

## Requirements

This script requires access to the full forcing data (precipitation, PET, temperature maps) at the domain resolution for the entire calibration period. It also requires the LISFLOOD settings template and associated static maps (forest fraction, pixel area, elevation standard deviation).
