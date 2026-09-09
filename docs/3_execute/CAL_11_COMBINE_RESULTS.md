# CAL_11_COMBINE_RESULTS.py

## Overview

This script combines calibration results from all catchments into a single summary CSV file and an optional GeoJSON file for spatial visualisation. It reads the best-ranked parameter history (`pHistoryWRanks.csv`) from each catchment folder and merges it with station metadata.

The script produces the following outputs in a `calibration/summary/` directory:
- `calibration_summary.csv`: tabular summary with station metadata and calibration results (KGE, parameters, rankings) for all catchments.
- `calibration_summary.json`: GeoJSON file with intercatchment polygon geometries and calibration attributes, suitable for visualisation in GIS tools.

The spatial data is derived by vectorising each catchment's mask map (`masksmall.map`) into polygon geometries using rasterio and geopandas.

## Usage

```bash
CAL_11_COMBINE_RESULTS.py SETTINGS CUTMAPS_PATH
```

### Arguments

- `SETTINGS`: path to the calibration settings file. The script derives the main directory from this path (expects the structure `.../calibration/settings.txt`) and looks for station metadata at `data/stations/stations_data.csv` relative to the main directory.
- `CUTMAPS_PATH`: path to the cutmaps directory containing per-catchment subfolders with `maps/ldd.nc` and `maps/masksmall.map`.

## Requirements

This script requires the following Python packages:
- `geopandas`
- `rasterio`
- `xarray`
- `pcraster`

## Notes

- Catchments without a `pHistoryWRanks.csv` file (i.e. not yet calibrated) are skipped with a warning but still appear in the output CSV with NaN values for calibration columns.
- Catchments without spatial data (missing mask or LDD maps) are skipped for the GeoJSON output.
- The GeoJSON uses 8-connectivity for vectorisation to match the LDD diagonal flow directions.
