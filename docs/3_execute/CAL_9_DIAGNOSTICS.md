# CAL_9_DIAGNOSTICS.py

## Overview

This script generates diagnostic plots for a list of calibrated catchments. It reads the calibration outputs (parameter history, observed and simulated streamflow) and produces the following visualisations:
- Discharge time series plot comparing observed vs simulated streamflow.
- Diagnostic variable plots grouped by parameter categories.

The script iterates over all catchment IDs provided in the input file and generates plots for each catchment.

## Usage

To use this script, you need to provide the settings file `SETTINGS`, a catchment list file `CATCHMENTS_FILE`, and optionally a save path for the output plots:

```bash
CAL_9_DIAGNOSTICS.py SETTINGS CATCHMENTS_FILE [SAVEPATH]
```

- `SETTINGS`: calibration settings file (INI format with a `[Path]` section containing `subcatchment_path`).
- `CATCHMENTS_FILE`: CSV file listing the catchment IDs to process (one per row, no header).
- `SAVEPATH` (optional): directory where output plots are saved. Defaults to current directory.
