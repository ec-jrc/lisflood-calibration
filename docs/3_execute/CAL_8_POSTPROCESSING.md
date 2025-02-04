# CAL_8_POSTPROCESSING.py

## Overview

This script generates figures associated with the main calibration statistics of the catchments:
- "Speedometer" plot with Kling Gupta Efficiency (KGE), correlation, Bias, etc.
- Monthly box plot of percentiles
- Discharge time series

This script requires an ECMWF library "plotflood" in order to run. Please contact corentin.carton@ecmwf.int for more information.

## Usage

To use this script, you need to provide the settings file `SETTINGS` and the station ID `STATION_ID` as command-line arguments:

```bash
CAL_8_POSTPROCESSING.py SETTINGS STATION_ID
```