# CAL_7_LONGTERM_RUN.py

## Overview

This script runs the long term run of the model for a specified catchment, meaning that it will run LISFLOOD for the whole forcing period. The LISFLOOD outputs from this task can be then used as inflow for downstream catchments (`dis.tss` and `chanq.tss`).

The output files are written in the `subcatchment_path/STATION_ID/out/long_term_run` directory.

## Usage

To use this script, you need to provide the settings file `SETTINGS` and the station ID `STATION_ID` as command-line arguments:

```bash
CAL_7_LONGTERM_RUN.py SETTINGS STATION_ID
```
