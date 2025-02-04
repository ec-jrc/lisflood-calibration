# CAL_2_HYDRO_DEPENDENCIES.py

## Overview

This script creates the hydrological dependencies between the calibration stations from the description of the river network.
The main output of the script is a CSV file `stations_links.csv` containing the dependencies between stations (outlet vs inlets) but the script also provide the following maps:
- `inlets.map`: PCRaster map containing the inflow points for the catchments where one or more upstream calibration stations are available.
- `gauges.map`: PCRaster map containing the calibration stations.
- `interstation_regions.map`: PCRaster map containing the catchments associated to stations based on the station ID.
- `sampling_frequency.map`: PCRaster map containing how many times a grid point is included in a catchments.

All these maps are used in the calibration to identify the catchment or sub-catchment corresponding to a station and to identify if the catchment depends on the outputs of another catchment. This is important for the calibration as a catchment cannot be calibrated without the inflow data obtained from the upstream catchments. The upstream catchments therefore need to be calibrated first.

## Usage

To use this script, you need to provide the stations data CSV file, LDD map, output stations directory, and temporary directory as command-line arguments. For example:

```bash
CAL_2_HYDRO_DEPENDENCIES.py stations.csv ldd.map /path/to/stations /path/to/temp
```
In this example, stations.csv is the stations data CSV file, ldd.map is the LDD map, /path/to/stations is the output stations directory, and /path/to/temp is a temporary directory that will be used for temporary files (can be useful for debugging).
