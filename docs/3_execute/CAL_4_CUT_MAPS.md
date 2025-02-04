# CAL_4_CUT_MAPS.py

## Overview

This script will "cut" (i.e. extract a subset) a list of NetCDF or PCRaster maps and extract only the data corresponding to the mask computed from the CAL_3_MASK.py script, leading to a smaller dataset that can be used for the calibration, avoid large I/O during the calibration of the model for a dedicated catchment.

The provided path to the maps can be either a folder or direct path to a map.

## Usage

To use this script, you need to provide the settings file, the path to the maps and the station ID attached to the catchment to process as command-line arguments. For example:

```bash
CAL_4_CUT_MAPS.py settings.txt /path/to/maps STATION_ID
```

## Note
The `CAL_4b_CUT_MAPS_list.py` script provides a solution to loop all the catchments and cut the maps one catchment at a time. This script could take a long time to run for a large number of catchments, so scheduling the catchments one by one on an HPC can accelerate the process for large domains.
