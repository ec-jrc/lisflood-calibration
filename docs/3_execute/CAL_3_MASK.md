# CAL_3_MASK.py

## Overview

This script create the mask and outlets PCRaster maps for one or multiple catchments attached to a station.

The script will generate the following outputs, using the `subcatchment_path` entry in the settings file and the STATION_ID of the station:
```bash
subcatchment_path/STATION_ID/inflow/inflow.map  # inflow stations of the catchment on the grid
subcatchment_path/STATION_ID/maps/outlets.map  # location of the outflow station on the grid
subcatchment_path/station_id/maps/mask.map  # mask of the catchment on the grid
```

## Usage

To use this script, you need to provide the settings file and the catchments to process file as command-line arguments. For example:

```bash
CAL_3_MASK.py settings.txt catchments.txt
```