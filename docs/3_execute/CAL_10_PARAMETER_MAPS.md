# CAL_10_PARAMETER_MAPS.py

**PAGE IN CONSTRUCTION**

## Overview

This script generates the final parameter maps by aggregating the optimised parameters of all the calibrated catchments.

## Usage

```bash
usage: CAL_10_PARAMETER_MAPS.py [-h] --stations STATIONS --catchments CATCHMENTS --output OUTPUT --params PARAMS [--template TEMPLATE] [--regionalisation REGIONALISATION]

options:
  -h, --help            show this help message and exit
  --stations STATIONS, -s STATIONS
                        Path to stations folder containing interstation_regions.map and stations_data.csv
  --catchments CATCHMENTS, -c CATCHMENTS
                        Path to catchments folder
  --output OUTPUT, -o OUTPUT
                        Output folder
  --params PARAMS, -p PARAMS
                        Path to calibration parameters ranges csv file
  --template TEMPLATE, -t TEMPLATE
                        Path to NetCDF template
  --regionalisation REGIONALISATION, -r REGIONALISATION
                        Path to regionalisation csv file
```