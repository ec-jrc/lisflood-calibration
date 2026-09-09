# CAL_10_PARAMETER_MAPS_FROMCSV.py

## Overview

This script generates global parameter maps from pre-assembled CSV files containing calibrated and regionalised parameter values. Unlike `CAL_10_PARAMETER_MAPS.py` which reads individual `pareto_front.csv` files from each catchment folder, this script takes consolidated CSVs as input, making it suitable for workflows where parameter values have already been collected or post-processed.

The script uses the interstation regions NetCDF map to spatially assign parameter values to each grid cell based on the catchment ID. For grid cells marked as invalid (-1 in the interstation map), the script can optionally apply nearest-neighbor interpolation from the closest valid catchment rather than leaving them as NaN.

The main outputs are NetCDF parameter maps (one per calibration parameter), ready for use in LISFLOOD production runs.

## Usage

```bash
CAL_10_PARAMETER_MAPS_FROMCSV.py --interstation PATH --output PATH --params PATH --calibrated PATH --regionalisation PATH [--useNN]
```

### Arguments

| Argument | Short | Description |
|----------|-------|-------------|
| `--interstation` | `-i` | Path to `interstation_regions.nc` NetCDF file |
| `--output` | `-o` | Output folder for parameter NetCDF maps |
| `--params` | `-p` | Path to calibration parameters ranges CSV file |
| `--calibrated` | `-c` | Path to calibrated parameters CSV file (index = station ID, columns = parameter names) |
| `--regionalisation` | `-r` | Path to regionalisation CSV file (same format as calibrated) |
| `--useNN` | | Flag to enable nearest-neighbor interpolation for invalid (-1) points |

### Input format

The calibrated and regionalisation CSV files must be indexed by station ID and contain one column per calibration parameter matching the names in the parameter ranges file.

## Pixel classification and value assignment

Each pixel in the interstation regions map is classified into one of four categories, which determines its value in the output parameter maps:

| Interstation pixel value | Condition | Output value |
|--------------------------|-----------|--------------|
| NaN | Outside domain (ocean, no-data) | NaN |
| -1 | Station with incomplete parameters (see below) | Nearest valid neighbor (`--useNN`) or NaN |
| Valid ID present in CSVs | Calibrated or regionalised station | Station-specific parameter value |
| Valid ID absent from CSVs (e.g. 0) | Uncalibrated region | Default value from `param_ranges.csv` |

### How -1 values are generated

Before the parameter mapping step, the `handle_empty_rows` function inspects the combined calibrated + regionalised DataFrame. If any station row contains at least one NaN value (i.e. incomplete or failed calibration), all pixels in the interstation map matching that station ID are replaced with -1. This prevents stations with partially-missing results from injecting NaN into the parameter maps.

### Nearest-neighbor interpolation (`--useNN`)

When `--useNN` is enabled, a KDTree is built from all valid pixel coordinates (those that are neither NaN nor -1). For each -1 pixel, the spatially nearest valid pixel is identified and its parameter value is assigned. This provides spatial infill for incomplete catchments based on geographic proximity.

Without `--useNN`, -1 pixels remain NaN in the output.

### Default value assignment

Grid cells whose integer ID passes the valid mask (not NaN, not -1) but is not present in either the calibrated or regionalised CSV will receive the default value from the `DefaultValue` column of the parameter ranges file. This commonly applies to:
- ID 0 (background or uncalibrated land pixels)
- Any station ID that was filtered out before calibration or regionalisation

## Notes

- The script concatenates calibrated and regionalised CSVs into a single lookup table. If a station ID appears in both, the calibrated entry takes precedence (it appears first in the concatenation).
- Output NetCDF files are named `{param_name}_EFASv6.nc` or `{param_name}_GloFASv5.nc` (depending on the domain) with zlib compression enabled.
- The interstation map is expected to contain a `Band1` variable with integer station IDs.
