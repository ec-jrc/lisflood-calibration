# CAL_4c_CUT_MAPS_parallel.py

## Overview

This script is the parallel version of `CAL_4b_CUT_MAPS_list.py`. It cuts global static and forcing maps to subcatchment extents for a list of stations, processing multiple stations concurrently using Python's `ThreadPoolExecutor`.

Internally, it delegates the actual map cutting to `CAL_4_CUT_MAPS.py` for each station, but launches multiple instances in parallel to accelerate processing for large domains with many catchments.

The script skips stations whose output maps already exist and are non-empty, making it safe to re-run after partial completions or interruptions.

## Usage

```bash
CAL_4c_CUT_MAPS_parallel.py SETTINGS PATH_MAPS STATION_LIST [--max-workers N]
```

### Arguments

| Argument | Description |
|----------|-------------|
| `SETTINGS` | Path to calibration settings file |
| `PATH_MAPS` | Path to global maps directory (or a single map file) |
| `STATION_LIST` | Path to a text file listing station IDs to process (one per line) |
| `--max-workers` | Optional: maximum number of parallel workers (default: auto, based on available CPUs) |

### Example

```bash
python CAL_4c_CUT_MAPS_parallel.py settings.txt /data/global_maps/ catchments_to_process.txt --max-workers 8
```

## How it works

1. Reads the settings file to locate the subcatchment path and stations metadata.
2. Loads the station list and sorts stations by draining area (ascending) to process smaller catchments first.
3. For each station, checks whether output maps already exist. If all maps are present, the station is skipped.
4. For stations that need processing, spawns a parallel call to `CAL_4_CUT_MAPS.py` with the `--use-dask-config` flag.
5. Waits for all parallel tasks to complete.

## Notes

- This script requires `CAL_4_CUT_MAPS.py` to be located in the same directory.
- The `--use-dask-config` flag is automatically passed to enable dask-based chunked processing for large NetCDF files.
- Stations not found in the station list file are silently skipped.
- If running on an HPC with limited I/O bandwidth, consider setting `--max-workers` to a reasonable value to avoid disk contention.
