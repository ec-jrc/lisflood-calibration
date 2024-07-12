# CAL_5_EXTRACT_STATION.py

## Overview

This script is used to extract the data for a specific station STATION_ID from the following files provided in the settings file SETTINGS:
- `stations_data`: CSV file containing the metadata of the stations.
- `observed_discharges`: CSV file containing the observed discharge.

The script will write the outputs in the following files:
- stations metadata: `subcatchment_path/STATION_ID/station_data.csv`
- observations: `subcatchment_path/STATION_ID/observations.csv`
with `subcatchment_path` provided in the `Path` section of the settings files SETTINGS.

The script also raises an exeption if not enough observation data is available for the station.
## Usage

To use this script, you need to provide the settings file SETTINGS and the station ID STATION_ID as command-line arguments:

```bash
CAL_4_EXTRACT_STATION.py SETTINGS STATION_ID
```

You can also use the --no_check option to turn off the check for whether enough observation data is available.
