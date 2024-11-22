# CAL_1_FILTER_STATIONS.py

## Overview
This script is used for filtering stations based on certain criteria for calibration purposes.

The filtering criteria are the following:
- Check if a minimum number of observations are available for the calibration, taking into account that the LISFLOOD model needs at least a few years to initialise. The scripts can also distinguish from 6-hourly to daily observations following the information provided in the settings file and the stations metadata CSV file.
- Manually include or exclude stations using the stations group ID (integer) specified at the command line. This means users can create test groups by increasing the stations group ID of a few stations. For instance group ID 1 will use all the stations and group ID 2 only a subset.  

The script will print a summary of the stations filtered and then write the calibration stations in the CSV file provided in the settings file.

## Usage
To use this script, you need to provide the settings file, stations CSV file, and station group ID as command-line arguments. For example:

```bash
CAL_1_FILTER_STATIONS.py SETTINGS_FILE STATIONS_CSV GROUP_ID
```
An example of settings file can be found in [Create a settings file](../3_data/index.md). For this script, the following options are required in the settings file ```SETTINGS_FILE```:

- ```forcing_start```: The start date for the forcing data.
- ```forcing_end```: The end date for the forcing data.
- ```timestep```: The timestep for the calibration. It only supports 360 (6 hours) and 1440 minutes (1 day).
- ```observed_discharges```: The observed discharges for the stations.
- ```stations_data```: The CSV file containing the stations metadata, **THIS WILL BE THE MAIN OUTPUT OF THIS SCRIPT**.

The input ```STATIONS_CSV``` file should at least contain the following entries per station:
- ```Min_calib_days```: minimum number of days in observations to include the station in the calibration.
- ```Spinup_days```: number of days required to spinup the model, that means the calibration period can only start after ```forcing_start+spinup_days```.
- ```CAL_TYPE```: 
- ```EC_calib```: group ID of the station, used to include or exclude stations from the calibration manually, the stations are filtered using this entry in the CSV and the command line argument station group ID.