# CAL_5b_SET_PRERUN_IN_SETTINGS.py

## Overview

This script is used to set the pre-run start date in a settings file ```SETTINGS``` based on the data from a station CSV file ```STATIONS_CSV``` for a specified station id ```STATION_ID```.

As stations can have different prerun start dates, this script will take the ```prerun_start``` column from the stations metadata CSV file and update the value of the field ```prerun_start``` in the ```Main``` section of the provided settings file.

## Usage

To use this script, you need to provide the settings file SETTINGS, the station metadata STATIONS_CSV and the station ID STATION_ID as command-line arguments:

```bashe
CAL_5b_SET_PRERUN_IN_SETTINGS.py SETTINGS STATIONS_CSV STATION_ID
```
