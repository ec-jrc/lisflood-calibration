## Execute calibration script

### CAL_9_PARAMETER_MAPS.py

```bash
python CAL_9_PARAMETER_MAPS.py settings.txt
```

This script reads the settings file (settings.txt in this case) and produces, for each parameter, a map with the calibrated parameter values assigned to the corresponding interstation regions. 
Ungauged regions are assigned the default parameter values taken from the CSV file specified in the settings file.

### CAL_10_COMPUTE_STATS_AND_FIGURES_TIME_SERIES.py

```bash
python CAL_10_COMPUTE_STATS_AND_FIGURES_TIME_SERIES.py settings.txt
```

Reads the settings file (settings.txt in this case) and loops through the catchments to create figures of the calibration and validation results.

## Output

Results of calibration scripts are saved into catchments folders, under the relative subdirectory (e.g. results for catchment with ID G0001 will be stored under catchments/G0001 folder).
Moreover, in same folders are placed plots generated from last script.