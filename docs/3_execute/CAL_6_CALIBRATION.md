# CAL_6_CALIBRATION.py

## Overview

This script runs the calibration of the model for a specified catchment linked to a station ID. This script will run many instances of LISFLOOD in parallel, driven by the DEAP optimisation algorithm.

Amongst many options provided in the settings file, the script requires the following files in order to run properly:
- A series of static maps and forcings maps, located in the `subcatchment_path/STATION_ID/maps` directory.
- A CSV file containing the station metadata, located in the `subcatchment_path/STATION_ID/station` directory.
- A CSV file containing the observations at the station, located in the `subcatchment_path/STATION_ID/station` directory.
- A PCRaster map containing the inflow points of the catchment, located in the `subcatchment_path/STATION_ID/inflow` directory.
- A LISFLOOD settings template, from the `LISFLOODSettings` entry in the settings

The main outputs of the calibration are the following files:
- `front_history.csv`: contains the main statistics of each generation of the DEAP optimisation algorithm, such as the minimum, maximum, mean and standard deviation of the efficiency.
- `paramsHistory.csv`: contains all the parameters used for each instance of LISFLOOD together with the resulting statistics, such as the correlation with observations or the Kling Gupta Efficiency (KGE).
- `pareto_front.csv`: contains the optimal parameters of the calibration.
- `pHistoryWRanks.csv`: contains a ranking of the best candidates throughout the calibration.

At the beginning of the calibration, auxiliary data is also computed. Typicall the inflow signal from the upstream catchments are gathered in the `subcatchment_path/STATION_ID/inflow/chanq.tss`.

## Usage

To use this script, you need to provide the settings file `SETTINGS`, the station ID `STATION_ID` and the number of CPUS `N_CPUS` as command-line arguments:

```bash
CAL_6_CALIBRATION.py SETTINGS STATION_ID N_CPUS
```

And optional argument `--seed NUMBER` can be provided to set the seed of the random number generation used by the DEAP algorithm. This can be useful for testing purposes and allows to obtained reproducable results, meaning we get exactly the same results by running the calibration twice. 
