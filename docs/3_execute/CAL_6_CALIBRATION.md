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
## Calibration behaviour and convergence

The calibration is driven by the DEAP optimisation algorithm and runs for a bounded number of generations, controlled by the `[DEAP]` section of the settings file (`min_gen`, `max_gen`, `gen_offset`, `effmax_tol`, ...).

By default the tool can perform up to `max_gen` generations (e.g. 30). The optimisation stops early when both of the following are true:

- The change in the objective function (KGE or KGE-JSD) over the last `gen_offset` generations (e.g. 4) is smaller than `effmax_tol` (e.g. 0.002), i.e. `effmax[gen] - effmax[gen - gen_offset] < effmax_tol`. This "no-improvement" criterion is only evaluated once `gen >= min_gen` and `gen >= gen_offset`.
- If the optional statistical stall check is enabled (`apply_statistical_stall_check = 1`), an additional t-test on KGE/KGE-JSD is performed between the current population and the combined statistics of the previous `gen_offset` populations. The optimisation keeps going only if a *significant improvement* is detected, that is when `mean_current - mean_previous > 0.0001`, the t-test p-value `< 0.05`, and `std_current > 0.001`. Otherwise (negative t-test, i.e. only a small, non-significant difference between the samples) the run is considered stalled and stops.

When the statistical stall check is disabled, reaching the no-improvement criterion alone is enough to stop the calibration.

The best parameter set of a run is chosen from the `pHistoryWRanks.csv` file using a combined ranking of KGE/KGE-JSD, correlation and SAE (sum of absolute errors). Each candidate receives a normalised rank score for correlation (`corrRank`), SAE (`saeRank`) and KGE/KGE-JSD (`KGERank`/`KGEJSDRank`); the combined `paretoRank` is the product of these three scores, and the best candidate is the one with the smallest `paretoRank`.

## Calibration workflow using the KGE-JSD objective (retries)

When calibrating with the KGE-JSD objective (also referred to as JDKGE), the tool applies a quality check to the selected solution and can retry the calibration up to two additional times before falling back to a plain KGE calibration. The workflow is as follows.

### 1. First KGE-JSD run

Run the KGE-JSD calibration using the initial seed (if set; in our reference setup `seed = 13`). Select the best individual of this first KGE-JSD calibration (`bestKGEJSD1st`) from the `pHistoryWRanks.csv` file (smallest `paretoRank`).

Two checks are then applied:

- **Low-KGE check.** If the KGE term of `bestKGEJSD1st` is `< -0.41`, the calibration of this catchment is stopped: the long-term run is executed and the downstream catchments are stopped (the simulated best streamflow and `chanq` files are renamed accordingly). This step is skipped unless `stop_on_low_kgejsd` is set in the settings file (i.e. it is `0`/unset or `False` by default). The stop-and-longterm behaviour is triggered when `stop_on_low_kgejsd == 1`.
- **Quality check.** If the JSD term of `bestKGEJSD1st` is `> 0.1`, or `DeltaCorrelation > 0.095`, or `DeltaKGE > 0.095`, the KGE-JSD quality check is considered failed and a new KGE-JSD calibration is run with a new seed (`seed = 233`). Otherwise the long-term run is executed directly using `bestKGEJSD1st`.

`DeltaCorrelation` is the difference between the maximum correlation value found in `pHistoryWRanks.csv` and the correlation value of `bestKGEJSD1st`. `DeltaKGE` is the difference between the maximum KGE value in `pHistoryWRanks.csv` and the KGE value of `bestKGEJSD1st`.

### 2. Second KGE-JSD run (new seed)

After the new KGE-JSD calibration (seed 233), a new individual `bestKGEJSD2nd` is selected. The same quality check is applied to `bestKGEJSD2nd`. If it passes, this individual is used for the long-term run. Otherwise, a new calibration is run using the plain **KGE** objective (reverting to the original seed).

### 3. Fallback KGE run and final comparison

After the KGE calibration, a new individual `bestKGE` is selected. It is compared against the best of the two KGE-JSD individuals (`bestKGEJSD1st` and `bestKGEJSD2nd`).

First, the reference `bestKGEJSD` is chosen between the two KGE-JSD runs:

- The `bestKGEJSD` individual is the one having `JSD <= 0.1`.
- If both have `JSD <= 0.1`, or both have `JSD > 0.1`, then `bestKGEJSD1st` is selected when its correlation is at least 0.05 higher than the correlation of `bestKGEJSD2nd`, or when the correlations are similar (difference `<= 0.05`) but its KGE value is higher. Otherwise `bestKGEJSD2nd` is used as `bestKGEJSD`.

Then `bestKGE` and `bestKGEJSD` are compared:

- If the JSD term of `bestKGEJSD` is `> 0.1`, `bestKGE` is selected if its correlation or its KGE is better than that of `bestKGEJSD`.
- If the JSD term of `bestKGEJSD` is `<= 0.1`, `bestKGE` is selected only if its correlation or its KGE value is at least 0.05 higher than the correlation or KGE value of `bestKGEJSD`.

If `bestKGE` is not selected, the calibration resumes (restores) the outputs of the chosen KGE-JSD attempt (`1st` or `2nd`), which are then used for the long-term run.

Throughout this workflow, the outputs of each intermediate run (`out` folder, `settings`, `pHistoryWRanks.csv`, `paramsHistory.csv`, `front_history.csv`, `runs_log.csv`) are renamed with a suffix identifying the run (`KGEJSD_1st`, `KGEJSD_2nd`, `KGE`), and calibration status files (`CalibrationStatus_*_run_*.txt`) are written to record why each retry was triggered.
## Management of calibration parameters

Not all calibration parameters are relevant for every catchment. After the model is initialised, the framework filters the parameter ranges (`filter_param_ranges`) and removes the parameters that do not apply to the current subcatchment. Removed parameters are simply not calibrated and keep their template/default value.

The filtering rules are:

- **Lakes (`LakeMultiplier`).** If the catchment has no lakes (LISFLOOD option `simulateLakes` is off), `LakeMultiplier` is not calibrated. When lakes are present and `split_lake_params = 1` in the `[DEAP]` settings, a catchment with more than one lake gets a dedicated `LakeMultiplier_<lakeId>` parameter per lake, so that each lake is calibrated with its own multiplier value; with a single lake (or `split_lake_params = 0`) a single `LakeMultiplier` is used for all lakes.
- **Reservoirs (`ReservoirFloodStorage`, `ReservoirFloodOutflowFactor`).** If the catchment has no reservoirs (LISFLOOD option `simulateReservoirs` is off), the reservoir parameters `ReservoirFloodStorage` and `ReservoirFloodOutflowFactor` are not calibrated.
- **MCT channel roughness (`CalChanMan3`).** The Muskingum-Cunge-Todini channel roughness multiplier `CalChanMan3` is calibrated only when MCT routing is active for the catchment (LISFLOOD option `MCTRouting` is on, i.e. the MCT mask has at least one active pixel within the catchment mask). Otherwise it is not calibrated.
- **Snow melt (`SnowMeltCoef`).** `SnowMeltCoef` is not calibrated when the catchment is effectively snow-free, i.e. when the minimum snow-adjusted temperature over the mask and the whole calibration period (`min_TAvgS`, computed by `CAL_5c_FORCING_STATS.py`) is greater than the LISFLOOD `TempSnow` threshold. In that case no pixel ever accumulates snow and the parameter has no effect.
- **Transmission losses (`TransSub`).** When the aridity check is enabled (`use_aridity_index_check`), `TransSub` is not calibrated for non-arid catchments, i.e. when the minimum pixel-wise aridity index `min_AridIdx` (from `CAL_5c_FORCING_STATS.py`) is `>= 0.5`.

## Complementary performance metrics in paramsHistory.csv

In addition to the KGE components used to drive the optimisation, each evaluated parameter set is stored in `paramsHistory.csv` together with a set of complementary performance metrics, computed for every individual. On top of the core statistics (`Kling Gupta Efficiency`, `Correlation`, `Signal ratio (s/o) (Bias)`, `Noise ratio (s/o) (Spread)`, `sae`), the following metrics are recorded:

- `Evaporative Index` and `Fractional Budyko Distance`: Budyko-based water-balance diagnostics (using the catchment `precip_budyko` and `PET_budyko` computed by `CAL_5c_FORCING_STATS.py`).
- `NSE`: Nash-Sutcliffe Efficiency.
- `FDC_FHV` and `FDC_FLV`: high-flow and low-flow bias of the flow duration curve (Yilmaz et al., 2008, https://doi.org/10.1029/2007WR006716).
- `FDC_mFHV` and `FDC_mFLV`: modified high-/low-segment FDC volume biases based on the top 2% / bottom segment of observed flows.
- `KGE_JSD` and `JSD`: the Jensen-Shannon Divergence augmented KGE (JDKGE) and its JSD component (flow duration curve distributional similarity).

These metrics are recorded for diagnostic purposes for all calibrations. `KGE_JSD`/`JSD` are additionally used by the KGE-JSD calibration workflow and its quality checks described above.
