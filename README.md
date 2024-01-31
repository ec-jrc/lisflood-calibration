# Lisflood OS

This repository hosts source code of LISFLOOD Calibration tool.
Go to [Lisflood OS page](https://ec-jrc.github.io/lisflood/) for more information.

Other useful resources

| **Project**         | **Documentation**                                         | **Source code**                                                 |
| ------------------- | --------------------------------------------------------- | --------------------------------------------------------------- |
| Lisflood            | [Model docs](https://ec-jrc.github.io/lisflood-model/)    | https://github.com/ec-jrc/lisflood-code                         |
|                     | [User guide](https://ec-jrc.github.io/lisflood-code/)     |                                                                 |
| Lisvap              | [Docs](https://ec-jrc.github.io/lisflood-lisvap/)         | https://github.com/ec-jrc/lisflood-lisvap                       |
| Calibration tool    | [Docs](https://ec-jrc.github.io/lisflood-calibration/)    | https://github.com/ec-jrc/lisflood-calibration (this repository)|
| Lisflood Utilities  |                                                           | https://github.com/ec-jrc/lisflood-utilities                    |
| Lisflood Usecases   |                                                           | https://github.com/ec-jrc/lisflood-usecases                     |


This repository contains a collection of Python tools and libraries for calibrating the LISFLOOD hydrological model.

The calibration tools were created by Hylke Beck 2014 (JRC, Princeton) hylkeb@princeton.edu.
The submodule Hydrostats was created 2011 by Sat Kumar Tomer (modified by Hylke Beck).
Modified by Feyera Aga Hirpa in 2015 (JRC) feyera.hirpa@ouce.ox.ac.uk.
Modified by Valerio Lorini (valerio.lorini@ec.europa.eu) and Alfieri Lorenzo (lorenzo.alfieri@ec.europa.eu) in 2018.
The calibration tools were completely refactored by ECMWF (corentin.carton@ecmwf.int) for the EFAS 5.0 and GloFAS 4.0 releases in 2023.

The calibration procedure consists of several Python scripts to be run consecutively. If needed, the tools are also available through a Python library called `liscal`:
```python
import liscal
```
All the scripts found in the *bin* folder are just entry points for `liscal` functionalities.

### Installation

You can install the calibration tools and the `liscal` library from GitHub:
```bash
git clone https://github.com/ec-jrc/lisflood-calibration.git
cd lisflood-calibration
pip install .
```
Once installed, you can run the scripts located in the bin folder or build your own workflow using the liscal library.

### Calibration Workflow

The calibration worfklow is split into five main parts:
1. Setting up the calibration and prepare the static data and station list.
2. Filter the stations and compute the catchments dependencies graph together with some pre-processing maps (catchments masks, stations outlets, etc.).
Scripts involved in this step:
    - `CAL_1_FILTER_STATIONS.py`: filters stations and keep those with enough observations.
    - `CAL_2_HYDRO_DEPENDENCIES.py`: computes the hydrological dependencies between the stations.
3. Cut all the static maps and forcings and create one folder per catchment, named using the station ID of the station corresponding to the catchment. Best practice is to save the cutmaps outputs in a different folder than the calibration to avoid having to recompute them.
Scripts involved in this step:
    - `CAL_3_MASK.py`: extracts the mask of a specified stations.
    - `CAL_4_CUT_MAPS.py`: cuts all the static and forcing maps from the mask computed in the previous step.
4. From these maps, calibrate the catchment with respect to the observations at the station outlet of the catchment.
Scripts involved in this step:
    - `CAL_5_EXTRACT_STATION.py`: extracts station metadata and observation for a specified catchment.
    - Optional: `CAL_5b_SET_PRERUN_IN_SETTINGS.py` to set prerun date in settings file from station metadata.
    - `CAL_6_CALIBRATION.py`: runs the calibration for a specified catchment.
    - `CAL_7_LONGTERM_RUN.py`: runs the long term run for a specified catchment using the calibrated parameters.
    - Optional: `CAL_8_POSTPROCESSING.py` computes calibration statistics for the catchment.
    - Optional: `CAL_9_DIAGNOSTICS.py` computes calibration diagnostics for a list of catchments.
5. Once all the catchments have been calibrated, we can compute the global parameter map:
    - `CAL_10_PARAMETER_MAPS.py`: concatenates all the catchments calibrated parameters into PCRaster parameters maps.

These scripts need to be run in sequence. Note that steps 3 and 4 are executed per catchment (or station). As the process can be heavy, it is up to the user to dispatch the workflow on the computing architecture and to make sure the upstream catchnments are run first. An example of workflow using PBS scheduling can be found in CAL_6-7_PERFORM_CAL_PBS.py.

for more details, please refer to the main documentation: https://ec-jrc.github.io/lisflood-calibration/
