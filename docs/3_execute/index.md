# Main calibration workflow

The calibration worfklow is split into five main parts:
1. Setting up the calibration and prepare the static data and station list.
2. Filter the stations and compute the catchments dependencies graph together with some pre-processing maps (catchments masks, stations outlets, etc.).
Scripts involved in this step:
    - [CAL_1_FILTER_STATIONS.py](CAL_1_FILTER_STATIONS.md): filters stations and keep those with enough observations.
    - [CAL_2_HYDRO_DEPENDENCIES.py](CAL_2_HYDRO_DEPENDENCIES.md): computes the hydrological dependencies between the stations.
3. Cut all the static maps and forcings and create one folder per catchment, named using the station ID of the station corresponding to the catchment. Best practice is to save the cutmaps outputs in a different folder than the calibration to avoid having to recompute them.
Scripts involved in this step:
    - [CAL_3_MASK.py](CAL_3_MASK.md): extracts the mask of a specified stations.
    - [CAL_4_CUT_MAPS.py](CAL_4_CUT_MAPS.md): cuts all the static and forcing maps from the mask computed in the previous step.
4. From these maps, calibrate the catchment with respect to the observations at the station outlet of the catchment.
Scripts involved in this step:
    - [CAL_5_EXTRACT_STATION.py](CAL_5_EXTRACT_STATION.md): extracts station metadata and observation for a specified catchment.
    - [CAL_5b_SET_PRERUN_IN_SETTINGS.py](CAL_5b_SET_PRERUN_IN_SETTINGS.md) (optional): to set prerun date in settings file from station metadata.
    - [CAL_6_CALIBRATION.py](CAL_6_CALIBRATION.md): runs the calibration for a specified catchment.
    - [CAL_7_LONGTERM_RUN.py](CAL_7_LONGTERM_RUN.md): runs the long term run for a specified catchment using the calibrated parameters.
    - [CAL_8_POSTPROCESSING.py](CAL_8_POSTPROCESSING.md) (optional): computes calibration statistics for the catchment.
    - [CAL_9_DIAGNOSTICS.py](CAL_9_DIAGNOSTICS.md) (optional): computes calibration diagnostics for a list of catchments.
5. Once all the catchments have been calibrated, we can compute the global parameter map:
    - [CAL_10_PARAMETER_MAPS.py](CAL_10_PARAMETER_MAPS.md): concatenates all the catchments calibrated parameters into PCRaster parameters maps.

These scripts need to be run in sequence. Note that steps 3 and 4 are executed per catchment (or station). As the process can be heavy, it is up to the user to dispatch the workflow on the computing architecture and to make sure the upstream catchnments are run first. An example of workflow using PBS scheduling can be found in CAL_6-7_PERFORM_CAL_PBS.py.