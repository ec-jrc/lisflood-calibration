## Prepare static data

To calibrate the model for your catchments/subcatchments, you need input data:

- time series of observed discharge data
- time series of input meteo variables for the entire period of calibration (e.g. ERA5 datasets)
 
Copy those datasets in a folder on your machine. Paths must be configured in a `settings.txt` file.

## Edit a settings file

In settings file you can set all configuration parameters for calibration. Like path to scripts to run the lisflood model (or any other hydrological model). 
So you need to have access and execution rights to the model you want to calibrate.
 
The easy way is to make a copy of settings_calibration.txt (e.g. my_settings.txt) and edit it according your requirements.

Below you find an example of settings file:

```ini 
Root=/absolute/path/to/lisflood-calibration/  
ForcingStart=1/1/1986 00:00  # Starting of Meteo Forcings Precipitation Evapotranspiration TAvg  
ForcingEnd=31/12/2017 00:00  # Ending of Meteo Forcings Precipitation Evapotranspiration TAvg  
SubsetMeteoData=0 
WarmupDays=366  # Number of days for the Warmup period of the Model  
MinQlength=4  # Catchments with streamflow records shorter than the number of years specifed by MinQlength in settings.txt will not be processed 
No_of_calibration_lists=21 # Number of lists of catchments to process in parallel. i.e. If one agrees 10 nodes for running the calibration, a maximum of 10 (or less depending on direct links between subcatchments) lists will be generated with the name CatchmentsToProcess_XX.txt and 10 will be the maximum number of jobs submitted at the same time.  
MaxPercArea=0.1  

[CSV]  
Qgis=/absolute/path/to/Qgis.csv # File containing metadata of Stations available with observation  
Qtss=/absolute/path/to/Qts.csv # Observed data   

[Path]  
Temp=%(Root)s/temp  
Result=%(Root)s/result  
Templates=%(Root)s/templates       
SubCatchmentPath=%(Root)s/catchments  
ParamRanges=%(Root)s/ParamRanges_LISFLOOD.csv   # Values range for parameters to calibrate   
CatchmentDataPath=/FLOODS/lisflood/CalibrationTest/static_data  # static maps for lisflood model (landuse wateruse area ldd etc)  
MeteoData=/FLOODS/glofas/meteo/ERA5/ # path to netcdf forcing data  
PCRHOME=/ADAPTATION/usr/anaconda2/bin/ # path to pcraster binaries  
PYTHONCMD=/ADAPTATION/usr/anaconda2/bin/python # path to python executable  (in case of several versions)  

[Templates]  
LISFLOODSettings=%(Root)s/templates/settings_LF.xml # Settings for Lisflood Model  
RunLISFLOOD=%(Root)s/templates/runLF_linux.sh  # Script for launching PreRun and Run of the model, for every parameters combination during genetic algorithm runs   

[DEAP]  
use_multiprocessing=1  # Flag for using multiprocessing, meaning running several lisflood runs on several cores (each using 1 core)  
ngen=16  # number of MAX generation to run  
mu=16  # initial population  
lambda_=32  # size of generation of offsprings 
```

## Summary

1. Clone the repository lisflood-calibration or download the code
2. Make a copy of templates/runLF_linux.sh and edit it (you only need to change paths)
3. Make a copy of templates/settings_LF.xml and edit it if needed
4. Prepare static maps (dem, landuse etc.) and netcdf forcing data (meteo input as ERA5 datasets)
5. Make a copy of settings_calibration.txt and edit according your system. You have to change LISFLOODSettings and RunLISFLOOD parameters in order to use your customized files and configure paths to static data.

