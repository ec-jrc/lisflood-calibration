## Prepare static data

To calibrate the model for your catchments/subcatchments, you need input data:

- time series of observed discharge data (e.g. ERA5 datasets)
- time series of input meteo variables for the entire period of calibration
 
Copy those datasets in a folder on your machine. Paths must be configured in a `settings.txt` file.

## Edit a settings file

In settings file you can set all configuration parameters for calibration. Below you find an example:

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
LISFLOODSettings=%(Root)s/templates/settings_LF.xml # Settings for Lisflood Model see documentation on Lisflood Repo  
RunLISFLOOD=%(Root)s/templates/runLF_linux_cut.sh  # Script for launching PreRun and Run for every parameters combination during genetic algorithm runs   

[DEAP]  
use_multiprocessing=1  # Flag for using multiprocessing, meaning running several lisflood runs on several cores (each using 1 core)  
ngen=16  # number of MAX generation to run  
mu=16  # initial population  
lambda_=32  # size of generation of offsprings 
```
