## Prepare static data

To calibrate the model for your catchments/subcatchments, you need input data:

- time series of observed discharge data
- time series of input meteo variables for the entire period of calibration (e.g. ERA5 datasets)
 
Copy those datasets in a folder on your machine. Paths must be configured in a `settings.txt` file.


_**Note**_: To ensure that LISFLOOD runs as fast as possible, check that the meteorological forcing NetCDF fles use chunking, are uncompressed, and do not have time as an unlimited dimension. 
You can check this by opening the files using Panoply (http://www.giss.nasa.gov/tools/panoply/). 
In addition, the forcing data should be stored on the local disk instead of over the network.


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

## Format of Qgis and Qtss files

### Qgis.csv (metadata)

```csv
ID,Provider_1,ProvID_1,Provider_2,ProvID_2,Stationnam,RiverName,RiverBasin,Country,CountryNam,Continent,HydroRegio,DrainArPro,DrainArLDD,YProvided,XProvided,YCorrected,XCorrected,Suitable,ValidGLOFAS,Legend_for,Comments,StartDate,EndDate,AddedDate,Lake,Reservoir,Lastupdate
G0001,GRDC,2903430,NA,NA,Stolb,Lena,Lena,RU,Russian Federation,Asia,512,2460000,2443690,72.37,126.8,72.25,126.75,1,1, , ,1/1/1951,12/31/2002, ,0,0, 
G0002,GRDC,2999910,NA,NA,7.5Km D/S Of Mouth Of River Pur,Olenek,Olenek,RU,Russian Federation,Asia,512,198000,218622,72.12,123.22,72.15,123.35,1,1, , ,1/1/1953,12/31/2011, ,0,0, 
G0003,GRDC,2999150,NA,NA,Saskylakh,Anabar,Anabar,RU,Russian Federation,Asia,512,78800,86783,71.98,114.057,71.95,114.05,1,1, , ,6/1/1954,11/22/2011, ,0,0, 
G0004,GRDC,2999850,NA,NA,Khatanga,Khatanga,Khatanga,RU,Russian Federation,Asia,512,275000,285113,71.98,102.45,71.95,102.15,1,1, , ,6/7/1971,9/30/1991, ,0,0, 
```
### Qtss.csv (observed data per reported station in QGis.csv, column oriented)

```csv
DATE,G0001,G0002,G0003,G0004
1/1/1914,,,41.5,51.5
2/1/1914,,,41.5,51.5
3/1/1914,,,41.5,51.5
4/1/1914,,,41.5,51.5
...
```

## Summary

1. Clone the repository lisflood-calibration or download the code
2. Make a copy of templates/runLF_linux.sh and edit it (you only need to change paths)
3. Make a copy of templates/settings_LF.xml and edit it if needed
4. Prepare static maps (dem, landuse etc.) and netcdf forcing data (meteo input as ERA5 datasets)
5. Prepare the Qgis csv file. This is a file containing stations' metadata
6. Prepare the Qtss csv file. This file contains observed discharge data for each station
7. Make a copy of settings_calibration.txt and edit according your system
You have to change LISFLOODSettings and RunLISFLOOD parameters in order to use your customized files and configure paths to static data.

