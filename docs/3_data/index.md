# Prepare static data

To calibrate the model for your catchments/subcatchments, you need input data:

- time series of observed discharge data
- time series of input meteo variables for the entire period of calibration (e.g. ERA5 datasets)
 
Copy those datasets in a folder on your machine. Paths must be configured in a `settings.txt` file.


_**Note**_: To ensure that LISFLOOD runs as fast as possible, check that the meteorological forcing NetCDF fles use chunking, are uncompressed, and do not have time as an unlimited dimension. 
You can check this by opening the files using Panoply (http://www.giss.nasa.gov/tools/panoply/). 
In addition, the forcing data should be stored on the local disk instead of over the network.


# Create a settings file

In the calibration settings file you can set all configuration parameters for calibration. Like path to scripts to run the lisflood model (or any other hydrological model). 
So you need to have access and execution rights to the model you want to calibrate.
 
The easy way is to make a copy of the template integration/settings.txt (e.g. my_settings.txt) and edit it according your requirements.

Below you find an example of settings file:

```ini 
[Main]
forcing_start = 02/01/1990 06:00  # Starting of Meteo Forcings Precipitation Evapotranspiration TAvg 
forcing_end = 31/12/2017 06:00  # Ending of Meteo Forcings Precipitation Evapotranspiration TAvg
timestep = 360  # Timestep of the calibration run (typicall 6-hourly or daily)
prerun_start = 02/01/1990 06:00  # When to start the prerun
prerun_end = 31/12/2017 06:00 # When to end the prerun
prerun_timestep = 1440  # Timestep of the prerun (can be different than calibration to accelerate the process)
fast_debug = 0  # Flag to set to 1 for quicker debugging
min_obs_years = 3.5  # Minimum number of years of observation required to calibrate the station

[Stations]
stations_data = STATIONS/stations_data.csv  # Path to the stations CSV file
stations_links = STATIONS/stations_links.csv  # Path to the stations hydrological dependencies file
observed_discharges = OBS  # Path to the observations

[Path]
param_ranges = TEMPLATES/param_ranges.csv  # Path to the parameters ranges file
subcatchment_path = CATCHMENTS_DIR  # Root of the catchments
summary_path = SUMMARY_DIR  # Where to put the summary files of the calibration (Global statistics)

[Templates]
LISFLOODSettings = TEMPLATES/settings_lisflood.xml  # Path to the LISFLOOD settings template

[DEAP]
numCPUs = NCPUS  # Number of processes to use per catchment
min_gen = 6  # Minimum number of generation to run
max_gen = 16  # Maximum number of generation to run
mu = 18  # Initial population  
lambda_ = 36  # Size of generation of offsprings 
pop = 72  # Population
gen_offset = 3  # Stopping criteria: check efficiency vs that from 3 generations before
effmax_tol = 0.003  # Stopping criteria: if efficiency difference lower than tolerance, stop calibration
```

## Format of station and observations files

### stations.csv (metadata)

```csv
ID,Provider_1,ProvID_1,Provider_2,ProvID_2,Stationnam,RiverName,RiverBasin,Country,CountryNam,Continent,HydroRegio,DrainArPro,DrainArLDD,YProvided,XProvided,YCorrected,XCorrected,Suitable,ValidGLOFAS,Legend_for,Comments,StartDate,EndDate,AddedDate,Lake,Reservoir,Lastupdate
G0001,GRDC,2903430,NA,NA,Stolb,Lena,Lena,RU,Russian Federation,Asia,512,2460000,2443690,72.37,126.8,72.25,126.75,1,1, , ,1/1/1951,12/31/2002, ,0,0, 
G0002,GRDC,2999910,NA,NA,7.5Km D/S Of Mouth Of River Pur,Olenek,Olenek,RU,Russian Federation,Asia,512,198000,218622,72.12,123.22,72.15,123.35,1,1, , ,1/1/1953,12/31/2011, ,0,0, 
G0003,GRDC,2999150,NA,NA,Saskylakh,Anabar,Anabar,RU,Russian Federation,Asia,512,78800,86783,71.98,114.057,71.95,114.05,1,1, , ,6/1/1954,11/22/2011, ,0,0, 
G0004,GRDC,2999850,NA,NA,Khatanga,Khatanga,Khatanga,RU,Russian Federation,Asia,512,275000,285113,71.98,102.45,71.95,102.15,1,1, , ,6/7/1971,9/30/1991, ,0,0, 
```
### observations.csv (observed data per reported station in QGis.csv, column oriented)

```csv
DATE,G0001,G0002,G0003,G0004
1/1/1914,,,41.5,51.5
2/1/1914,,,43.5,51.8
3/1/1914,,,44.1,50.1
4/1/1914,,30.1,42.7,51.8
...
```

# Summary

1. Create a settings template for LISFLOOD (you can start from there: https://github.com/ec-jrc/lisflood-code/blob/master/src/lisfloodSettings_reference.xml).
2. Prepare static maps (dem, landuse etc.) and NetCDF forcing data (such as ERA5 dataset).
3. Prepare the stations csv file. This is a file containing the stations metadata.
4. Prepare the observations csv file. This file contains observed discharge data for each station.
5. Make a copy of integration/settings.txt and edit according your system.

