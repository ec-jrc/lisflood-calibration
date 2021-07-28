#!/usr/bin/env python3

from osgeo import gdal
import pandas  as pd
import xarray as xr
import os
import sys
import numpy as np
from configparser import ConfigParser # Python 3.8

# import binFileIO
from liscal.pcr_utils import pcrasterCommand, getPCrasterPath
# import liscal.raster_operations as ro
# import domain as dom
# import pyproj
import dask
from dask.diagnostics import ResourceProfiler, Profiler, CacheProfiler, visualize
import pcraster as pcr


def clipPCRasterMap(inputFilename, outputFilename, mask):

  if outputFilename.find("outlets") == -1 and outputFilename.find("res.") == -1 and outputFilename.find("lakes") == -1:
    # load the small mask we use to clip the input file with
    pcr.setclone(mask)

    # Use PCRasters pcrcalc with ifthen to generate a map with missing values (mv) where a condition is not met (value = 0)
    maskSmall = pcr.ifthen(mask, inputFilename)
    pcr.report(maskSmall, outputFilename)

    # Resample with PCRaster, which cuts maps where it's set to mv
    # Not to be confused with the resampling as gdal warp does
    pcrasterCommand(resample + " -c 0 F0 F1", {"F0": outputFilename, "F1": outputFilename+'.tmp'})
    os.system('mv ' + outputFilename + '.tmp ' + outputFilename)
  if outputFilename.find("ldd") > -1:
    ldd = pcr.readmap(outputFilename)
    pcr.setclone(outputFilename)
    lddr = pcr.lddrepair(ldd)
    pcr.report(lddr, outputFilename)
  # print(outputFilename)
  # print(pcr.pcr2numpy(pcr.readmap(outputFilename), np.nan))
  return outputFilename


def findMainVar(ds):
  # new safer code that doesn't rely on a specific variable order in netCDF file (R.COUGHLAN & D.DECREMER)
  # Attempt at checking if input files are not in the format we expect
  varNames = [[jt[0] for jt in ds.variables.items()][it] for it in range(len(ds.variables.items()))]
  targets = list()
  for it in varNames:
    if it not in ['x', 'y', 'lat', 'lon', 'laea', 'lambert_azimuthal_equal_area', 'time', 'crs', 'wgs_1984']:
      targets.append(it)
  # Return warning if we have more than 1 non-coordinate-related variable (i.e. x, y, laea, time) OR if the last variable in the netCDF file is not the variable to get data for
  if len(targets) > 1:
    raise Exception("Too many core variables found in netCDF file {}\n".format(ds))
  else:
    value = targets[0]

  return value


########################################################################
#   Read settings file
########################################################################
iniFile = os.path.normpath(sys.argv[1])
file_CatchmentsToProcess = os.path.normpath(sys.argv[2])
print("=================== START ===================")
print(">> Reading settings file ("+sys.argv[1]+")...")

parser = ConfigParser() # python 3.8
parser.read(iniFile)

path_maps = parser.get("Path", "input_maps_path")

subcatchment_path = parser.get('Path','subcatchment_path')

config = {}
for execname in ["pcrcalc","map2asc","asc2map","col2map","map2col","mapattr","resample"]:
    config[execname] = execname

pcrcalc = config["pcrcalc"]
col2map = config["col2map"]
map2col = config["map2col"]
resample = config["resample"]

file_CatchmentsToProcess = os.path.normpath(sys.argv[2])
stations_data_path = parser.get("Stations", "stations_data")

print(">> Reading stations_data file...")
stationdata = pd.read_csv(stations_data_path, sep=",", index_col='ObsID')
stationdata_sorted = stationdata.sort_values(by=['DrainingArea.km2.LDD'],ascending=True)
CatchmentsToProcess = pd.read_csv(file_CatchmentsToProcess,sep=",",header=None)

prof = Profiler()
rprof = ResourceProfiler(dt=0.25)
cprof = CacheProfiler() #metric=nbytes)
prof.register()
rprof.register()
cprof.register()

with dask.config.set(scheduler='threads'): #, pool=ThreadPool(cfg.ncpus)):  # [distributed, multiprocessing, processes, single-threaded, sync, synchronous, threading, threads]

    for index, row in stationdata_sorted.iterrows():
        Series = CatchmentsToProcess[0]
        if len(Series[Series==index]) == 0: # Only process catchments whose ObsID is in the CatchmentsToProcess.txt file
            continue
        print("=================== "+str(index)+" ====================")
        print(">> Starting map subsetting for catchment "+str(index)+", size "+str(row['DrainingArea.km2.LDD'])+" pixels...")

        #t = time.time()
        
        path_subcatch = os.path.join(subcatchment_path,str(index))
        path_subcatch_maps = os.path.join(path_subcatch,'maps')

        # Cut bbox from ALL static maps and forcings for subcatchment
        maskpcr = os.path.join(path_subcatch,'maps','mask.map')

        if os.path.isfile(maskpcr):
            print('maskmap',maskpcr)
            maskmap = pcr.readmap(maskpcr)
        else:
            print('wrong input mask file')
            sys.exit(1)

        masknp = pcr.pcr2numpy(maskmap,False)
        mask_filter = np.where(masknp)
        x_min = np.min(mask_filter[1])
        x_max = np.max(mask_filter[1])
        y_min = np.min(mask_filter[0])
        y_max = np.max(mask_filter[0])
        
        # Enter in maps dir and walk through subfolders
        for root,dirs,files in os.walk(path_maps, topdown=False, followlinks=True):
            for afile in files:
                filenc = os.path.join(root, afile)
                if filenc.find("bak") > -1:
                  continue
                ext = filenc[-4:][filenc[-4:].find("."):]
                fileout = os.path.join(path_subcatch_maps, afile)
                if os.path.isfile(fileout) and os.path.getsize(fileout) > 0:
                    print("skipping already existing %s" % fileout)
                    continue
                else:

                    print('creating...',fileout)
                    if ext == ".map": # and (not os.path.isfile(filenc.replace(ext, ".nc")) or os.path.getsize(filenc.replace(ext, ".nc")) == 0):
                        pcr.setclone(maskpcr)
                        clipPCRasterMap(filenc, fileout, maskpcr)
                    elif ext == ".nc":
                        try:
                            ds = xr.open_dataset(filenc, chunks = {'time': 'auto'})
                        except:
                            ds = xr.open_dataset(filenc, decode_coords=True)

                        if 'lon' in ds.coords and 'lat' in ds.coords:
                            ds_out = ds.isel(lat=range(y_min, y_max + 1), lon=range(x_min, x_max + 1))
                        elif 'x' in ds.coords and 'y' in ds.coords:
                            ds_out = ds.isel(y=range(y_min, y_max + 1), x=range(x_min, x_max + 1))
                        else:
                            raise Exception('Could not find lat/lon or x/y coordinates in dataset:\n {}'.format(ds))

                        # DD: Domenico's fix to avoid bug of having different grid point values in cut map than the original map
                        var = findMainVar(ds)
                        ds_out.to_netcdf(fileout, encoding={var: {'zlib': False}})

                        # Close memory access
                        ds.close()
                        ds_out.close()
                    else:
                        os.system("cp " + filenc + " " + fileout)

        # # Transform into numpy binary
        # if ver.find('3.') > -1:
        #     binFileIO.main(iniFile, file_CatchmentsToProcess)
        print('finito...')
