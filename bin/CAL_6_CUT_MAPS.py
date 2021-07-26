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


def getMapAttributes(res, projection, domain, rasterOrigin=None, nxny=None):
  if rasterOrigin is None:
    if domain == "Eric":
      rasterOrigin = (2500000, 5430000)
    elif domain == "XDOM":
      rasterOrigin = (2500000, 5500000)
    elif domain == "oldDom":
      rasterOrigin = (-1700000, 1360000)
    elif domain == "PRONEWS":
      rasterOrigin = (5035000, 2260000)
    elif domain == "SEEMHEWS":
      rasterOrigin = (4790000, 2550000)
  if nxny is None:
    if domain == "Eric":
      nxny = (1000, 936)
    elif domain == "XDOM":
      nxny = (1000, 950)
    elif domain == "oldDom":
      nxny = (3570, 2980)
    elif domain == "PRONEWS":
      nxny = (50, 76)
    elif domain == "SEEMHEWS":
      nxny = (61, 51)
  mapAttributes = {
    'projection':projection,
    'rasterOrigin':rasterOrigin,
    'pixelWidth':5000,
    'pixelHeight':-5000,
    'pixelLength':5000,
    'nx':nxny[0],
    'ny':nxny[1]
  }
  if projection == "EPSG:3035":
    if domain == "Eric":
      mapAttributes['rasterOrigin'] = rasterOrigin
    if res == 1:
      # 1km grid
      mapAttributes['pixelWidth'] = 5000 / 5
      mapAttributes['pixelHeight'] = -5000 / 5
      mapAttributes['pixelLength'] = 5000 / 5
      mapAttributes['nx'] = nxny[0] * 5
      mapAttributes['ny'] = nxny[1] * 5
      if domain == "Eric":
        mapAttributes['ny'] = nxny[1] * 5 # This is for Eric area, which has a few latitudes less in the North
    elif res == 5:
      # 5km grid
      mapAttributes['pixelWidth'] = 5000
      mapAttributes['pixelHeight'] = -5000
      mapAttributes['pixelLength'] = 5000
      mapAttributes['nx'] = nxny[0]
      mapAttributes['ny'] = nxny[1]
      if domain == "Eric":
        mapAttributes['ny'] = nxny[1]
  elif projection == "EPSG:4326":
    mapAttributes['rasterOrigin'] = (15, 45)
    if res == 1:
      # 1km grid
      mapAttributes['pixelWidth'] = 0.05 / 5
      mapAttributes['pixelHeight'] = -0.05 / 5
      mapAttributes['pixelLength'] = 5000 / 5
      mapAttributes['nx'] = 200 * 5
      mapAttributes['ny'] = 200 * 5
    elif res == 5:
      # 5km grid
      mapAttributes['pixelWidth'] = 0.05
      mapAttributes['pixelHeight'] = -0.05
      mapAttributes['pixelLength'] = 5000
      mapAttributes['nx'] = 200
      mapAttributes['ny'] = 200
  elif projection == 'oldDom':
    mapAttributes['projection'] = "+proj=laea +lat_0=48 +lon_0=9 +x_0=0 +y_0=0 +a=6378388 +b=6378388 +units=m +no_defs"
    mapAttributes['rasterOrigin'] = rasterOrigin
    mapAttributes['pixelWidth'] = 1000
    mapAttributes['pixelHeight'] = -1000
    mapAttributes['pixelLength'] = 1000
    mapAttributes['nx'] = nxny[0]
    mapAttributes['ny'] = nxny[1]
  else:
    raise("Unknown projection. Please define your chosen projection in the getMapDimensions function.")
    sys.exit(1)
  return mapAttributes


def setProj(dsOut, projection):
  # Find the name of the main variable
  newVar = findMainVar(dsOut)
  # Set the ESRI string
  # For regular latlon (WGS84)
  if projection == "EPSG:4326":
    dsOut.variables[newVar].attrs['esri_pe_string']='GEOGCS[\"GCS_WGS_1984\",DATUM[\"D_WGS_1984\",SPHEROID[\"WGS_1984\",6378137,298.257223563]],PRIMEM[\"Greenwich\",0],UNIT[\"Degree\",0.0174532925199433]]"'
  # For EPSG:3035
  else:
    try:
      dsOut.variables[newVar].attrs["esri_pe_string"] = "PROJCS[\"ETRS_1989_LAEA\",GEOGCS[\"GCS_ETRS_1989\",DATUM[\"D_ETRS_1989\",SPHEROID[\"GRS_1980\",6378137.0,298.257222101]],PRIMEM[\"Greenwich\",0.0],UNIT[\"Degree\",0.0174532925199433]],PROJECTION[\"Lambert_Azimuthal_Equal_Area\"],PARAMETER[\"false_easting\",4321000.0],PARAMETER[\"false_northing\",3210000.0],PARAMETER[\"central_meridian\",10.0],PARAMETER[\"latitude_of_origin\",52.0],UNIT[\"Meter\",1.0]]"
    except KeyError:
      newVar = afile[:afile.find(".")]
      dsOut.variables[newVar].attrs["esri_pe_string"] = "PROJCS[\"ETRS_1989_LAEA\",GEOGCS[\"GCS_ETRS_1989\",DATUM[\"D_ETRS_1989\",SPHEROID[\"GRS_1980\",6378137.0,298.257222101]],PRIMEM[\"Greenwich\",0.0],UNIT[\"Degree\",0.0174532925199433]],PROJECTION[\"Lambert_Azimuthal_Equal_Area\"],PARAMETER[\"false_easting\",4321000.0],PARAMETER[\"false_northing\",3210000.0],PARAMETER[\"central_meridian\",10.0],PARAMETER[\"latitude_of_origin\",52.0],UNIT[\"Meter\",1.0]]"
    dsOut.variables[newVar].attrs["grid_mapping"] = "lambert_azimuthal_equal_area"
  return dsOut


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


def createOutputFile(outputFile, ds, smallArray, dims):
  try:
    tArray = ds.variables['time'].data
    yArray = np.array(ds.variables['y'].data[dims['y1']:dims['y2']+1])
    xArray = np.array(ds.variables['x'].data[dims['x1']:dims['x2']+1])
    dsOut = xr.Dataset({os.path.basename(outputFile).replace('.nc', ""): smallArray},
                       coords={'time': tArray, 'ySmall': yArray, 'xSmall': xArray})
  except:
    yArray = np.array(ds.variables['y'].data[dims['y1']:dims['y2'] + 1])
    xArray = np.array(ds.variables['x'].data[dims['x1']:dims['x2'] + 1])
    dsOut = xr.Dataset({os.path.basename(outputFile).replace('.nc', ''): smallArray},
                       coords={'ySmall': yArray, 'xSmall': xArray})
  return dsOut



def sliceArray(inArray, dims):
  if len(inArray.shape) == 3:
    outArray = inArray[:, dims['y1']:dims['y2']+1, dims['x1']:dims['x2']+1]
  else:
    outArray = inArray[dims['y1']:dims['y2']+1, dims['x1']:dims['x2']+1]
  return outArray



def sliceArrayXapply(inArray, dims):
  return xr.apply_ufunc(sliceArray, inArray,
                        input_core_dims=[['y', 'x']],
                        output_core_dims=[['ySmall', 'xSmall']],
                        output_sizes={'ySmall': dims['y2']-dims['y1']+1, 'xSmall': dims['x2']-dims['x1']+1},
                        dask='parallelized',
                        output_dtypes=[float],
                        kwargs={'dims': dims})


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
