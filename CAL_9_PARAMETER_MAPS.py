# -*- coding: utf-8 -*-
import os
import sys
import pandas as pd
from datetime import datetime
import numpy as np
ver = sys.version
ver = ver[:ver.find('(')-1]
if ver.find('3.') > -1:
  from configparser import ConfigParser # Python 3.8
else:
  from ConfigParser import SafeConfigParser # Python 2.7-15
import HydroStats
import subprocess
import xarray as xr
import gdal


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


def array2raster(outputFile, mapAttributes, array, outputFormat=None, outputType=None, mv=None):
  if mv is None:
    mv = np.nan
    if outputType is None:
      outputType = gdal.GDT_Float32
    if outputFormat is None:
      outputFormat = 'netCDF'
  cols = array.shape[1]
  rows = array.shape[0]
  originX = mapAttributes['rasterOrigin'][0]
  originY = mapAttributes['rasterOrigin'][1]
  driver = gdal.GetDriverByName(outputFormat)
  if outputFile.find("ldd") > -1:
    if outputFormat == "PCRaster":
      outRaster = driver.Create(outputFile, cols, rows, 1, gdal.GDT_Byte, ["PCRASTER_VALUESCALE=VS_LDD"])
    else:
      outRaster = driver.Create(outputFile, cols, rows, 1, gdal.GDT_Byte)
    array = array.astype(np.uint8)
  else:
    try:
      checkint = str(array.dtype)
    except:
      checkint = "x"
    if checkint in ["int16", "int32", "int64", "uint16", "uint32", "uint64"]:
      if outputFormat == "PCRaster":
        outRaster = driver.Create(outputFile, cols, rows, 1, gdal.GDT_Int32, ["PCRASTER_VALUESCALE=VS_NOMINAL"])
      else:
        outRaster = driver.Create(outputFile, cols, rows, 1, gdal.GDT_Int32)
      array = array.astype(checkint)
    elif checkint in ["int8", "uint8"]:
      if outputFormat == "PCRaster":
        outRaster = driver.Create(outputFile, cols, rows, 1, gdal.GDT_Int8, ["PCRASTER_VALUESCALE=VS_NOMINAL"])
      else:
        outRaster = driver.Create(outputFile, cols, rows, 1, gdal.GDT_Int32)
      array = array.astype(checkint)
    else:
      if np.nanmax(array) > 1e6 or np.nanmin(array) < 1e-6 or outputType == gdal.GDT_Float64:
        if outputFormat == "PCRaster":
          outRaster = driver.Create(outputFile, cols, rows, 1, gdal.GDT_Float64, ["PCRASTER_VALUESCALE=VS_SCALAR"])
        else:
          outRaster = driver.Create(outputFile, cols, rows, 1, gdal.GDT_Float64)
      else:
        if outputFormat == "PCRaster":
          outRaster = driver.Create(outputFile, cols, rows, 1, gdal.GDT_Float32, ["PCRASTER_VALUESCALE=VS_SCALAR"])
        else:
          outRaster = driver.Create(outputFile, cols, rows, 1, gdal.GDT_Float32)
  outband = outRaster.GetRasterBand(1)
  outband.WriteArray(array)
  outband.SetNoDataValue(mv)
  outRasterSRS = osr.SpatialReference()
  if mapAttributes['projection'] == "EPSG:3035":
    outRasterSRS.ImportFromEPSG(3035)
  elif mapAttributes['projection'] == "EPSG:4326":
    outRasterSRS.ImportFromEPSG(4326)
  outRaster.SetProjection(outRasterSRS.ExportToWkt())
  outRaster.SetGeoTransform((originX, mapAttributes['pixelWidth'], 0, originY, 0, mapAttributes['pixelHeight']))
  outband.FlushCache()
  return outputFile


def any2netCDF(inputFile, mapAttributes, mv=None, mvOverride=None, flipUD=None, varName=None, outputType=None):
  if mvOverride is None:
    mvOverride = False
  if flipUD is None:
    flipUD = False
  if varName is None:
    varName = inputFile[:inputFile.find(".")]
  if outputType is None:
    outputType = gdal.GDT_Float32
  inputFilePath, ext = os.path.splitext(inputFile)
  outputFile = inputFile.replace(ext, ".nc")
  tempFile = outputFile.replace(".nc", "_tmp.nc")
  if outputFile == inputFile:
    sameFile = True
  else:
    sameFile = False
  # Get some basic dimension and properties
  ds = gdal.Open(inputFile)
  cols = ds.RasterXSize
  rows = ds.RasterYSize
  geoinfo = ds.GetGeoTransform()
  minx = geoinfo[0]
  miny = geoinfo[3] + (rows * geoinfo[5])
  maxx = geoinfo[0] + (cols * geoinfo[1])
  maxy = geoinfo[3]
  # Convert file to netCDF - simply with gdal defaults
  if not sameFile:
    # Replace undefined values
    values = np.array(ds.GetRasterBand(1).ReadAsArray()).astype(np.float64)
    if mvOverride:
      if mv is not None:
        values[abs(values - np.min(values)) < 1e-20] = mv
        rmiss = mv
      else:
        if os.path.basename(inputFile).find('ldd') > -1:
          values[abs(values - 255) < 1e-20] = 255
          rmiss = 255
        else:
          if str(values.dtype) in ['uint8', 'int8']:
            values[abs(values - 255) < 1e-20] = 255
            rmiss = 255
          elif str(values.dtype) in ['uint16', 'uint32', 'uint64', 'int16', 'int32', 'int64']:
            values[abs(values - 0) < 1e-20] = -9999
            rmiss = -9999
          else:
            values[abs(values - np.min(values)) < 1e-20] = np.nan
            rmiss = np.nan
    else:
      if mv is None:
        mv = findMv(inputFile)
      rmiss = mv
    if np.nanmax(values) > 1e6 or np.nanmin(values) < 1e-6 or outputType == gdal.GDT_Float64:
      outputType = gdal.GDT_Float64
    ds.GetRasterBand(1).DataType
    ds.GetRasterBand(1).SetNoDataValue(rmiss)
    ds.GetRasterBand(1).WriteArray(values)
    nds = gdal.Translate(tempFile, ds,
      format = "netCDF",
      outputSRS = mapAttributes['projection'],
      xRes = mapAttributes['pixelLength'],
      yRes = mapAttributes['pixelLength'],
      noData = rmiss,
      outputType = outputType
    )
    nds = None
    ds = None
    print(inputFile)
  else:
    ds = None
    os.system("mv " + outputFile + " " + tempFile)
  # Now we need to correct the file for Lisflood compatibility
  # i.e. reorder and rename dimension variables, flip maps along the equator
  ds = netCDF4.Dataset(tempFile)
  nds = netCDF4.Dataset(outputFile, mode='w', format='NETCDF3_CLASSIC')
  # Get main variable's name
  varNames = [[jt[0] for jt in ds.variables.items()][it] for it in range(len(ds.variables.items()))] # Python 3.8
  targets = list()
  for it in varNames:
    if not it == 'lon' and not it == 'lons' and not it == 'longitude' and not it == 'x' and \
       not it == 'lat' and not it == 'lats' and not it == 'latitude' and not it == 'y' and \
       not it == 'laea' and not it == "lambert_azimuthal_equal_area" and \
       not it == 'time':
      targets.append(it)
  # Return warning if we have no or more than 1 non-coordinate-related variable (i.e. x, y, laea, time)
  if len(targets) > 1:
    warnings.warn("Wrong number of variables found in netCDF file %s" % outputFile)
  else:
    mainVarName = targets[0]
    destVarName = inputFilePath.split("/")[-1]
    print(destVarName)
  # Encode dimension variables
  if "lon" in varNames and "lat" in varNames:
    xDim = "longitude"
    yDim = "latitude"
  elif "longitude" in varNames and "latitude" in varNames:
    xDim = "lon"
    yDim = "lat"
  elif "x" in varNames and "y" in varNames:
    xDim = "x"
    yDim = "y"
  if "time" in varNames:
    if len(ds.variables['time'][:]) > 1: # If there is only one timestep, we will ignore it
      tDim = "time"
  nds.createDimension(xDim,len(ds.variables[xDim][:]))
  nds.createDimension(yDim,len(ds.variables[yDim][:]))
  lon = nds.createVariable(xDim, ds.variables[xDim].datatype,(xDim,), fill_value = None)
  lon.setncatts({k: ds.variables[xDim].getncattr(k) for k in ds.variables[xDim].ncattrs()}) #copy lat attrs
  lat = nds.createVariable(yDim, ds.variables[yDim].datatype,(yDim,), fill_value = None)
  lat.setncatts({k: ds.variables[yDim].getncattr(k) for k in ds.variables[yDim].ncattrs()}) #copy lat attrs
  if 'tDim' in locals():
    nds.createDimension(tDim,len(ds.variables[tDim][:]))
    tim = nds.createVariable(tDim, ds.variables[tDim].datatype,(tDim,), fill_value = None)
    tim.setncatts({k: ds.variables[tDim].getncattr(k) for k in ds.variables[tDim].ncattrs()}) #copy lat attrs
  # Set the CRS properly
  try:
    crs = nds.createVariable('lambert_azimuthal_equal_area', ds.variables['lambert_azimuthal_equal_area'].datatype)
    crs.setncatts({k: ds.variables['lambert_azimuthal_equal_area'].getncattr(k) for k in ds.variables['lambert_azimuthal_equal_area'].ncattrs()}) #copy crs attrs
  except:
    crs = nds.createVariable('lambert_azimuthal_equal_area', ds.variables['laea'].datatype)
    crs.setncatts({k: ds.variables['lambert_azimuthal_equal_area'].getncattr(k) for k in ds.variables['laea'].ncattrs()})  # copy crs attrs
  # Create variables
  if 'tDim' in locals():
    var = nds.createVariable(str(destVarName).replace(".nc", ""), ds.variables[getVarName(varNames, mainVarName)].datatype, (tDim, yDim, xDim), fill_value=ds.variables[getVarName(varNames, mainVarName)]._FillValue)
  else:
    var = nds.createVariable(str(destVarName).replace(".nc", ""), ds.variables[getVarName(varNames, mainVarName)].datatype,(yDim,xDim),fill_value = ds.variables[getVarName(varNames, mainVarName)]._FillValue)
  var.long_name = str(destVarName)
  var.missing_value = ds.variables[getVarName(varNames, mainVarName)]._FillValue
  var.grid_mapping = 'lambert_azimuthal_equal_area'
  # Fill in new values and flip the map around the equator (except if the input file was already a netCDF file)
  xl = minx + mapAttributes['pixelLength'] / 2
  xr = xl + cols * mapAttributes['pixelLength']
  yu = maxy - mapAttributes['pixelLength'] / 2
  yd = yu - rows * mapAttributes['pixelLength']
  lats = np.linspace(yu, yd, rows, endpoint=False)
  lons = np.linspace(xl, xr, cols, endpoint=False)
  lon[:] = lons
  lat[:] = lats
  # Test needed in case we have boolean or ldd maps as they can't be casted to int8 in netCDF
  if str(var.dtype) == "uint8":
    var = np.int32(var)
  if flipUD or sameFile:
    if not 'tDim' in locals():
      var[:, :] = np.flipud(ds.variables[getVarName(varNames, mainVarName)][:,:])
    else:
      var[:, :, :] = np.array(ds.variables[getVarName(varNames, mainVarName)][:, ::-1, :])
  else:
    if not 'tDim' in locals():
      var[:, :] = ds.variables[getVarName(varNames, mainVarName)][:, :]
    else:
      var[:, :, :] = np.array(ds.variables[getVarName(varNames, mainVarName)][:, :, :])
  # Close files and save to disk
  ds.close()
  nds.close()
  # Clean up
  os.system("rm -rf " + tempFile)
  return outputFile


def findMainVar(ds):
  try:
    # new safer code that doesn't rely on a specific variable order in netCDF file (R.COUGHLAN & D.DECREMER)
    # Attempt at checking if input files are not in the format we expect
    varNames = [[jt[0] for jt in ds.variables.items()][it] for it in range(len(ds.variables.items()))]
    targets = list()
    for it in varNames:
      if not it == 'x' and not it == 'y' and not it == 'laea' and not it == "lambert_azimuthal_equal_area" and not it == 'time':
        targets.append(it)
    # Return warning if we have more than 1 non-coordinate-related variable (i.e. x, y, laea, time) OR if the last variable in the netCDF file is not the variable to get data for
    if len(targets) > 1:
      print("Wrong number of variables found in netCDF file\n")
    else:
      value = targets[0]
  except:
    # original code
    value = ds.variables.items()[-1][0]
  return value


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
      newVar = dsOut[:dsOut.find(".")]
      dsOut.variables[newVar].attrs["esri_pe_string"] = "PROJCS[\"ETRS_1989_LAEA\",GEOGCS[\"GCS_ETRS_1989\",DATUM[\"D_ETRS_1989\",SPHEROID[\"GRS_1980\",6378137.0,298.257222101]],PRIMEM[\"Greenwich\",0.0],UNIT[\"Degree\",0.0174532925199433]],PROJECTION[\"Lambert_Azimuthal_Equal_Area\"],PARAMETER[\"false_easting\",4321000.0],PARAMETER[\"false_northing\",3210000.0],PARAMETER[\"central_meridian\",10.0],PARAMETER[\"latitude_of_origin\",52.0],UNIT[\"Meter\",1.0]]"
  return dsOut


def setCatchmentValue(inArray, interstationsData, catchment, paramvalue):
  outArray = np.where(interstationsData == int(catchment), paramvalue, inArray)
  dims = tuple([inArray.sizes.mapping['y'], inArray.sizes.mapping['x']])
  yArray = inArray['y']
  xArray = inArray['x']
  dsOut = xr.DataArray(outArray, coords={"y": yArray, "x": xArray}, dims=["y", "x"])
  return dsOut


def setParamValueInMap(inArray, interstationsData, catchment, paramvalue):
  return xr.apply_ufunc(setCatchmentValue, inArray, interstationsData, catchment, paramvalue, dask='parallelized')


def createOutputFile(outputFile, ds, largeArray):
  try:
    dims = tuple([ds.sizes.mapping['time'], ds.sizes.mapping['y'], ds.sizes.mapping['x']])
    tArray = ds.variables['time'].data
    yArray = np.array([ds.variables['y'].data[0] - res*(i-1) for i in range(-1, dims[1]-1)])
    xArray = np.array([ds.variables['x'].data[0] + res*(i-1) for i in range(-1, dims[2]-1)])
    dsOut = xr.Dataset({os.path.basename(outputFile).replace(".nc", ""): largeArray}, coords={"time": tArray, "y": yArray, "x": xArray})
  except:
    dims =  tuple([ds.sizes.mapping['y'], ds.sizes.mapping['x']])
    yArray = np.array([ds.variables['y'].data[0] - res*(i-1) for i in range(-1, dims[0]-1)])
    xArray = np.array([ds.variables['x'].data[0] + res*(i-1) for i in range(-1, dims[1]-1)])
    dsOut = xr.Dataset({os.path.basename(outputFile).replace(".nc", ""): largeArray}, coords={"y": yArray, "x": xArray})
  return dsOut


def plotMap(mymap):
  from matplotlib import pyplot as plt
  plt.pcolor(mymap.x, mymap.y, mymap)
  plt.show()


if __name__=="__main__":



  # DD Code to blend the GwLoss and GwLoss=0 runs
  with open("/perm/rd/nedd/EFAS/efasCalib/src/GwLossStations.txt") as f:
    GwLossStations = [i.replace('\n', '').replace('\r', '') for i in f.readlines()]
  f.close()

  # DD Code to blend the GwLoss and GwLoss=0 runs
  with open("/perm/mo/mocm/proj/efas/hprot/data/hydro/EC_stations/soilmoisture_stations.txt") as f:
    soilMoistureStations = [i.replace('\n', '').replace('\r', '') for i in f.readlines()]
  f.close()

  ########################################################################
  #   Read settings file
  ########################################################################

  iniFile = os.path.normpath(sys.argv[1])
  print("=================== START ===================")
  print(">> Reading settings file ("+sys.argv[1]+")...")

  if ver.find('3.') > -1:
    parser = ConfigParser()  # python 3.8
  else:
    parser = SafeConfigParser()  # python 2.7-15
  parser.read(iniFile)

  path_temp = parser.get('Path', 'Temp')
  path_maps = os.path.join(parser.get('Path', 'CatchmentDataPath'),"maps")
  path_result = parser.get('Path', 'Result')

  ParamRangesPath = parser.get('Path','ParamRanges')
  SubCatchmentPath = parser.get('Path','SubCatchmentPath')

  ObservationsStart = datetime.strptime(parser.get('DEFAULT','ObservationsStart'),"%d/%m/%Y %H:%M")  # Start of forcing
  ObservationsEnd = datetime.strptime(parser.get('DEFAULT','ObservationsEnd'),"%d/%m/%Y %H:%M")  # Start of forcing
  ForcingStart = datetime.strptime(parser.get('DEFAULT','ForcingStart'),"%d/%m/%Y %H:%M")  # Start of forcing
  ForcingEnd = datetime.strptime(parser.get('DEFAULT','ForcingEnd'),"%d/%m/%Y %H:%M")  # Start of forcing

  Qtss_csv = parser.get('CSV', 'Qtss')
  Qgis_csv = parser.get('CSV', 'Qgis')

  pcraster_path = parser.get('Path', 'PCRHOME')

  upareamap = parser.get("Path", "upareaPath")

  # Interstations map
  interstation_regions_map = os.path.join(path_result,"interstation_regions.nc")
  try:
    dsInterstations = xr.open_dataset(interstation_regions_map, chunks={'time': 100})
  except:
    dsInterstations = xr.open_dataset(interstation_regions_map)
  interStationsVar = findMainVar(dsInterstations)
  interstationsData = dsInterstations[interStationsVar]

  ########################################################################
  #   Make stationdata array from the qgis csv
  ########################################################################

  # Reading station data
  print(">> Reading Qgis2.csv file...")
  stationdataFile = "Qgis2.csv"
  if os.path.exists(stationdataFile.replace(".csv", ".npy")) and os.path.getsize(stationdataFile.replace(".csv", ".npy")) > 0:
    stationdata = pd.DataFrame(np.load(stationdataFile.replace(".csv", ".npy"), allow_pickle=True))
    stationdata.index = np.load(stationdataFile.replace(".csv", "_index.npy"), allow_pickle=True)
    stationdata.columns = np.load(stationdataFile.replace(".csv", "_columns.npy"), allow_pickle=True)
  else:
    stationdata = pd.read_csv(os.path.join(path_result, stationdataFile), sep=",", index_col=0)
    np.save(stationdataFile.replace(".csv", ".npy"), stationdata)
    np.save(stationdataFile.replace(".csv", "_index.npy"), stationdata.index)
    np.save(stationdataFile.replace(".csv", "_columns.npy"), stationdata.columns.values)

  # Reading station observed discharge
  print(">> Reading ecQts.csv file...")
  if os.path.exists(Qtss_csv.replace(".csv", ".npy")) and os.path.getsize(Qtss_csv.replace(".csv", ".npy")) > 0:
    streamflow_data = pd.DataFrame(np.load(Qtss_csv.replace(".csv", ".npy"), allow_pickle=True))
    streamflow_datetimes = np.load(Qtss_csv.replace(".csv", "_dates.npy"), allow_pickle=True).astype('string_')
    streamflow_data.index = [datetime.strptime(i.decode('utf-8'), "%d/%m/%Y %H:%M") for i in streamflow_datetimes]
    streamflow_data.columns = np.load(Qtss_csv.replace(".csv", "_catchments.npy"), allow_pickle=True)
  else:
    streamflow_data = pd.read_csv(Qtss_csv, sep=",", index_col=0)
    streamflow_data.index = pd.date_range(start=ObservationsStart, end=ObservationsEnd, periods=len(streamflow_data))
    np.save(Qtss_csv.replace(".csv", ".npy"), streamflow_data)
    np.save(Qtss_csv.replace(".csv", "_dates.npy"), streamflow_data.index)
    np.save(Qtss_csv.replace(".csv", "_catchments.npy"), streamflow_data.columns.values)

  # Parameter Histories
  print(">> Searching for paramsHistories...")
  paramsHistoriesFile = "paramsHistories.npy"
  if not os.path.exists(paramsHistoriesFile) or os.path.getsize(paramsHistoriesFile) == 0:
    SubCatchmentPath = "/vol/floods/nedd/EFASCalib/backup/20200331GwLossZeroNoWateruse" # DD soil-moisture fix for EFAS 4.0
    cmd = "find -L " + SubCatchmentPath + " -type f -name \"paramsHistory.csv\" | sort"
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stdin=subprocess.PIPE, stderr=subprocess.STDOUT)
    paramsHistories1 = p.communicate()[0].split()
    p.wait()
    print(len(paramsHistories1))
    SubCatchmentPath = "/vol/floods/nedd/EFASCalib/backup/20200331GwLossNoWateruse" # DD soil-moisture fix for EFAS 4.0
    cmd = "find -L " + SubCatchmentPath + " -type f -name \"paramsHistory.csv\" | sort"
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stdin=subprocess.PIPE, stderr=subprocess.STDOUT)
    paramsHistories2 = p.communicate()[0].split()
    p.wait()
    print(len(paramsHistories2))
    paramsHistories = paramsHistories1 + paramsHistories2 #  DD soil-moisture fix for EFAS 4.0
    np.save(paramsHistoriesFile, paramsHistories)
  else:
    paramsHistories = np.load(paramsHistoriesFile, allow_pickle=True)

  ########################################################################
  #   Assign calibrated parameter values to maps
  ########################################################################

  # Load paramranges file
  ParamRanges = pd.read_csv(ParamRangesPath,sep=",",index_col=0)

  # DD Make maps of the KGE components and total volumetric error sae as well
  parValues = ParamRanges.index.values
  parValues = np.append(parValues, ["Kling Gupta Efficiency", "corr", "bias", "var", "sae"])
  newRow = pd.DataFrame({"MinValue":-1e20, "MaxValue":1, "DefaultValue":-9999}, index =["KGE"])
  ParamRanges = pd.concat([ParamRanges, newRow], sort=False)
  newRow = pd.DataFrame({"MinValue":-1, "MaxValue":1, "DefaultValue":-9999}, index =["Correlation"])
  ParamRanges = pd.concat([ParamRanges, newRow], sort=False)
  newRow = pd.DataFrame({"MinValue":-1e20, "MaxValue":1e20, "DefaultValue":-9999}, index =["Bias"])
  ParamRanges = pd.concat([ParamRanges, newRow], sort=False)
  newRow = pd.DataFrame({"MinValue":-1e20, "MaxValue":1e20, "DefaultValue":-9999}, index =["Spread"])
  ParamRanges = pd.concat([ParamRanges, newRow], sort=False)
  newRow = pd.DataFrame({"MinValue":0, "MaxValue":1e20, "DefaultValue":-9999}, index =["sae"])
  ParamRanges = pd.concat([ParamRanges, newRow], sort=False)
  newRow = pd.DataFrame({"MinValue": -1e20, "MaxValue": 1, "DefaultValue": -9999}, index=["KGELongTermRun"])
  ParamRanges = pd.concat([ParamRanges, newRow], sort=False)
  newRow = pd.DataFrame({"MinValue": -1e20, "MaxValue": 1, "DefaultValue": -9999}, index=["KGESkillScore"])
  ParamRanges = pd.concat([ParamRanges, newRow], sort=False)
  newRow = pd.DataFrame({"MinValue": -1e20, "MaxValue": 1, "DefaultValue": -9999}, index=["NSELongTermRun"])
  ParamRanges = pd.concat([ParamRanges, newRow], sort=False)

  # Assign calibrated parameter values to maps
  results = pd.DataFrame([], columns=["ObsID", "KGE", "Correlation", "Bias", "Spread", "sae", "KGELongTermRun", "KGESkillScore", "NSELongTermRun"])

  # Assign these to maps
  for iii in range(0, len(ParamRanges)):
    skip = False
    paramName = ParamRanges.index[iii]

    outArray = interstationsData.astype(np.float64)
    outArray[:] = ParamRanges['DefaultValue'][paramName]

    count_front = 0
    count_nofront = 0
    for h in paramsHistories:
      print(h)
      h = h.decode('utf-8')
      if h != "":
        catchment = os.path.basename(os.path.dirname(h))

        # Check CatchmentArea is valid
        catchmentArea = stationdata[stationdata.index == int(catchment)]['CatchmentArea'].values[0]
        if np.isnan(catchmentArea):
          print("WARNING: invalid catchment area => skipping catchment " + catchment)
          continue

        # DD Exception for soil moisture fix in EFAS4.0 - 2020/08/26
        path_subcatch = os.path.dirname(h) #join(SubCatchmentPath, catchment)
        if (h.find('GwLossZero') > -1 and os.path.basename(path_subcatch) in GwLossStations) or (h.find('GwLossZero') == -1 and os.path.basename(path_subcatch) not in GwLossStations):
          continue

        if os.path.isfile(os.path.join(path_subcatch, "pareto_front.csv")):
          count_front = count_front + 1
          pareto_front = pd.read_csv(os.path.join(path_subcatch, "pareto_front.csv"))

          # Skip incomplete 2nd calibration step
          if not os.path.exists(os.path.join(path_subcatch, "streamflow_simulated_best.csv")):
            continue

          # DD Attempt to use the streamflows for the long-term stats
          Qobs = streamflow_data[catchment]
          Qobs[Qobs < 0] = np.NaN
          Q_obs_Cal = Qobs.loc[ForcingStart:ForcingEnd]
          QobsMask = np.isfinite(Q_obs_Cal)
          Qsim = pd.read_csv(os.path.join(path_subcatch, "streamflow_simulated_best.csv"), sep=",", index_col=0, header=None)
          Qsim.index = pd.date_range(start=ForcingStart, end=ForcingEnd, periods=len(Qsim))
          Q_sim_Cal = Qsim.loc[ForcingStart:ForcingEnd]

          # DD use the paramsHistory.csv files
          pHistory = pd.read_csv(h, sep=",")[3:]
          # DD Exception for soil moisture fix in EFAS4.0 - 2020/08/26
          pHistory = pHistory[pHistory['PowerPrefFlow'] >= 0.5]
          # Keep only the best 10% of the runs for the selection of the parameters for the next generation
          pHistory = pHistory.sort_values(by="Kling Gupta Efficiency", ascending=False)
          pHistory = pHistory.head(int(round(len(pHistory) * 0.1)))
          n = len(pHistory)
          # Give ranking scores to corr
          pHistory = pHistory.sort_values(by="Correlation", ascending=False)
          pHistory["corrRank"] = [float(i + 1) / n for i, ii in enumerate(pHistory["Correlation"].values)]
          # Give ranking scores to sae
          pHistory = pHistory.sort_values(by="sae", ascending=True)
          pHistory["saeRank"] = [float(i + 1) / n for i, ii in enumerate(pHistory["sae"].values)]
          # Give ranking scores to KGE
          pHistory = pHistory.sort_values(by="Kling Gupta Efficiency", ascending=False)
          pHistory["KGERank"] = [float(i + 1) / n for i, ii in enumerate(pHistory["Kling Gupta Efficiency"].values)]
          # Give pareto score
          pHistory["paretoRank"] = pHistory["corrRank"].values * pHistory["saeRank"].values * pHistory["KGERank"].values
          pHistory = pHistory.sort_values(by="paretoRank", ascending=True)
          # Select the best pareto candidate
          bestParetoIndex = pHistory["paretoRank"].nsmallest(1).index
          pHistory = pHistory.loc[bestParetoIndex]

          # Determine parameter value
          if paramName == "KGE":
            paramvalue = np.array(pHistory["Kling Gupta Efficiency"])[0]
          elif paramName == "Correlation":
            paramvalue = np.array(pHistory["Correlation"])[0]
          elif paramName == "Bias":
            paramvalue = np.array(pHistory["Signal ratio (s/o) (Bias)"])[0]
          elif paramName == "Spread":
            paramvalue = np.array(pHistory["Noise ratio (s/o) (Spread)"])[0]
          elif paramName == "sae":
            paramvalue = np.array(pHistory["sae"])[0]
          elif paramName == "KGELongTermRun":
            paramvalue = HydroStats.KGE(s=np.array(Q_sim_Cal[QobsMask]), o=np.array(Q_obs_Cal[QobsMask]), warmup=0)
          elif paramName == "KGESkillScore":
            paramvalue = (HydroStats.KGE(s=np.array(Q_sim_Cal[QobsMask]), o=np.array(Q_obs_Cal[QobsMask]), warmup=0) - (1 - np.sqrt(2))) / (1 - (1 - np.sqrt(2)))
          elif paramName == "NSELongTermRun":
            paramvalue = HydroStats.NS(s=np.array(Q_sim_Cal[QobsMask]), o=np.array(Q_obs_Cal[QobsMask]), warmup=0)
          else:
            try:
              if os.path.basename(path_subcatch) not in soilMoistureStations:
                paramvalue = pareto_front["param_" + str(iii).zfill(2) + "_" + paramName][0]
              else:
                # DD Exception for soil moisture fix in EFAS4.0 - 2020/08/26
                paramvalue = pHistory[paramName].values[0]
            except KeyError:
              skip = True
          if not skip:
            # DD Exception added for the GwLoss zero runs. Force those to be 0 in the output
            if paramName.find("GwLoss") > -1:
              if os.path.basename(path_subcatch) not in GwLossStations:
                paramvalue = 0.0
                print("GwLoss 0-value set in " + path_subcatch)
                print()
            elif catchment == '851':
              if paramName.find("adjust_Normal_Flood") > -1:
                paramvalue = 0.8
              elif paramName.find("ReservoirRnormqMult") > -1:
                paramvalue = 1.0

            # Apply the value to the masked catchment
            outArray = setCatchmentValue(outArray, interstationsData, catchment, paramvalue)
        else: # If pareto_front.csv doesn't exist, put -1
          count_nofront = count_nofront+1;
          outArray = setCatchmentValue(outArray, interstationsData, catchment, -1.0)


    # Projection settings
    res = 5000
    projection = "EPSG:3035"
    domain = "XDOM"
    mapAttributes = getMapAttributes(res / 1000, projection, domain)

    # Flip outArray around horizontal
    outArray = outArray.sortby('y', ascending=False)
    filenc = os.path.join(path_result, "params_" + paramName + ".nc")
    fileTmp = array2raster(filenc, mapAttributes, np.array(outArray), outputFormat='netCDF', outputType=gdal.GDT_Float64)
    any2netCDF(fileTmp, mapAttributes, flipUD=False, varName=paramName, outputType=gdal.GDT_Float64)


  # Close memory access
  dsInterstations.close()

  print("---------------------------------------------")
  print("Number of catchments with pareto_front.csv: "+str(count_front)+"!")
  print("Number of catchments with missing pareto_front.csv: "+str(count_nofront)+"!")
  print("---------------------------------------------")
  print("==================== END ====================")
