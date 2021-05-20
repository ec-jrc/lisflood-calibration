from osgeo import gdal
import pandas  as pd
import xarray as xr
import os
import sys
import numpy as np
ver = sys.version
ver = ver[:ver.find('(')-1]
if ver.find('3.') > -1:
  from configparser import ConfigParser # Python 3.8
else:
  from ConfigParser import SafeConfigParser # Python 2.7-15
import binFileIO
from pcrasterCommand import pcrasterCommand, getPCrasterPath
import rasterOps as ro
import domain as dom
import pyproj
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

if ver.find('3.') > -1:
  parser = ConfigParser() # python 3.8
else:
  parser = SafeConfigParser()  # python 2.7-15
parser.read(iniFile)

path_maps = parser.get("Path", "CatchmentDataPath") + "../"

CatchmentDataPath = parser.get('Path','CatchmentDataPath')
SubCatchmentPath = parser.get('Path','SubCatchmentPath')

pcraster_path = parser.get('Path', 'PCRHOME')

config = {}
for execname in ["pcrcalc","map2asc","asc2map","col2map","map2col","mapattr","resample"]:
    config[execname] = getPCrasterPath(pcraster_path,sys.argv[1],execname)

pcrcalc = config["pcrcalc"]
col2map = config["col2map"]
map2col = config["map2col"]
resample = config["resample"]

path_MeteoData = parser.get('Path', 'MeteoData')
path_result = parser.get('Path', 'Result')
switch_SubsetMeteoData = int(parser.get('DEFAULT', 'SubsetMeteoData'))
file_CatchmentsToProcess = os.path.normpath(sys.argv[2])

print(">> Reading Qmeta2.csv file...")
stationdata = pd.read_csv(os.path.join(path_result,"Qmeta2.csv"),sep=",",index_col=0)
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



def findMainVar(ds):
  try:
    # new safer code that doesn't rely on a specific variable order in netCDF file (R.COUGHLAN & D.DECREMER)
    # Attempt at checking if input files are not in the format we expect
    varNames = [[jt[0] for jt in ds.variables.items()][it] for it in range(len(ds.variables.items()))]
    print(varNames)
    targets = list()
    for it in varNames:
      if not it == 'x' and not it == 'y' and not it == 'laea' and not it == "lambert_azimuthal_equal_area" and not it == 'time' and not it == 'lon' and not it == 'lat':
        targets.append(it)
    # Return warning if we have more than 1 non-coordinate-related variable (i.e. x, y, laea, time) OR if the last variable in the netCDF file is not the variable to get data for
    if len(targets) > 1:
      print("Wrong number of variables found in netCDF file\n")
    else:
      value = targets[0]
  except:
    # original code
    value = [jt for jt in ds.variables.items()][-1][0]
  return value



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
        
        path_subcatch = os.path.join(SubCatchmentPath,str(index))
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
                            ds = xr.open_dataset(filenc, chunks = {'time': 1150}) #'auto'})
                        except:
                            ds = xr.open_dataset(filenc, decode_coords=True)
                        var = findMainVar(ds)

                        # DD original code which doesnt work for the evaporation maps (e0, et and es)
                        # if filenc.find("e0_hourly.nc") == -1 and filenc.find("es_hourly.nc") == -1 and filenc.find("et_hourly.nc") == -1 and filenc.find("e0.nc") == -1 and filenc.find("es.nc") == -1 and filenc.find("et.nc") == -1:
                        # For regular latlon (WGS84)
                        # ds.variables[var].attrs['esri_pe_string']='GEOGCS[\"GCS_WGS_1984\",DATUM[\"D_WGS_1984\",SPHEROID[\"WGS_1984\",6378137,298.257223563]],PRIMEM[\"Greenwich\",0],UNIT[\"Degree\",0.0174532925199433]]"'
                        # For EPSG:3035
                        ds.variables[var].attrs[
                          "esri_pe_string"] = "PROJCS[\"ETRS_1989_LAEA\",GEOGCS[\"GCS_ETRS_1989\",DATUM[\"D_ETRS_1989\",SPHEROID[\"GRS_1980\",6378137.0,298.257222101]],PRIMEM[\"Greenwich\",0.0],UNIT[\"Degree\",0.0174532925199433]],PROJECTION[\"Lambert_Azimuthal_Equal_Area\"],PARAMETER[\"false_easting\",4321000.0],PARAMETER[\"false_northing\",3210000.0],PARAMETER[\"central_meridian\",10.0],PARAMETER[\"latitude_of_origin\",52.0],UNIT[\"Meter\",1.0]]"
                        # ds.variables[var].attrs["grid_mapping"] = "lambert_azimuthal_equal_area"

                        # DD replaced to handle wrongly encoded files which have time var in it, but not the dim
                        # if 'time' in ds.variables:
                        if len(ds[var].shape) == 3:
                          dsOut = ds[var][:, y_min:y_max + 1, x_min:x_max + 1]
                          # t0 = pd.datetime.now()
                        else:
                          dsOut = ds[var][y_min:y_max + 1, x_min:x_max + 1]

                        # Fix to make maps zero-filled                        if len(ds[var].shape) == 3:
                        if len(ds[var].shape) < 3:
                          dsOut.variable.values[np.isnan(dsOut.variable.values)] = 0.0
                        #print(ds[var][y_min:y_max + 1, x_min:x_max + 1].values)
                        #dsOut.to_netcdf(fileout)
                        # DD: Domenico's fix to avoid bug of having different grid point values in cut map than the original map
                        dsOut.to_netcdf(fileout, encoding={var: {'zlib': False}})

                        # # DD Fix wrong slicing for evaporation maps e0, es and et
                        # # DD New fix with WKT2 standard, but isnt backward compatible for a single map genua3.nc
                        # else:
                        #   # Clip the map
                        #   outDims = {'x1': x_min, 'x2': x_max, 'y1': y_min, 'y2': y_max}
                        #   if len(ds[var].shape) < 3:
                        #     outArray = sliceArrayXapply(ds[var], outDims)
                        #   else:
                        #     outArray = sliceArrayXapply(ds[var], outDims)
                        #
                        #   # Create output file
                        #   dsOut = createOutputFile(fileout, ds, outArray, outDims)
                        #
                        #   # Change the name of the dimension variables
                        #   try:
                        #     dsOut = dsOut.rename({'ySmall': 'y', 'xSmall': 'x'})
                        #   except ValueError:
                        #     pass
                        #   outVar = findMainVar(dsOut)
                        #
                        #   # Add laea projection details
                        #   dsOut = setProj(dsOut, 'EPSG:3035')
                        #   # # origin = (dsOut['x'].values[0] - 2500, dsOut['y'].values[0] + 2500)
                        #   # origin = (ds['x'].values[x_min], ds['y'].values[y_max])
                        #   # nxny = dsOut[outVar].shape[::-1][:2]
                        #   # mapAttrClipDom = ro.getMapAttributes(5, 'EPSG:3035', 'XDOM', rasterOrigin=origin, nxny=nxny)
                        #   #
                        #   # Define domain object
                        #   # if mapAttrClipDom['projection'] == 'EPSG:3035':
                        #   #   projection = dom.Projection_LAEA('lambert_azimuthal_equal_area',
                        #   #                                    lat_0=52,
                        #   #                                    lon_0=10,
                        #   #                                    x_0=4321000,
                        #   #                                    y_0=3210000,
                        #   #                                    ellps='GRS80',
                        #   #                                    proj_keys='+units=m +no_defs')
                        #   # domain = dom.Domain_LAEA('EFAS Extended Domain 5km',
                        #   #                          projection,
                        #   #                          mapAttrClipDom['pixelLength'],
                        #   #                          mapAttrClipDom['pixelWidth'],
                        #   #                          mapAttrClipDom['rasterOrigin'][0],
                        #   #                          mapAttrClipDom['rasterOrigin'][1],
                        #   #                          mapAttrClipDom['nx'],
                        #   #                          mapAttrClipDom['ny'])
                        #   domain = dom.Domain_EFAS_Extended()
                        #   # create projection
                        #   proj = domain.projection
                        #   # crs = proj.xy_proj.crs #Corentins original
                        #   crs = pyproj.crs.CRS.from_epsg('3035')
                        #
                        #   dsOut[proj.name] = (['projection'], np.ones((1), dtype=np.double))
                        #
                        #   dsOut[proj.name].attrs['grid_mapping_name'] = proj.name
                        #   dsOut[proj.name].attrs['false_easting'] = float(proj.x_0)
                        #   dsOut[proj.name].attrs['false_northing'] = float(proj.y_0)
                        #   dsOut[proj.name].attrs['latitude_of_projection_origin'] = float(proj.lat_0)
                        #   dsOut[proj.name].attrs['longitude_of_projection_origin'] = float(proj.lon_0)
                        #   dsOut[proj.name].attrs['semi_major_axis'] = crs.ellipsoid.semi_major_metre
                        #   dsOut[proj.name].attrs['inverse_flattening'] = crs.ellipsoid.inverse_flattening
                        #   dsOut[proj.name].attrs['spatial_ref'] = crs.to_wkt(version='WKT1_ESRI') # ask for old=style WKT1 PROJCS instead of the new WKT2 PROJCRS
                        #   dsOut[proj.name].attrs['geotransform'] = "0 2000 0 0 0 -2000"
                        #
                        #   # Global Attributes
                        #   dsOut.attrs['datatype'] = domain.name
                        #   dsOut.attrs['Conventions'] = "CF-1.7"
                        #   dsOut.attrs['projection'] = proj.xy_proj.definition_string()
                        #   lon, lat = domain.lonlat_corners()
                        #   grid_corners = [lon[0], lat[0], lon[1], lat[1], lon[2], lat[2], lon[3], lat[3]]
                        #   dsOut.attrs['grid_corners'] = grid_corners
                        #
                        #   # y attributes
                        #   dsOut.y.attrs['standard_name'] = 'projection_y_coordinate'
                        #   dsOut.y.attrs['long_name'] = 'y coordinate of projection'
                        #   dsOut.y.attrs['units'] = 'm'
                        #
                        #   # x attributes
                        #   dsOut.x.attrs['standard_name'] = 'projection_x_coordinate'
                        #   dsOut.x.attrs['long_name'] = 'x coordinate of projection'
                        #   dsOut.x.attrs['units'] = 'm'
                        #
                        #   try:
                        #     time = 'time'
                        #     date = str(ds['time'].values[0]) #datetime.strptime(str(ds['time'].values[0]), '%Y-%m-%dT%H:%M:%S.000000000').strftime('%Y-%m-%d %H:%M')
                        #     if time:
                        #       dsOut[time].encoding['units'] = 'hours since ' + date
                        #       dsOut[time].encoding['standard_name'] = 'time'
                        #       dsOut[time].encoding['calendar'] = 'proleptic_gregorian'
                        #   except KeyError:
                        #     pass
                        #   name = afile[:afile.find(ext)]
                        #   if name:
                        #     dsOut.attrs['standard_name'] = name
                        #     dsOut.attrs['long_name'] = name
                        #
                        #   dsAttr = ds[var].attrs
                        #   for a in dsAttr:
                        #     dsOut[name].attrs[a] = dsAttr[a]
                        #
                        #   units = False #ds[var].units
                        #   if units:
                        #     dsOut[name].attrs['units'] = units
                        #
                        #   missing_value = np.nan
                        #   if missing_value:
                        #     dsOut.attrs['_FillValue'] = missing_value
                        #
                        #   # Flip around the horizontal
                        #   dsOut = dsOut.sortby('y', ascending=False)
                        #
                        #   # Write output file and perform the delayed computing
                        #   itsout = dsOut.to_netcdf(fileout, engine="netcdf4", compute=False)
                        #
                        #   with dask.diagnostics.ProgressBar():
                        #     results = itsout.compute()

                        # Close memory access
                        ds.close()
                        dsOut.close()
                    else:
                        os.system("cp " + filenc + " " + fileout)

        # Transform into numpy binary
        if ver.find('3.') > -1:
            binFileIO.main(iniFile, file_CatchmentsToProcess)
        print('finito...')
