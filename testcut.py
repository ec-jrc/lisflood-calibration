import pandas  as pd
import xarray as xr
import numpy as np
import dask # check if dask is installed (not used directly)
import sys
import os
from pcraster import *
import pdb
import time
import struct
from ConfigParser import SafeConfigParser
import shutil
from pcrasterCommand import pcrasterCommand, getPCrasterPath
from pcraster import *
from netCDF4 import Dataset

#filenc=sys.argv[1]
#filemask=sys.argv[2]
#var=sys.argv[3]
#fileout=filenc[:-3]+"_tr.nc"
#time=sys.argv[5]

########################################################################
#   Read settings file
########################################################################
from typing import Any

iniFile = os.path.normpath(sys.argv[1])
file_CatchmentsToProcess = os.path.normpath(sys.argv[2])
print "=================== START ==================="
print ">> Reading settings file ("+sys.argv[1]+")..."

parser = SafeConfigParser()
parser.read(iniFile)

path_maps = os.path.join(parser.get('Path', 'CatchmentDataPath'),"maps")

CatchmentDataPath = parser.get('Path','CatchmentDataPath')
SubCatchmentPath = parser.get('Path','SubCatchmentPath')

pcraster_path = parser.get('Path', 'PCRHOME')

path_MeteoData = parser.get('Path', 'MeteoData')
path_result = parser.get('Path', 'Result')
switch_SubsetMeteoData = int(parser.get('DEFAULT', 'SubsetMeteoData'))


print ">> Reading Qgis2.csv file..."
stationdata = pd.read_csv(os.path.join(path_result,"Qgis2.csv"),sep=",",index_col=0)
stationdata_sorted = stationdata.sort_index(by=['CatchmentArea'],ascending=True)
CatchmentsToProcess = pd.read_csv(file_CatchmentsToProcess,sep=",",header=None)

for index, row in stationdata_sorted.iterrows():
    Series = CatchmentsToProcess.ix[:,0]
    if len(Series[Series==str(row["ID"])]) == 0: # Only process catchments whose ID is in the CatchmentsToProcess.txt file
        continue
    print "=================== "+row['ID']+" ===================="
    print ">> Starting map subsetting for catchment "+row['ID']+", size "+str(row['CatchmentArea'])+" pixels..."

    #t = time.time()

    path_subcatch = os.path.join(SubCatchmentPath,row['ID'])
    path_subcatch_maps = os.path.join(path_subcatch,'maps')

    # Cut bbox from ALL static maps and forcings for subcatchment
    #open the maskmap
    maskpcr = os.path.join(path_subcatch,'maps','mask.map')
    maskmap=readmap(maskpcr)
    #get the mask coordinates
    masknp=pcr2numpy(maskmap,False)
    mask_filter=np.where(masknp)
    x_min=np.min(mask_filter[0])
    x_max=np.max(mask_filter[0])
    y_min=np.min(mask_filter[1])
    y_max=np.max(mask_filter[1])

    # Enter in static_map dir
    # walk through subfolders
    for root,dirs,files in os.walk(path_maps, topdown=False):
        for afile in files:
            filenc=os.path.join(root, afile)
            print filenc[-3:]
            if filenc[-3:]=='.nc':
                print filenc
                timeflag=False
                nc = xr.open_dataset(filenc)
                if 'time' in nc.variables:
                    timeflag=True

                var = nc.variables.items()[-1][0]
                #var=nc.variables[-1]

                if 'lat' in nc.variables:
                    x='lon'
                    y='lat'
                else:
                    x='x'
                    y='y'

                if timeflag:
                    sliced_var = nc[var][:,x_min:x_max+1,y_min:y_max+1]
                    t0 = pd.datetime.now()
                else:
                    sliced_var = nc[var][x_min:x_max+1,y_min:y_max+1]


                #print sliced_var


                if 'laea' in nc.variables:
                    sliced_var['laea'] = nc['laea']
                elif 'lambert_azimuthal_equal_area' in nc.variables:
                    sliced_var['lambert_azimuthal_equal_area'] = nc['lambert_azimuthal_equal_area']

                slicedfile=filenc+'.sliced'
                sliced_var.to_netcdf(slicedfile)

                new_nc = xr.open_dataset(slicedfile)  # type: nc4
                new_nc.attrs = nc.attrs
                fileout=os.path.join(path_subcatch_maps,afile)
                print 'out:',fileout
                if 'laea' in nc.variables:
                    if timeflag:
                        new_nc[[x,y,'time','laea',var]].to_netcdf(fileout)
                    else:
                        new_nc[[x,y,'laea',var]].to_netcdf(fileout)
                elif 'lambert_azimuthal_equal_area' in nc.variables:
                    if timeflag==True:
                        new_nc[[x,y,'time','lambert_azimuthal_equal_area',var]].to_netcdf(fileout)
                    else:
                        new_nc[[x,y,'lambert_azimuthal_equal_area',var]].to_netcdf(fileout)
                else:
                    if timeflag == True:
                        new_nc[[x, y, 'time', var]].to_netcdf(fileout)
                    else:
                        new_nc[[x, y, var]].to_netcdf(fileout)
                nc.close()
                os.remove(slicedfile)
            if filenc[-3:]=='.map':
#nc = xr.open_dataset(filenc)
                print 'pcraster'



#if time in nc.variables:
#    nc = xr.open_dataset(filenc,chunks={'time':100})
#else:
#    nc = xr.open_dataset(filenc)
    


#var = nc.variables.items()[-1][0]
#var=nc.variables[-1]

#if 'lat' in nc.variables:
#    x='lon'
#    y='lat'
#else:
#    x='x'
#    y='y'

#if 'time' in nc.variables:
#    sliced_var = nc[var][:,x_min:x_max+1,y_min:y_max+1]
#    t0 = pd.datetime.now()
#else:
#    sliced_var = nc[var][x_min:x_max+1,y_min:y_max+1]
#print sliced_var


#if 'laea' in nc.variables:
#       sliced_var['laea'] = nc['laea']
#else:
#       sliced_var['lambert_azimuthal_equal_area'] = nc['lambert_azimuthal_equal_area']


#sliced_var.to_netcdf('sliced.nc')

#new_nc = xr.open_dataset('sliced.nc')
#new_nc.attrs = nc.attrs
#if 'laea' in nc.variables:
#    if 'time' in nc.variables:
#       new_nc[[x,y,'time','laea',var]].to_netcdf(fileout)
#    else:
#       new_nc[[x,y,'laea',var]].to_netcdf(fileout)
#else:
#    if 'time' in nc.variables:
#       new_nc[[x,y,'time','lambert_azimuthal_equal_area',var]].to_netcdf(fileout)
#    else:
#       new_nc[[x,y,'lambert_azimuthal_equal_area',var]].to_netcdf(fileout)



