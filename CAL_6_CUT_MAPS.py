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
import subprocess

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
file_CatchmentsToProcess = os.path.normpath(sys.argv[2])

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
    maskmap=setclone(maskpcr)
    #get the mask coordinates
    
    
    if os.path.isfile(maskpcr):
        print 'maskmap',maskpcr
        maskmap=readmap(maskpcr)
    
    else:
        print 'wrong input mask file'
        sys.exit(1)

    
    #x_ul=clone().west()
    #x_lr=clone().west() + clone().cellSize()*clone().nrCols()
    #y_ul=clone().north()
    #y_lr=clone().north() - clone().cellSize()*clone().nrRows()
      
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
            fileout=os.path.join(path_subcatch_maps,afile)
            #if os.path.isfile(fileout):
            #    print fileout,'alreay existing'
            #    continue
                
            print filenc[-3:]
            if filenc[-3:]=='.nc':
                #pdb.set_trace()
                print 'creating...',fileout
                try:
                    nc = xr.open_dataset(filenc,chunks={'time':100})
                except:
                    nc = xr.open_dataset(filenc)
                var = nc.variables.items()[-1][0]
                nc.variables[var].attrs['esri_pe_string']='GEOGCS[\"GCS_WGS_1984\",DATUM[\"D_WGS_1984\",SPHEROID[\"WGS_1984\",6378137,298.257223563]],PRIMEM[\"Greenwich\",0],UNIT[\"Degree\",0.0174532925199433]]"'
#var=nc.variables[-1]

                if 'time' in nc.variables:
                    sliced_var = nc[var][:,x_min:x_max+1,y_min:y_max+1]
                    t0 = pd.datetime.now()
                else:
                    sliced_var = nc[var][x_min:x_max+1,y_min:y_max+1]
                sliced_var.to_netcdf(fileout)
                nc.close()

                #cmd = ' '.join(['gdal_translate -projwin',str(x_ul),str(y_ul),str(x_lr),str(y_lr),'-projwin_srs EPSG:4326 -of netCDF  --config GDAL_CACHEMAX 128 -co WRITE_BOTTOMUP=NO -co NUM_THREADS=32'])
                #fullCmd=' '.join([cmd,filenc,fileout])
                #print 'fullCmd',fullCmd 
                #subprocess.Popen(fullCmd,shell=True)
                
                print 'finito...'
    #            
            #if filenc[-7:]=='lat.map':
            #    print 'pcraster'
            #    print 'creating...',fileout
            #    cmd = ' '.join(['gdal_translate -projwin',str(x_ul),str(y_ul),str(x_lr),str(y_lr),'-projwin_srs EPSG:4326 -of PCRaster  --config GDAL_CACHEMAX 128 -co WRITE_BOTTOMUP=NO -co NUM_THREADS=32'])
            #    fullCmd=' '.join([cmd,filenc,fileout])
            #    print 'fullcmd',fullCmd
    #            
            #    subprocess.Popen(fullCmd,shell=True)
                
            #    print 'finito...'
                
#nc = xr.open_dataset(filenc)
    #            print 'pcraster'
    for root,dirs,files in os.walk(path_MeteoData, topdown=False):
        for afile in files:
            filenc=os.path.join(root, afile)
            print filenc[-3:]
            if filenc[-3:]=='.nc':
                #pdb.set_trace()
    #            
                fileout=os.path.join(path_subcatch_maps,afile)
                print 'creating...',fileout
                try:
                    nc = xr.open_dataset(filenc,chunks={'time':100})
                except:
                    nc = xr.open_dataset(filenc)
                var = nc.variables.items()[-1][0]
                nc.variables[var].attrs['esri_pe_string']='GEOGCS[\"GCS_WGS_1984\",DATUM[\"D_WGS_1984\",SPHEROID[\"WGS_1984\",6378137,298.257223563]],PRIMEM[\"Greenwich\",0],UNIT[\"Degree\",0.0174532925199433]]"'
#var=nc.variables[-1]

                if 'time' in nc.variables:
                    sliced_var = nc[var][:,x_min:x_max+1,y_min:y_max+1]
                    t0 = pd.datetime.now()
                else:
                    sliced_var = nc[var][x_min:x_max+1,y_min:y_max+1]
                sliced_var.to_netcdf(fileout)

                #cmd = ' '.join(['gdal_translate -projwin',str(x_ul),str(y_ul),str(x_lr),str(y_lr),'-projwin_srs EPSG:4326 -of netCDF  --config GDAL_CACHEMAX 128 -co WRITE_BOTTOMUP=NO -co NUM_THREADS=32'])
                #fullCmd=' '.join([cmd,filenc,fileout])
                #print 'fullCmd',fullCmd 
                #subprocess.Popen(fullCmd,shell=True)
                nc.close()
                
                print 'finito...'