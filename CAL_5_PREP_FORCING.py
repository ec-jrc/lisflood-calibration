# -*- coding: utf-8 -*-
import os
import sys
import random
import numpy as np
import pandas
import pdb
import time
import struct
import shutil
from pcrasterCommand import pcrasterCommand, getPCrasterPath
ver = sys.version
ver = ver[:ver.find('(')-1]
if ver.find('3.') > -1:
  from configparser import ConfigParser # Python 3.8
else:
  from ConfigParser import SafeConfigParser # Python 2.7-15

########################################################################
#   Read settings file
########################################################################

iniFile = os.path.normpath(sys.argv[1])
file_CatchmentsToProcess = os.path.normpath(sys.argv[2])
print("=================== START ===================")
print(">> Reading settings file ("+sys.argv[1]+")...")

if ver.find('3.') > -1:
	parser = ConfigParser()  # python 3.8
else:
	parser = SafeConfigParser()  # python 2.7-15
parser.read(iniFile)

path_maps = os.path.join(parser.get('Path', 'CatchmentDataPath'),"maps")
path_result = parser.get('Path', 'Result')

CatchmentDataPath = parser.get('Path','CatchmentDataPath')
SubCatchmentPath = parser.get('Path','SubCatchmentPath')

pcraster_path = parser.get('Path', 'PCRHOME')

path_MeteoData = parser.get('Path', 'MeteoData')

switch_SubsetMeteoData = int(parser.get('DEFAULT', 'SubsetMeteoData'))

path_gauges = parser.get("Path", "gaugesPath")

config = {}
for execname in ["pcrcalc","map2asc","asc2map","col2map","map2col","mapattr","resample"]:
	config[execname] = getPCrasterPath(pcraster_path,sys.argv[1],execname)

pcrcalc = config["pcrcalc"]
col2map = config["col2map"]
map2col = config["map2col"]
resample = config["resample"]


########################################################################
#   Make stationdata array from the qgis csv
########################################################################

print(">> Reading Qgis2.csv file...")
stationdata = pandas.read_csv(os.path.join(path_result,"Qgis2.csv"),sep=",",index_col=0)
stationdata_sorted = stationdata.sort_values(by=['CatchmentArea'],ascending=True)

CatchmentsToProcess = pandas.read_csv(file_CatchmentsToProcess,sep=",",header=None)

for index, row in stationdata_sorted.iterrows():
	catchment = index
	Series = CatchmentsToProcess[0]
	if len(Series[Series==catchment]) == 0: # Only process catchments whose ObsID is in the CatchmentsToProcess.txt file
		continue
	print("\n\n\n=================== "+str(catchment)+" ====================")
	print(">> Starting map subsetting for catchment "+str(catchment)+", size "+str(row['CatchmentArea'])+" pixels...")

	t = time.time()

	path_subcatch = os.path.join(SubCatchmentPath,str(catchment))
	path_temp = path_subcatch

	if not os.path.exists(path_subcatch):
		os.makedirs(path_subcatch)
	if not os.path.exists(os.path.join(path_subcatch,'maps')):
		os.makedirs(os.path.join(path_subcatch,'maps'))
	if not os.path.exists(os.path.join(path_subcatch,'inflow')):
		os.makedirs(os.path.join(path_subcatch,'inflow'))
	if not os.path.exists(os.path.join(path_subcatch,'out')):
		os.makedirs(os.path.join(path_subcatch,'out'))
	
	# Make mask map for subcatchment
	subcatchmask_map = os.path.join(path_subcatch,"maps","mask.map")
	interstation_regions_map = os.path.join(path_result,"interstation_regions.map")
	pcrasterCommand(pcrcalc + " 'F0 = boolean(if(scalar(F1) eq "+str(index)+",scalar(1)))'", {"F0":subcatchmask_map,"F1":interstation_regions_map})
	tmp1_map = os.path.join(path_temp,"tmp1.map")
	smallsubcatchmask_map = os.path.join(path_subcatch,"maps","masksmall.map")
	pcrasterCommand(pcrcalc + " 'F0 = if(F1==1,F2)'", {"F0": tmp1_map, "F1":subcatchmask_map,"F2":subcatchmask_map})
	pcrasterCommand(resample + " -c 0 F0 F1" , {"F0":tmp1_map, "F1":smallsubcatchmask_map})
	
	# Ensure that there is only one outlet pixel in outlet map
	station_map = path_gauges
	subcatchstation_map = os.path.join(path_subcatch,"maps","outlet.map")	
	pcrasterCommand(pcrcalc + " 'F0 = boolean(if(scalar(F1) eq "+str(index)+",scalar(1)))'", {"F0":subcatchstation_map,"F1":station_map})
	subcatchstation_small_map = os.path.join(path_subcatch,"maps","outletsmall.map")
	pcrasterCommand(resample + " F0 F1 --clone F2" , {"F0": subcatchstation_map, "F1":subcatchstation_small_map, "F2":smallsubcatchmask_map})

	# Make inlet map
	inlets_map = os.path.join(path_result,"inlets.map")
	subcatchinlets_map = os.path.join(path_subcatch,"inflow","inflow.map")
	shutil.copyfile(inlets_map,subcatchinlets_map)
	pcrasterCommand(pcrcalc + " 'F0 = F1*scalar(F2)'", {"F0":subcatchinlets_map,"F1":subcatchinlets_map,"F2":subcatchmask_map})
		
	elapsed = time.time() - t
	print("   Time elapsed: "+"{0:.2f}".format(elapsed)+" s")

print("==================== END ====================")
