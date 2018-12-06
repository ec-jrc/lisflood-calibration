# -*- coding: utf-8 -*-
"""Please refer to quick_guide.pdf for usage instructions"""

import os
import sys
import random
import numpy as np
import pandas
import pdb
import time
import struct
from ConfigParser import SafeConfigParser
import shutil
from pcrasterCommand import pcrasterCommand, getPCrasterPath


########################################################################
#   Read settings file
########################################################################

iniFile = os.path.normpath(sys.argv[1])
file_CatchmentsToProcess = os.path.normpath(sys.argv[2])
print "=================== START ==================="
print ">> Reading settings file ("+sys.argv[1]+")..."

parser = SafeConfigParser()
parser.read(iniFile)

path_temp = parser.get('Path', 'Temp')
path_maps = os.path.join(parser.get('Path', 'CatchmentDataPath'),"maps")
path_result = parser.get('Path', 'Result')

if not os.path.exists(path_temp): os.makedirs(path_temp)

CatchmentDataPath = parser.get('Path','CatchmentDataPath')
SubCatchmentPath = parser.get('Path','SubCatchmentPath')

pcraster_path = parser.get('Path', 'PCRHOME')

path_MeteoData = parser.get('Path', 'MeteoData')

switch_SubsetMeteoData = int(parser.get('DEFAULT', 'SubsetMeteoData'))

config = {}
for execname in ["pcrcalc","map2asc","asc2map","col2map","map2col","mapattr","resample"]:
	config[execname] = getPCrasterPath(pcraster_path,execname)

pcrcalc = config["pcrcalc"]
col2map = config["col2map"]
map2col = config["map2col"]
resample = config["resample"]


########################################################################
#   Make stationdata array from the qgis csv
########################################################################

print ">> Reading Qgis2.csv file..."
stationdata = pandas.read_csv(os.path.join(path_result,"Qgis2.csv"),sep=",",index_col=0)
stationdata_sorted = stationdata.sort_index(by=['CatchmentArea'],ascending=True)

CatchmentsToProcess = pandas.read_csv(file_CatchmentsToProcess,sep=",",header=None)

for index, row in stationdata_sorted.iterrows():
	Series = CatchmentsToProcess.ix[:,0]
	if len(Series[Series==str(row["ID"])]) == 0: # Only process catchments whose ID is in the CatchmentsToProcess.txt file
		continue
	print "=================== "+row['ID']+" ===================="
	print ">> Starting map subsetting for catchment "+row['ID']+", size "+str(row['CatchmentArea'])+" pixels..."

	t = time.time()

	path_subcatch = os.path.join(SubCatchmentPath,row['ID'])

	# Delete all files in catchment directory
	for root, dirs, files in os.walk(path_subcatch, topdown=False):
		 
		for name in files:
			os.remove(os.path.join(root, name))
			#print "   Deleting "+name
		for name in dirs:
			#print "   Deleting "+os.path.join(root, name)
			os.rmdir(os.path.join(root, name))
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
	
	## Make reservoir and lake maps for subcatchment
	## Needed to set reservoir and lake switch in xml
	#pcrasterCommand(pcrcalc + " 'F0 = scalar(F1)*scalar(F2)'", {"F0":os.path.join(path_subcatch,"maps","lakes.map"),"F1":os.path.join(CatchmentDataPath,"maps","lakes.map"),"F2":subcatchmask_map})
	#pcrasterCommand(pcrcalc + " 'F0 = scalar(F1)*scalar(F2)'", {"F0":os.path.join(path_subcatch,"maps","res.map"),"F1":os.path.join(CatchmentDataPath,"maps","res.map"),"F2":subcatchmask_map})
	
	# Resample input, by first clipping the mask, and then using the new mask as clonemap
	"""
	directories = {"maps",os.path.join("maps","percent"),os.path.join("maps","soilhyd"),
					os.path.join("maps","table2map"),"out","wateruse","tables",
				   os.path.join("lai","laiforest"),os.path.join("lai","laiother"),
				   os.path.join("lai","laiirg"),os.path.join("lai","lairice"),
				   os.path.join("lai","old_laiother")}
	if switch_SubsetMeteoData == 1:
		directories.add("meteo")
	tmp2_map = os.path.join(path_temp,"tmp2.map")
	cnt = 0
	for directory in directories:

		path = os.path.join(CatchmentDataPath,directory)
		if (directory is "meteo"):
			path = path_MeteoData

		sys.stdout.write("   Copying from "+path)

		files = os.listdir(path)
		if not os.path.exists(os.path.join(path_subcatch,directory)):
			os.makedirs(os.path.join(path_subcatch,directory))

		for i in files:

			if i.endswith(".map") or i[-3:].isdigit(): # file has to end with .map or end with numbers (in the case of meteo data)

				sys.stdout.write(".")
				
				input_map = os.path.join(path,i)
				output_map = os.path.join(path_subcatch,directory,i)

				# Open file in binary mode, read access
				try:
					CSFIn = open(input_map,'rb') # copied these lines from copysmall2.py
				except:
					continue
				CSFIn.seek(64)
				ValueScale=CSFIn.read(2)
				CSFIn.close()
				CellRep = struct.unpack("H", ValueScale)

				if CellRep[0]==240: # if ldd
					pcrasterCommand(pcrcalc + " 'F0 = scalar(F1)'", {"F0":tmp1_map, "F1":input_map})
					pcrasterCommand(resample + " F0 F1 --clone F2" , {"F0": tmp1_map, "F1":tmp2_map, "F2":smallsubcatchmask_map})
					pcrasterCommand(pcrcalc + " 'F0 = ldd(F1)'", {"F0":output_map, "F1":tmp2_map})
				else:
					pcrasterCommand(resample + " F0 F1 --clone F2" , {"F0": input_map, "F1":output_map, "F2":smallsubcatchmask_map})
				cnt += 1

			else: # not a map, but just a txt file or something
				input_file = os.path.join(path,i)
				output_file = os.path.join(path_subcatch,directory,i)
				if os.path.isfile(input_file):
					sys.stdout.write(",")                    
					shutil.copyfile(input_file,output_file)

		sys.stdout.write("\n")

	print "   Done subsetting/copying "+str(cnt)+" files"
	"""
	
	# Ensure that there is only one outlet pixel in outlet map
	station_map = os.path.join(path_result,"outlet.map")
	subcatchstation_map = os.path.join(path_subcatch,"maps","outlet.map")	
	pcrasterCommand(pcrcalc + " 'F0 = boolean(if(scalar(F1) eq "+str(index)+",scalar(1)))'", {"F0":subcatchstation_map,"F1":station_map})
	subcatchstation_small_map = os.path.join(path_subcatch,"maps","outletsmall.map")
	#shutil.copyfile(subcatchstation_map,subcatchstation_small_map)
	pcrasterCommand(resample + " F0 F1 --clone F2" , {"F0": subcatchstation_map, "F1":subcatchstation_small_map, "F2":smallsubcatchmask_map})

	# Make inlet map
	inlets_map = os.path.join(path_result,"inlets.map")
	subcatchinlets_map = os.path.join(path_subcatch,"inflow","inflow.map")
	shutil.copyfile(inlets_map,subcatchinlets_map)
	pcrasterCommand(pcrcalc + " 'F0 = F1*scalar(F2)'", {"F0":subcatchinlets_map,"F1":subcatchinlets_map,"F2":subcatchmask_map})
		
	elapsed = time.time() - t
	print "   Time elapsed: "+"{0:.2f}".format(elapsed)+" s"

print "==================== END ===================="