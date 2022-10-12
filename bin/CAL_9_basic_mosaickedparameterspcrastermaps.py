"""Please refer to quick_guide.pdf for usage instructions"""

import os
import sys
import pdb
import pandas
import string
import datetime
import time
import numpy as np
import re
from configparser import SafeConfigParser
import array
import logging
import random
from liscal.pcr_utils import pcrasterCommand, getPCrasterPath

if __name__=="__main__":

	########################################################################
	#   Read settings file
	########################################################################

	iniFile = os.path.normpath(sys.argv[1])
	print ("=================== START ===================")
	print( ">> Reading settings file ("+sys.argv[1]+")...")
 
  # Stef: very basic settings file to run this script settings_mosaickedparametersmaps.txt 

	parser = SafeConfigParser()
	parser.read(iniFile)

	#path_temp = parser.get('Path', 'Temp') # Stef: not needed for basic usage ...
	path_stations = parser.get ('Path', 'CatchmentStationDataPath')
	path_result = parser.get('Path', 'ResultMosaickedParametersMaps') # Stef: this is the folder where you will find the mosaic of the .map parameters values

	ParamRangesPath = parser.get('Path','ParamRanges') # Stef: this is the .csv file that you used to define the ranges of the calibration parameters
	SubCatchmentPath = parser.get('Path','CalibratedCatchments') # Stef: this is the folder where you collected all the calibrated catchments
	
	#ForcingStart = datetime.datetime.strptime(parser.get('DEFAULT','ForcingStart'),"%d/%m/%Y %H:%M")  # Start of forcing ## Stef: removed because it is not used to produce the parameters maps.
	#ForcingEnd = datetime.datetime.strptime(parser.get('DEFAULT','ForcingEnd'),"%d/%m/%Y %H:%M")  # Start of forcing ## Stef: removed because it is not used to produce the parameters maps.

	#Qtss_csv = parser.get('CSV', 'Qtss') ## Stef: removed because it is not used to produce the parameters maps.
	#Qgis_csv = parser.get('CSV', 'Qgis') ## Stef: removed because it is not used to produce the parameters maps.

	config = {}
 
	pcrcalc = "pcrcalc"
	col2map = "col2map"
	map2col = "map2col"
	resample = "resample"


	########################################################################
	#   Make stationdata array from the qgis csv
	########################################################################

	print (">> Reading Qgis2.csv file...")
	stationdata = pandas.read_csv(os.path.join(path_stations,"Qgis2.csv"),sep=",",index_col=0)

	
	########################################################################
	#   Assign calibrated parameter values to maps
	########################################################################

	# Load paramranges file
	ParamRanges = pandas.read_csv(ParamRangesPath,sep=",",index_col=0)

	# Initialize parameter maps
	interstation_regions_map = os.path.join(path_stations,"interstation_regions.map")
	for ii in range(0,len(ParamRanges)):
		exec("params_"+ParamRanges.index[ii]+"_map = os.path.join(path_result,\"params_"+ParamRanges.index[ii]+".map\")")
		exec("pcrasterCommand(pcrcalc + \" 'F0 = F1*0.0'\", {\"F0\": params_"+ParamRanges.index[ii]+"_map, \"F1\":interstation_regions_map})")    
		
	# Assign calibrated parameter values to maps
	count_front = 0
	count_nofront = 0
	for index, row in stationdata.iterrows():
		
		if np.isnan(row["DrainingArea.km2.LDD"]):
			continue
		print(row['ID'])
		print (">> Assigning values for catchment "+str(row['ID'])+", size "+str(row['DrainingArea.km2.LDD'])+" TOTAL drainage area from LDD in km2")
		
		# Load calibrated parameter values for this catchment        
		# We want values on first line, since pareto_front is sorted by overall efficiency 
		# in descending order
		path_subcatch = os.path.join(SubCatchmentPath,str(row['ID']))
		if os.path.isfile(os.path.join(path_subcatch,"pareto_front.csv")):
			count_front = count_front+1;
			pareto_front = pandas.read_csv(os.path.join(path_subcatch,"pareto_front.csv"))
		
			# Assign these to maps
			for ii in range(0,len(ParamRanges)):
				paramvalue = pareto_front["param_"+str(ii).zfill(2)+"_"+ParamRanges.index[ii]][0]
				exec("pcrasterCommand(pcrcalc + \" 'F0 = F0 + scalar(F1==scalar("+str(index)+"))*scalar("+str(paramvalue)+")'\", {\"F0\": params_"+ParamRanges.index[ii]+"_map, \"F1\":interstation_regions_map})")        
		else: # If pareto_front.csv doesn't exist, put -1
			count_nofront = count_nofront+1;
			for ii in range(0,len(ParamRanges)):
				exec("pcrasterCommand(pcrcalc + \" 'F0 = F0 + scalar(F1==scalar("+str(index)+"))*scalar(-9999)'\", {\"F0\": params_"+ParamRanges.index[ii]+"_map, \"F1\":interstation_regions_map})")        
	
	# Assign default values to uncalibrated areas
	# Ungauged areas have -1 in the interstation regions map
	# and -9999 in the parameter maps
	for ii in range(0,len(ParamRanges)):
		paramvalue = ParamRanges.iloc[ii,2]
		exec("pcrasterCommand(pcrcalc + \" 'F0 = F0 + scalar(F1==scalar(-1))*scalar("+str(paramvalue)+")'\", {\"F0\": params_"+ParamRanges.index[ii]+"_map, \"F1\":interstation_regions_map})")
		exec("pcrasterCommand(pcrcalc + \" 'F0 = F0 + scalar(F0==scalar(-9999))*(scalar("+str(paramvalue)+")+9999)'\", {\"F0\": params_"+ParamRanges.index[ii]+"_map})")
	print ("---------------------------------------------")
	print ("Number of catchments with pareto_front.csv: "+str(count_front)+"!")
	print ("Number of catchments with missing pareto_front.csv: "+str(count_nofront)+"!")
	print ("---------------------------------------------")
	print ("==================== END ====================")