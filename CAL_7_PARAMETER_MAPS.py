# -*- coding: utf-8 -*-
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
from ConfigParser import SafeConfigParser
import array
import logging
import random
from pcrasterCommand import pcrasterCommand, getPCrasterPath

if __name__=="__main__":

	########################################################################
	#   Read settings file
	########################################################################

	iniFile = os.path.normpath(sys.argv[1])
	print "=================== START ==================="
	print ">> Reading settings file ("+sys.argv[1]+")..."

	parser = SafeConfigParser()
	parser.read(iniFile)

	path_temp = parser.get('Path', 'Temp')
	path_maps = os.path.join(parser.get('Path', 'CatchmentDataPath'),"maps")
	path_result = parser.get('Path', 'Result')

	ParamRangesPath = parser.get('Path','ParamRanges')
	SubCatchmentPath = parser.get('Path','SubCatchmentPath')
	
	ForcingStart = datetime.datetime.strptime(parser.get('DEFAULT','ForcingStart'),"%m/%d/%Y")  # Start of forcing
	ForcingEnd = datetime.datetime.strptime(parser.get('DEFAULT','ForcingEnd'),"%m/%d/%Y")  # Start of forcing

	Qtss_csv = parser.get('CSV', 'Qtss')
	Qgis_csv = parser.get('CSV', 'Qgis')

	pcraster_path = parser.get('Path', 'PCRHOME')

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

	
	########################################################################
	#   Assign calibrated parameter values to maps
	########################################################################

	# Load paramranges file
	ParamRanges = pandas.read_csv(ParamRangesPath,sep=",",index_col=0)

	# Initialize parameter maps
	interstation_regions_map = os.path.join(path_result,"interstation_regions.map")
	for ii in range(0,len(ParamRanges)):
		exec("params_"+ParamRanges.index[ii]+"_map = os.path.join(path_result,\"params_"+ParamRanges.index[ii]+".map\")")
		exec("pcrasterCommand(pcrcalc + \" 'F0 = F1*0.0'\", {\"F0\": params_"+ParamRanges.index[ii]+"_map, \"F1\":interstation_regions_map})")    
		
	# Assign calibrated parameter values to maps
	count_front = 0
	count_nofront = 0
	for index, row in stationdata.iterrows():
		
		if np.isnan(row["CatchmentArea"]):
			continue
		
		print ">> Assigning values for catchment "+row['ID']+", size "+str(row['CatchmentArea'])+" pixels..."
		
		# Load calibrated parameter values for this catchment        
		# We want values on first line, since pareto_front is sorted by overall efficiency 
		# in descending order
		path_subcatch = os.path.join(SubCatchmentPath,row['ID'])
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
	print "---------------------------------------------"
	print "Number of catchments with pareto_front.csv: "+str(count_front)+"!"
	print "Number of catchments with missing pareto_front.csv: "+str(count_nofront)+"!"
	print "---------------------------------------------"
	print "==================== END ===================="