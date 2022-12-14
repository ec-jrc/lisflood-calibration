#!/usr/bin/env python3
import os
import sys
import argparse
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


def set_calibrated_parameters(param_ranges, index, path_subcatch, lakes_reservoirs_default=False):
	count_front = 0
	if os.path.isfile(os.path.join(path_subcatch, "pareto_front.csv")):
		print(f'found {index}')
		count_front = 1;
		pareto_front = pandas.read_csv(os.path.join(path_subcatch,"pareto_front.csv"))
	
		# Assign these to maps
		for ii in range(0, len(param_ranges)):
			if lakes_reservoirs_default and param_ranges.index[ii] in ['LakeMultiplier', 'adjust_Normal_Flood', 'ReservoirRnormqMult']:
				paramvalue = param_ranges.iloc[ii,2]
			else:
				paramvalue = pareto_front["param_"+str(ii).zfill(2)+"_"+param_ranges.index[ii]][0]
			exec("pcrasterCommand(pcrcalc + \" 'F0 = F0 + scalar(F1==scalar("+str(index)+"))*scalar("+str(paramvalue)+")'\", {\"F0\": params_"+param_ranges.index[ii]+"_map, \"F1\":interstation_regions_map})")        
	else: # If pareto_front.csv doesn't exist, put -1
		raise Exception(f'Could not find optimised parameters for catchment {index} in {path_subcatch}')
		# for ii in range(0,len(ParamRanges)):
		# 	exec("pcrasterCommand(pcrcalc + \" 'F0 = F0 + scalar(F1==scalar("+str(index)+"))*scalar(-9999)'\", {\"F0\": params_"+ParamRanges.index[ii]+"_map, \"F1\":interstation_regions_map})")        
	return count_front


if __name__=="__main__":

	print("=================== START ===================")
	parser = argparse.ArgumentParser()
	parser.add_argument('--stations', '-s', required=True, help='Path to stations folder')
	# parser.add_argument('--catchments', '-c', required=True, help='Path to catchments folder')
	parser.add_argument('--output', '-o', required=True, help='Output folder')
	parser.add_argument('--params', '-p', required=True, help='Calibration parameters ranges')
	parser.add_argument('--regionalisation', '-r', help='Path to regionalisation csv file')
	args = parser.parse_args()

	#path_temp = parser.get('Path', 'Temp') # Stef: not needed for basic usage ...
	path_stations = args.stations
	path_result = args.output # Stef: this is the folder where you will find the mosaic of the .map parameters values

	ParamRangesPath = args.params # Stef: this is the .csv file that you used to define the ranges of the calibration parameters
	SubCatchmentPath = args.catchments # Stef: this is the folder where you collected all the calibrated catchments

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

	print (">> Reading stations data file...")
	stationdata = pandas.read_csv(os.path.join(path_stations, 'stations_data.csv'), sep=",", index_col=0)


	########################################################################
	#   Assign calibrated parameter values to maps
	########################################################################

	# Load paramranges file
	param_ranges = pandas.read_csv(ParamRangesPath,sep=",",index_col=0)

	# Initialize parameter maps
	interstation_regions_map = os.path.join(path_stations,"interstation_regions.map")
	for ii in range(0,len(param_ranges)):
		exec("params_"+param_ranges.index[ii]+"_map = os.path.join(path_result,\"params_"+param_ranges.index[ii]+".map\")")
		exec("pcrasterCommand(pcrcalc + \" 'F0 = F1*0.0'\", {\"F0\": params_"+param_ranges.index[ii]+"_map, \"F1\":interstation_regions_map})")    
		
	# Assign calibrated parameter values to maps
	count_front = 0
	for index, row in stationdata.iterrows():

		print(index)
		
		# if np.isnan(row["DrainingArea.km2.LDD"]):
		# 	print('No draining area')
		# 	continue
		# print(row['ID'])
		# print (">> Assigning values for catchment "+str(row['ID'])+", size "+str(row['DrainingArea.km2.LDD'])+" TOTAL drainage area from LDD in km2")
		
		# Load calibrated parameter values for this catchment        
		# We want values on first line, since pareto_front is sorted by overall efficiency 
		# in descending order

		path_subcatch = os.path.join(SubCatchmentPath,str(index))
		count_front += set_calibrated_parameters(param_ranges, index, path_subcatch)
	
	print ("---------------------------------------------")
	print ("Number of calibrated catchments with pareto_front.csv: "+str(count_front)+"!")
	print ("---------------------------------------------")
	# Assign default values to uncalibrated areas
	# Ungauged areas have -1 in the interstation regions map
	# and -9999 in the parameter maps
	if args.regionalisation:
		count_reg_front = 0
		donors_data = pandas.read_csv(args.regionalisation, sep=",", index_col=0)
		for index, row in donors_data.iterrows():
			donor_id = row['DonorID']
			print(index, donor_id)
			path_subcatch = os.path.join(SubCatchmentPath, str(donor_id))

			count_reg_front += set_calibrated_parameters(param_ranges, index, path_subcatch, lakes_reservoirs_default=True)
		print ("---------------------------------------------")
		print ("Number of regionalised catchments with pareto_front.csv: "+str(count_reg_front)+"!")
		print ("---------------------------------------------")
	else:
		for ii in range(0,len(param_ranges)):
			paramvalue = param_ranges.iloc[ii,2]
			exec("pcrasterCommand(pcrcalc + \" 'F0 = F0 + scalar(F1==scalar(-1))*scalar("+str(paramvalue)+")'\", {\"F0\": params_"+param_ranges.index[ii]+"_map, \"F1\":interstation_regions_map})")
			exec("pcrasterCommand(pcrcalc + \" 'F0 = F0 + scalar(F0==scalar(-9999))*(scalar("+str(paramvalue)+")+9999)'\", {\"F0\": params_"+param_ranges.index[ii]+"_map})")

	print ("==================== END ====================")
