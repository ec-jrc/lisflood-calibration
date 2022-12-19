#!/usr/bin/env python3
import os
import argparse
import pandas
import numpy as np
import pcraster as pcr

from liscal.pcr_utils import pcrasterCommand, getPCrasterPath


def set_calibrated_parameters(param_ranges, index, path_subcatch, params, interstation, lakes_reservoirs_default=False):
	count_front = 0
	if os.path.isfile(os.path.join(path_subcatch, "pareto_front.csv")):
		count_front = 1
		pareto_front = pandas.read_csv(os.path.join(path_subcatch,"pareto_front.csv"))
	
		# Assign these to maps
		for ii in range(0, len(param_ranges)):
			if lakes_reservoirs_default and param_ranges.index[ii] in ['LakeMultiplier', 'adjust_Normal_Flood', 'ReservoirRnormqMult']:
				paramvalue = param_ranges.iloc[ii,2]
			else:
				paramvalue = pareto_front["param_"+str(ii).zfill(2)+"_"+param_ranges.index[ii]][0]
			param = param_ranges.index[ii]
			params[param] = pcr.ifthenelse(interstation==index, pcr.scalar(float(paramvalue)), pcr.scalar(params[param]))
	else:
		raise Exception(f'Could not find optimised parameters for catchment {index} in {path_subcatch}')
	return count_front


def export_netcdf(path_result, template, param_map, name):
	ds = xr.open_dataarray(template)
	ds.name = name
	ds.attrs['standard_name'] = name
	ds.attrs['long_name'] = name
	ds.values = pcr.pcr2numpy(param_map, np.NaN)
	ds.to_netcdf(os.path.join(path_result, f'{name}.nc'))


if __name__=="__main__":

	print("=================== START ===================")
	parser = argparse.ArgumentParser()
	parser.add_argument('--stations', '-s', required=True, help='Path to stations folder containing interstation_regions.map and stations_data.csv')
	parser.add_argument('--catchments', '-c', required=True, help='Path to catchments folder')
	parser.add_argument('--output', '-o', required=True, help='Output folder')
	parser.add_argument('--params', '-p', required=True, help='Path to calibration parameters ranges csv file')
	parser.add_argument('--template', '-t', required=True, help='Path to NetCDF template')
	parser.add_argument('--regionalisation', '-r', help='Path to regionalisation csv file')
	args = parser.parse_args()

	path_stations = args.stations
	path_result = args.output
	template = args.template
	
	if not os.path.exists(path_result):
		os.makedirs(path_result)

	ParamRangesPath = args.params
	SubCatchmentPath = args.catchments

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
	interstation = pcr.readmap(interstation_regions_map)
	params = {}
	for ii in range(0,len(param_ranges)):
		param = param_ranges.index[ii]
		params[param] = interstation*0.0
		
	# Assign calibrated parameter values to maps
	count_front = 0
	for index, row in stationdata.iterrows():
		print(index)
		path_subcatch = os.path.join(SubCatchmentPath,str(index))
		count_front += set_calibrated_parameters(param_ranges, index, path_subcatch, params, interstation)
	
	print ("---------------------------------------------")
	print ("Number of calibrated catchments with pareto_front.csv: "+str(count_front)+"!")
	print ("---------------------------------------------")
	if args.regionalisation:
		count_reg_front = 0
		donors_data = pandas.read_csv(args.regionalisation, sep=",", index_col=0)
		for index, row in donors_data.iterrows():
			donor_id = row['DonorID']
			print(index, donor_id)
			path_subcatch = os.path.join(SubCatchmentPath,str(donor_id))

			count_reg_front += set_calibrated_parameters(param_ranges, index, path_subcatch, params, interstation, lakes_reservoirs_default=True)
		print ("---------------------------------------------")
		print ("Number of regionalised catchments with pareto_front.csv: "+str(count_reg_front)+"!")
		print ("---------------------------------------------")

	# Assign default values to uncalibrated areas
	# Ungauged areas have -1 in the interstation regions map
	print ("Setting all -1 values to default")
	for ii in range(0,len(param_ranges)):
		param = param_ranges.index[ii]
		paramvalue = param_ranges.iloc[ii,2]
		params[param] = pcr.ifthenelse(interstation==-1, pcr.scalar(paramvalue), params[param])
		pcr.report(params[param], f"params_{param_ranges.index[ii]}.map")
		export_netcdf(path_result, template, params[param], param_ranges.index[ii])

	print ("==================== END ====================")
