#!/usr/bin/env python3
import os
import sys
import argparse
import pandas
import datetime
import numpy as np

from liscal.pcr_utils import pcrasterCommand, getPCrasterPath

if __name__=="__main__":

	print("=================== START ===================")
	parser = argparse.ArgumentParser()
	parser.add_argument('stations_data', help='Stations data csv file')
	parser.add_argument('ldd', help='Path to ldd map')
	parser.add_argument('stations_dir', help='Output stations directory')
	parser.add_argument('tmp_dir', help='Temporary directory')
	args = parser.parse_args()

	########################################################################
	#   Read settings file
	########################################################################
	path_result = args.stations_dir

	stations_file = args.stations_data

	pcrcalc = "pcrcalc"
	col2map = "col2map"
	map2col = "map2col"
	resample = "resample"

	path_temp = args.tmp_dir
	tmp_map = os.path.join(path_temp,"tmp.map")
	tmp2_map = os.path.join(path_temp,"tmp2.map")
	tmp_txt = os.path.join(path_temp,"tmp.txt")
	tmp2_txt = os.path.join(path_temp,"tmp2.txt")

	ldd_map = os.path.join(args.ldd)
	station_map = os.path.join(path_result, 'gauges.map')

	########################################################################
	#   Read stations data file
	########################################################################
	print(">> Reading stations data csv file...")
	stationdata = pandas.read_csv(stations_file, sep=",", index_col=0)
	
	########################################################################
	#   Make map with station locations
	########################################################################
	print(">> Make map with station locations (outlet.map)...")
	if not os.path.exists(path_temp): os.makedirs(path_temp)
	if not os.path.exists(path_result): os.makedirs(path_result)

	station_txt = os.path.join(path_temp, "station.txt")
	with open(station_txt, 'w') as f:
		for index, row in stationdata.iterrows():
			f.write(str(row['LisfloodX']) + " ")
			f.write(str(row['LisfloodY']) + " ")
			f.write(str(float(index)) + "\n")
		f.close()
	pcrasterCommand(col2map + " F0 F1 -N --clone F2 --large"  , {"F0": station_txt, "F1":station_map, "F2":ldd_map})

		
	########################################################################
	# 	Check for station location conflicts (stations with same coordinates)
	########################################################################	
	
	print(">> Check for station conflicts...")
	counter = 0
	for index, row in stationdata.iterrows():			
		#pcrasterCommand(pcrcalc + " 'F0 = if(scalar(F1)==scalar("+str(int(index))+"),scalar(1.0))'", {"F0": tmp_map, "F1":station_map})
		#pcrasterCommand(pcrcalc + " 'F0 = if(scalar(F1)==scalar(5555),scalar(1.0))'", {"F0": tmp_map, "F1":station_map})
		pcrasterCommand(pcrcalc + " 'F0 = if(scalar(F1)=="+str(int(index))+",scalar(F1))'", {"F0": tmp_map, "F1":station_map})
		pcrasterCommand(map2col + " F0 F1"  , {"F0": tmp_map, "F1":tmp_txt})
		with open(tmp_txt,"r") as f:
			counter2 = 0
			for line in f.readlines():
				(X,Y,value) = line.split()
				counter2 = counter2+1
		if counter2==0:
			print("Station ID "+str(index)+" not found in outlet.map! Is there another station with the same location?")
			counter = counter+1
		elif counter2>1:		
			print("Station ID "+str(index)+" found multiple times in outlet.map!")
			counter = counter+1
		if counter2==1:
			if int(value)!=index:
				print("Wrong station ID for station "+str(index)+"; instead of "+str(index)+" we found "+value)
				counter = counter+1
			else:
				print("Station "+str(index)+" OK")			
	if counter>0:
		print("Number of station location conflicts: "+str(counter))
		print("Fix these! Enter 'c' to continue for now")
		raise Exception("ERROR")
	
	
	########################################################################
	#   Compute catchment mask maps and write to temporary directory and
	#   make array with catchment upstream number of pixels
	########################################################################

	sys.stdout.write(">> Compute catchment masks (catchmaskXXXXX.map)")

	stationdata['CatchmentArea'] = np.nan
	accuflux_map = os.path.join(path_temp,"accuflux.map")	
	pcrasterCommand(pcrcalc + " 'F0 = accuflux(F1,1)'", {"F0": accuflux_map, "F1":ldd_map})	
	for index, row in stationdata.iterrows():
		sys.stdout.write(".")

		# make map of station location
		content = "%s %s %s\n" % (row['LisfloodX'],row['LisfloodY'],1)
		with open(tmp_txt,"w") as f1:
			f1.write(content)
			f1.close()
		pcrasterCommand(col2map + " F0 F1 -N --clone F2 --large" ,{"F0": tmp_txt, "F1":tmp_map, "F2":ldd_map})

		# compute catchment mask
		catchment_map = os.path.join(path_temp,"catchmask%05d.map" % float(index))
		pcrasterCommand(pcrcalc + " 'F0 = boolean(catchment(F1,F2))'", {"F0": catchment_map, "F1":ldd_map, "F2":tmp_map})
		pcrasterCommand(pcrcalc + " 'F0 = if((scalar(F0) gt (scalar(F0) *0)) then F0)' ", {"F0": catchment_map})

		# compute number of upstream pixels for station
		pcrasterCommand(pcrcalc + " 'F0 = if(defined(F1),F2)'", {"F0": tmp2_map, "F1":tmp_map, "F2":accuflux_map})
		pcrasterCommand(map2col + " F0 F1"  , {"F0": tmp2_map, "F1":tmp2_txt})
		with open(tmp2_txt,"r") as f:
			for line in f.readlines():
				(X,Y,value) = line.split()
				stationdata.loc[index,'CatchmentArea'] = float(value)
			f.close()


	########################################################################
	#   Make map with IDs for interstation regions
	########################################################################

	sys.stdout.write("\n>> Compute interstation regions (interstation_regions.map)")

	interstation_regions_map = os.path.join(path_result,"interstation_regions.map")
	pcrasterCommand(pcrcalc + " 'F0 = scalar(F1)*0-1'", {"F0": interstation_regions_map, "F1":ldd_map}) # initialize interstation_regions_map
	stationdata_sorted = stationdata.sort_values(by=['CatchmentArea'],ascending=False)
	for index, row in stationdata_sorted.iterrows():
		sys.stdout.write(".")
		catchment_map = os.path.join(path_temp,"catchmask%05d.map" % float(index))
		pcrasterCommand(pcrcalc + " 'F0 = F0 * (1-scalar(cover(F1,0.0)))'", {"F0": interstation_regions_map, "F1":catchment_map})
		pcrasterCommand(pcrcalc + " 'F0 = F0 + scalar(cover(F1,0.0)) * " + str(index) + "'", {"F0": interstation_regions_map, "F1":catchment_map})


	########################################################################
	#   Make map with sampling frequency for interstation regions
	#   The map indicates how many times a particular pixel is included in a
	#   catchment
	########################################################################

	sys.stdout.write("\n>> Compute sampling frequency (sampling_frequency.map)...")

	sampling_frequency_map = os.path.join(path_result,"sampling_frequency.map")
	pcrasterCommand(pcrcalc + " 'F0 = scalar(F1)*0'", {"F0": sampling_frequency_map, "F1":ldd_map}) # initialize interstation_regions_map
	for index, row in stationdata.iterrows():
		sys.stdout.write(".")
		catchment_map = os.path.join(path_temp,"catchmask%05d.map" % float(index))
		pcrasterCommand(pcrcalc + " 'F0 = F0 + scalar(cover(F1,0.0))'", {"F0": sampling_frequency_map, "F1":catchment_map})


	########################################################################
	#   Make csv listing for each catchment 1) the directly connected
	#   subcatchments and 2) the corresponding inflow locations
	########################################################################

	sys.stdout.write("\n>> Compute links and inflow locations (inlets.map and direct_links.csv)")

	stationdata['SamplingFrequency'] = np.nan
	direct_links_csv = os.path.join(path_result, "direct_links.csv")
	for index, row in stationdata.iterrows():
		sys.stdout.write(".")

		# make map of station location
		content = "%s %s %s\n" % (row['LisfloodX'],row['LisfloodY'],1)
		f1 = open(tmp_txt,"w")
		f1.write(content)
		f1.close()
		pcrasterCommand(col2map + " F0 F1 -N --clone F2 --large" ,{"F0": tmp_txt, "F1":tmp_map, "F2":ldd_map})

		# read sampling frequency for catchment from sampling_frequency.map
		pcrasterCommand(pcrcalc + " 'F0 = if(defined(F1) then F2)'", {"F0":tmp2_map,"F1":tmp_map, "F2":sampling_frequency_map})
		pcrasterCommand(map2col + " F0 F1"  , {"F0": tmp2_map, "F1":tmp2_txt})
		with open(tmp2_txt,"r") as f:
			for line in f.readlines():
				(X, Y, value) = line.split()
				stationdata.loc[index,'SamplingFrequency'] = float(value)
			f.close()

	tmp3_map = os.path.join(path_temp,"tmp3.map")
	inlets_map = os.path.join(path_result,"inlets.map")
	pcrasterCommand(pcrcalc + " 'F0 = scalar(F1)*0'", {"F0":inlets_map,"F1":ldd_map})
	f2 = open(direct_links_csv,"w")
	f2.write("ID,IDs of directly connected nested subcatchments,,,,,,,,,,,,,,,,,,,\n")
	for index, row in stationdata.iterrows():
		sys.stdout.write(".")

		# find subcatchments
		catchment_map = os.path.join(path_temp,"catchmask%05d.map" % float(index))
		pcrasterCommand(pcrcalc + " 'F0 = if(defined(F1) then F2)'", {"F0":tmp2_map,"F1":catchment_map,"F2":station_map})
		pcrasterCommand(map2col + " F0 F1"  , {"F0": tmp2_map, "F1":tmp_txt})
		text2 = str(index) # directly connected subcatchments
		f3 = open(tmp_txt,"r")

		commas = 20

		# loop over subcatchments
		for line in f3.readlines():
			(X,Y,subcatchment) = line.split()
			subcatchment = int(subcatchment)

			# check if directly connected subcatchment
			# if SamplingFrequency of subcatchment is 1 higher than catchment, 
			# then they are directly connected
			if int(row['SamplingFrequency']+1) == int(stationdata.loc[subcatchment,'SamplingFrequency']):
				text2 += ","+str(subcatchment)
				commas -= 1

				# inflow location is one pixel downstream
				pcrasterCommand(pcrcalc + " 'F0 = cover(F1,0) eq "+str(subcatchment)+"'", {"F0":tmp_map,"F1":station_map}) # select directly connected stations
				pcrasterCommand(pcrcalc + " 'F0 = upstream(F1,scalar(F2))*"+str(subcatchment)+"'", {"F0": tmp2_map, "F1":ldd_map, "F2":tmp_map}) # move directly connected stations 1 pixel downstream

				# check if inlet overlaps with other another inlet, if yes then move 1 more pixel downstream
				# TEST THIS!!!
				value = str(2)
				while float(value) > 1:
					pcrasterCommand(pcrcalc + " 'F0 = scalar(F1!=0)+scalar(F2!=0)'", {"F0": tmp3_map, "F1":inlets_map, "F2":tmp2_map})
					pcrasterCommand(pcrcalc + " 'F0 = if(F0!=0 then F0)'", {"F0": tmp3_map}) # replace 0 with mv
					pcrasterCommand(map2col + " F0 F1", {"F0": tmp3_map, "F1":tmp_txt}) # output pixel value to text file
					with open(tmp_txt,"r") as f:
						for line in f.readlines():
							(X,Y,value) = line.split()
							if float(value) > 1: # this means there is overlap, therefore move inlet further downstream
								pcrasterCommand(pcrcalc + " 'F0 = upstream(F1,scalar(F2))*"+str(subcatchment)+"'", {"F0": tmp3_map, "F1":ldd_map, "F2":tmp2_map}) # move directly connected stations 1 pixel downstream
								pcrasterCommand(pcrcalc + " 'F0 = F1'", {"F0": tmp2_map, "F1":tmp3_map})
						f.close()

				pcrasterCommand(pcrcalc + " 'F0 = F0+F1'", {"F0": inlets_map,"F1": tmp2_map}) # add inlet to inlets map
				pcrasterCommand(pcrcalc + " 'F0 = if(F0!=0 then F0)'", {"F0": tmp2_map}) # replace 0 with mv
				pcrasterCommand(map2col + " F0 F1", {"F0": tmp2_map, "F1":tmp_txt}) # output pixel location to text file
				with open(tmp_txt,"r") as f:
					for line in f.readlines():
						(X,Y,value) = line.split()
					f.close()
				#text2 += ":%f;%f"%(float(X),float(Y))
		f3.close()
		for cc in range(commas): # Add commas because the number of commas must be the same for each
			text2 += ","
		text2 += "\n"
		f2.write(text2)
	f2.close()	
	