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
ver = sys.version
ver = ver[:ver.find('(')-1]
if ver.find('3.') > -1:
  from configparser import ConfigParser # Python 3.8
else:
  from ConfigParser import SafeConfigParser # Python 2.7-15
import array
import logging
import random

if __name__=="__main__":

	########################################################################
	#   Read settings file
	########################################################################
	
	iniFile = os.path.normpath(sys.argv[1])

	print( "=================== START ===================")
	print( ">> Reading settings file ("+sys.argv[1]+")...")

	if ver.find('3.') > -1:
		parser = ConfigParser()  # python 3.8
	else:
		parser = SafeConfigParser()  # python 2.7-15
	parser.read(iniFile)

	path_result = parser.get('Path', 'Result')	
	if os.path.exists(path_result)==False:
		os.mkdir(path_result)

	MinQlength = int(parser.get('DEFAULT', 'MinQlength'))
	WarmupDays = int(parser.get('DEFAULT', 'WarmupDays'))

	ObservationsStart = datetime.datetime.strptime(parser.get('DEFAULT', 'ObservationsStart'), "%d/%m/%Y %H:%M")  # Start of forcing
	ObservationsEnd = datetime.datetime.strptime(parser.get('DEFAULT', 'ObservationsEnd'), "%d/%m/%Y %H:%M")  # Start of forcing
	ForcingStart = datetime.datetime.strptime(parser.get('DEFAULT','ForcingStart'),"%d/%m/%Y %H:%M")  # Start of forcing
	ForcingEnd = datetime.datetime.strptime(parser.get('DEFAULT','ForcingEnd'),"%d/%m/%Y %H:%M")  # Start of forcing

	Qtss_csv = parser.get('CSV', 'Qtss')
	Qmeta_csv = parser.get('CSV', 'Qmeta')


	########################################################################
	#   Make stationdata array from the Qmeta csv
	########################################################################

	(drive, path) = os.path.splitdrive(Qmeta_csv)
	(path, fil)  = os.path.split(path)
	print( ">> Reading Qmeta file ("+fil+")...")
	stationdata = pandas.read_csv(Qmeta_csv,sep=",",index_col='ObsID')
	# stationdata = stationdata[stationdata['EC_calib']==3] # for EFAS4.1
	#stationdata = stationdata[np.logical_and(stationdata['EC_calib'] == 3, np.logical_or(stationdata['Country code'] == 'SE', stationdata['Country code'] == 'NO'))] # for test powerPrefFlow in Scandinavia
	# stationdata = stationdata[np.logical_or(stationdata['EC_calib'] == '3', stationdata['EC_calib'] == '4')] # fix of powerPrefFlow EFAS4.1
	stationdata = stationdata[stationdata['EC_calib'] == 'T'] # to test calib
	# stationdata.index = stationdata['ObsID']


	########################################################################
	#   Compute start and end of calibration and validation periods
	########################################################################

	sys.stdout.write(">> Compute calibration and validation periods")

	stationdata['EnoughQdata'] = np.nan
	stationdata['Val_Start'] = np.nan
	stationdata['Val_End'] = np.nan
	stationdata['Cal_Start'] = np.nan
	stationdata['Cal_End'] = np.nan

	streamflow_data = pandas.read_csv(Qtss_csv, sep=",", index_col=0)
	streamflow_data.index = pandas.date_range(start=ObservationsStart, end=ObservationsEnd, periods=len(streamflow_data))

	for index, row in stationdata.iterrows():
      
		# Retrieve observed streamflow
		observed_streamflow = streamflow_data[str(index)]
		print(np.sum(np.isnan(observed_streamflow)))

		# For which period do we have forcing and observed streamflow?
		mask = observed_streamflow.isnull()
		ForcingStart_inobs = mask.index.searchsorted(ForcingStart)
		ForcingEnd_inobs = mask.index.searchsorted(ForcingEnd)
		mask[:ForcingStart_inobs] = True
		mask[ForcingEnd_inobs+1:] = True

		# If record is twice as long as MinQlength, then split the record in equally
		# long calibration and validation periods. However, if the record is between
		# MinQlength and 2*MinQlength long, then take MinQlength for the calibration
		# period and the remainder for the validation period. If record is shorther than
		# MinQlength, reject the catchment.
		mask_cumsum = np.cumsum((mask.values-1)*-1+0) # cumulative sum counting number of days with non-missing values
		total_number_vals =  np.max(mask_cumsum)
		# DD since we have 6-hourly data, the calibration period is not 365 days, but 4x that. However, data is very sparse,
		# so we decided internally to not have a validation period when data is 6-hourly.
		# In addition, we relax the MinQlength = 4 years to 85% of that to include all the stations that were manually filtered by C.Mazzetti
		# in preparation of the calibration.
		stationdata.loc[index, 'EnoughQdata'] = 1
		if str(row['CAL_TYPE']).find("_6h") > -1:
			SplitFraction = 0.0
			if str(row['CAL_TYPE']).find("NRT_6h") > -1:
				obsEnd = stationdata.loc[index, 'EndDate_nrt_6']
			elif str(row['CAL_TYPE']).find("HIST_6h") > -1:
				obsEnd = stationdata.loc[index, 'EndDate_hist_6']
		else:
			if str(row['CAL_TYPE']).find("NRT_24h") > -1:
				obsEnd = stationdata.loc[index, 'EndDate_nrt_24']
			elif str(row['CAL_TYPE']).find("HIST_24h") > -1:
				obsEnd = stationdata.loc[index, 'EndDate_hist_24']
			if total_number_vals <= (365.0 * MinQlength * 0.85):
				#sys.stdout.write(row['ObsID']+': streamflow record too short, setting EnoughQdata to 0')
				stationdata.loc[index,'EnoughQdata'] = 0
				print("\nCatchment: " + str(row['ObsID']))
				print("# valid obs days = " + str(total_number_vals))
				print("==> Deleted because # years < MinQlength")
				print("\n")
				stationdata.loc[index, 'Val_Start'] = 'Streamflow record too short'
				stationdata.loc[index, 'num obs steps found'] = total_number_vals
				continue
			# Splitfraction gives fraction of totvals that is used for validation
			SplitFraction = 1.0 - float(365.0 * MinQlength) / float(total_number_vals)
			# Make sure at least half of data is used for calibration
			if SplitFraction > 0.5:
				SplitFraction = 0.5
		# Make sure that the validation period is not smaller than a day
		if SplitFraction < 1.0 / total_number_vals:
			SplitFraction = 1.0 / total_number_vals
		# Find the index where we have accumulated enough days to cover the validation period
		Split = np.where(mask_cumsum == np.round(total_number_vals*SplitFraction))[0]
		Split = np.asarray(Split)
		Split = Split[0]
		# Convert it into a time stamp - spinup days
		Split = mask.index[Split] - datetime.timedelta(days=WarmupDays)
		# Make sure it doesn't start before the forcing
		if Split < ForcingStart:
			Split = ForcingStart
		# Set the start as obs beginning
		Start = mask[mask == False].index[0]
		# Make sure it doesn't start before the forcing
		if Start < ForcingStart:
			Start = ForcingStart
		End = mask[mask == False].index[-1]
		print("\nCatchment: " + str(index))
		print("Calibration Type: " + str(row['CAL_TYPE']))
		print("# valid obs = " + str(total_number_vals) + " days (" + str(total_number_vals/365.0) + " yrs)")
		print("Forcings: " + str(ForcingStart) + " ~ " + str(ForcingEnd))
		print("Obs: " + str(observed_streamflow[observed_streamflow.notna()].index[0]) + " ~ " + str(observed_streamflow[observed_streamflow.notna()].index[-1]))
		print("Val (excl. spinup): " + str(Start) + " ~ " + str(str(np.int(Split.year)+int(WarmupDays/365.0))+"-01-01 06:00:00"))
		print("Cal (incl. spinup): " + str(str(np.int(Split.year))+"-01-02 06:00:00") + " ~ " + str(End))
		print("Start (excl. spinup): " + str(Start))
		print("Split (incl. spinup): " + str(Split))
		print("End: " + str(End))
		print("Validation period (excl. spinup) = " + str((datetime.datetime.strptime("01/01/" + str(np.int(Split.year)+int(WarmupDays/365.0)) + " 06:00", "%d/%m/%Y %H:%M") - Start).days / 365.0) + " yrs")
		print("Calibration period (excl. spinup + round(split)) = " + str((End - datetime.datetime.strptime("01/01/" + str(np.int(Split.year)+int(WarmupDays/365.0)) + " 06:00", "%d/%m/%Y %H:%M")).days / 365.0) + " yrs")
		sys.stdout.write("\n")

#		if (Split-Start).days>365:
		stationdata.loc[index,'Val_Start'] = Start.strftime("%d/%m/%Y %H:%M")
		stationdata.loc[index,'Val_End'] = "02/01/"+str(np.int(Split.year)+int(WarmupDays/365.0))+" 06:00" # Round to December 31st of previous year
#		else: DD Keep all stations because they have manually been validated
#			stationdata.loc[index,'Val_Start'] = "Streamflow record too short for validation"
#			stationdata.loc[index,'EnoughQdata'] = 0
			
		stationdata.loc[index,'Cal_Start'] = "02/01/"+str(np.int(Split.year))+" 06:00" # Round to January 1st of year
		stationdata.loc[index,'Cal_End'] = End.strftime("%d/%m/%Y %H:%M")

	if len(stationdata) == 0:
		raise Exception("ERROR: Not a single station has data for calibration.")

	# Save dataframe with new catchment area and cal val period columns
	print("\n>> Saving Qmeta2.csv with Cal_Start, Cal_End, Val_Start, and Val_End columns, includes only catchments with suitable==1 & EnoughQdata==1")
	stationdata.to_csv(os.path.join(path_result,"Qmeta2_noFilter.csv").replace("\\","/"),',')
	stationdata = stationdata[stationdata['EnoughQdata']==1]
	stationdata.to_csv(os.path.join(path_result,"Qmeta2.csv").replace("\\","/"),',')
	print("==================== END ====================")

