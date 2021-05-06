# -*- coding: utf-8 -*-
"""Please refer to quick_guide.pdf for usage instructions"""

import os
import sys
#import numpy as np
import pandas
#import re
import pdb
#import time
ver = sys.version
ver = ver[:ver.find('(')-1]
if ver.find('3.') > -1:
  from configparser import ConfigParser # Python 3.8
else:
  from ConfigParser import SafeConfigParser # Python 2.7-15
import glob
#import datetime


########################################################################
#   Read settings file
########################################################################

iniFile = os.path.normpath(sys.argv[1])
(drive, path) = os.path.splitdrive(iniFile)
(path, fil)  = os.path.split(path)
print(">> Reading settings file ("+fil+")...")

file_CatchmentsToProcess = os.path.normpath(sys.argv[2])

if ver == '3.6.8':
	parser = ConfigParser()  # python 3.8
else:
	parser = SafeConfigParser()  # python 2.7-15
parser.read(iniFile)

path_result = parser.get('Path', 'Result')

SubCatchmentPath = parser.get('Path','SubCatchmentPath')

# Path to calibration run without GwLoss
#version1Path = "/vol/floods/nedd/EFASCalib/backup/backupRerun20190827WideParams/output/catchments"
version1Path = "/vol/floods/nedd/EFASCalib/backup/20200331GwLossZeroNoWateruse/output/catchments"
# Path to calibration run with GwLoss
#version2Path = "/vol/floods/nedd/EFASCalib/backup/backupRerun20191204GwLossComplete/output/catchments"
version2Path = "/vol/floods/nedd/EFASCalib/backup/20200331GwLossNoWateruse/output/catchments"

########################################################################
#   Loop through catchments and delete useless files
########################################################################

print(">> Reading Qmeta2.csv file...")
stationdata = pandas.read_csv(os.path.join(path_result,"Qmeta2.csv"),sep=",",index_col=0)
stationdata_sorted = stationdata.sort_values(by=['CatchmentArea'],ascending=True)

CatchmentsToProcess = pandas.read_csv(file_CatchmentsToProcess,sep=",",header=None)

for index, row in stationdata_sorted.iterrows():
	Series = CatchmentsToProcess.iloc[:,0]
	path_subcatch = os.path.join(SubCatchmentPath, str(row['ObsID']))

	print("=================== " + str(row['ObsID']) + " ====================")
	if len(Series[Series==row["ObsID"]]) == 0: # Only process catchments whose ID is in the CatchmentsToProcess.txt file
		# Make a link to version 1 output
		os.system("ln -fs %s %s" % (os.path.join(version1Path, str(row['ObsID'])), path_subcatch))
	else:
		# Make a link to version 2 output
		os.system("ln -fs %s %s" % (os.path.join(version2Path, str(row['ObsID'])), path_subcatch))
