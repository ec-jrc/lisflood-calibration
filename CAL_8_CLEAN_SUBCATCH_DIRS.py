# -*- coding: utf-8 -*-
import os
import sys
import pandas
import pdb
from ConfigParser import SafeConfigParser
import glob


########################################################################
#   Read settings file
########################################################################

iniFile = os.path.normpath(sys.argv[1])
(drive, path) = os.path.splitdrive(iniFile)
(path, fil)  = os.path.split(path)
print ">> Reading settings file ("+fil+")..."

file_CatchmentsToProcess = os.path.normpath(sys.argv[2])

parser = SafeConfigParser()
parser.read(iniFile)

path_result = parser.get('Path', 'Result')

SubCatchmentPath = parser.get('Path','SubCatchmentPath')

########################################################################
#   Loop through catchments and delete useless files
########################################################################

print ">> Reading Qgis2.csv file..."
stationdata = pandas.read_csv(os.path.join(path_result,"Qgis2.csv"),sep=",",index_col=0)
stationdata_sorted = stationdata.sort_values(by=['CatchmentArea'],ascending=True)

CatchmentsToProcess = pandas.read_csv(file_CatchmentsToProcess,sep=",",header=None)

for index, row in stationdata_sorted.iterrows():
	Series = CatchmentsToProcess.ix[:,0]
	if len(Series[Series==row["ObsID"]]) == 0: # Only process catchments whose ID is in the CatchmentsToProcess.txt file
		continue
	print "=================== "+str(row['ObsID'])+" ===================="
	path_subcatch = os.path.join(SubCatchmentPath,str(row['ObsID']))
	
	# Delete all .xml, .bat, .tmp, and .txt files created for the runs
	for filename in glob.glob(os.path.join(path_subcatch,"*.xml")):
		os.remove(filename)
	for filename in glob.glob(os.path.join(path_subcatch,"*.bat")):
		os.remove(filename)
	for filename in glob.glob(os.path.join(path_subcatch,"*.tmp")):
		os.remove(filename)
	for filename in glob.glob(os.path.join(path_subcatch,"*.txt")):
		os.remove(filename)
	for filename in glob.glob(os.path.join(path_subcatch,"out","lzavin*.map")):
		os.remove(filename)
	for filename in glob.glob(os.path.join(path_subcatch,"out","avgdis*.map")):
		os.remove(filename)
	for filename in glob.glob(os.path.join(path_subcatch,"out","lz*")):
		os.remove(filename)
	for filename in glob.glob(os.path.join(path_subcatch,"out","dis*.tss")):
		os.remove(filename)
