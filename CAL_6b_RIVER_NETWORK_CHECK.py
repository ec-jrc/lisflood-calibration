from osgeo import gdal
import pandas  as pd
import xarray as xr
import numpy as np
import pcraster as pcr
import os
import sys
import warnings
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
import binFileIO
import subprocess
from shutil import copyfile
from pcrasterCommand import pcrasterCommand, getPCrasterPath
from rasterConversions import getMapAttributes, any2PCRaster

########################################################################
#   Read settings file
########################################################################
iniFile = os.path.normpath(sys.argv[1])
print("=================== START ===================")
print(">> Reading settings file ("+sys.argv[1]+")...")

if ver.find('3.') > -1:
  parser = ConfigParser()  # python 3.8
else:
  parser = SafeConfigParser()  # python 2.7-15
parser.read(iniFile)

path_maps = parser.get("Path", "CatchmentDataPath") + "../"

CatchmentDataPath = parser.get('Path','CatchmentDataPath')
SubCatchmentPath = parser.get('Path','SubCatchmentPath')

pcraster_path = parser.get('Path', 'PCRHOME')

config = {}
for execname in ["pcrcalc","map2asc","asc2map","col2map","map2col","mapattr","resample"]:
    config[execname] = getPCrasterPath(pcraster_path,sys.argv[1],execname)

pcrcalc = config["pcrcalc"]
col2map = config["col2map"]
map2col = config["map2col"]
resample = config["resample"]

path_MeteoData = parser.get('Path', 'MeteoData')
path_result = parser.get('Path', 'Result')
switch_SubsetMeteoData = int(parser.get('DEFAULT', 'SubsetMeteoData'))

# Extract all station locations
print(">> Reading Qmeta2.csv file...")
stationdata = pd.read_csv(os.path.join(path_result,"Qmeta2.csv"),sep=",",index_col='ObsID')
stationdata['ObsID'] = stationdata.index
stationLocations = stationdata[["ObsID","LisfloodX","LisfloodY"]].values.tolist()

root = parser.get('DEFAULT', 'Root')
catchments = root + "/catchments"
asciiGraphs = root + "/result/asciiGraphs"

# Locate all inflow.map files
cmd = "find " + catchments + " -name 'inflow.map' | sort"
p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stdin=subprocess.PIPE, stderr=subprocess.STDOUT)
inflows = p.communicate()[0].decode('utf-8').split("\n")
p.wait()

# Locate all the ascii graphs
cmd = "find " + asciiGraphs + " -name '*.txt' | sort"
p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stdin=subprocess.PIPE, stderr=subprocess.STDOUT)
fullCatchments = p.communicate()[0].decode('utf-8').split("\n")
p.wait()

for inflow in inflows:
  if inflow != "":
    # Derive catchment number
    catchment = int(os.path.basename(inflow.replace("/inflow/inflow.map", "")))

    # Check coordinates
    # Extract the coordinates of the map's pixels
    p = subprocess.Popen("map2col \"" + inflow + "\" inflowMapCoordinates.txt && cat inflowMapCoordinates.txt && rm -rf inflowMapCoordinates.txt", shell=True, stdout=subprocess.PIPE, stdin=subprocess.PIPE, stderr=subprocess.STDOUT)
    buffer = p.communicate()[0].decode('utf-8').split("\n")
    p.wait()
    inflowMapCoordinates = []
    for i in buffer[1:-1]:
      line = i.split(" ")
      inflowMapCoordinates.append((int(float(line[2])), int(float(line[4]))))
    # Attempt to find the station locations in the locations file. They should all exist
    try:
      stationLocation = [(int(i[1]), int(i[2])) for i in stationLocations if i[0] == catchment][0]
    except:
      raise Exception("OOPS")
    if not stationLocation in inflowMapCoordinates:
      warnings.warn("Mismatched inflow: " + str(catchment))

    # Check inflow points
    inflowPoints = []
    for ic in fullCatchments:
      if ic != "":
        links = open(ic, "r").readlines()
        inflowers = [int(re.sub("\t.*\n", "", re.sub("\t.*-> ", "",  i)))  for i in links if i.find(" -> ") > -1]
        inflowers = set(inflowers)
        inflowers = sorted(inflowers)
        if len(inflowers) > 0:
          # Now look up the catchment number in the inflow map file
          inflowMap = pcr.readmap(str(inflow))
          inflowArray = pcr.pcr2numpy(inflowMap, np.nan)
          inflowArray = inflowArray[~np.isnan(inflowArray)]
          inflowArrayFiltered = set([int(i) for i in inflowArray if i > 0])
          inflowArrayFiltered = sorted(inflowArrayFiltered)
          if sorted(set(inflowers)) != inflowArrayFiltered:
            warnings.warn(
              "ERROR in catchment " \
              + str(catchment) + ": " \
              + str(inflowers) + " VS " \
              + str(inflowArrayFiltered) \
            )
            #   os.system("aguila ${f} &")
            print('\n')
        else:
          continue

      # # Same but for very large ascii files. The memory mapping is faster
      # f = open(ic)
      # s = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
      # if s.find(" " + str(catchment) + " -> ") != -1:
      #   print(str(s))
      # f.close()

    # Check coordinates in the cut map too
    inflowCut = str(inflow).replace(".map", "_cut.map")
    if os.path.exists(inflowCut) and os.path.getsize(inflowCut) > 0:
      p = subprocess.Popen("module load pcraster && map2col \"" + inflowCut + "\" inflowMapCoordinates.txt && cat inflowMapCoordinates.txt && rm -rf inflowMapCoordinates.txt", shell=True, stdout=subprocess.PIPE, stdin=subprocess.PIPE, stderr=subprocess.STDOUT)
      buffer = p.communicate()[0].decode('utf-8').split("\n")
      p.wait()
      inflowMapCoordinatesCut = []
      for i in buffer[2:-1]:
        line = i.split(" ")
        inflowMapCoordinatesCut.append((int(float(line[2])), int(float(line[4]))))
      # Attempt to find the station locations in the locations file. They should all exist
      try:
        stationLocation = [(int(i[1]), int(i[2])) for i in stationLocations if i[0] == catchment][0]
      except:
        raise Exception("OOPS")
      if not stationLocation in inflowMapCoordinatesCut:
        warnings.warn("Station not located in inflow_cut map: " + str(catchment))
      # And chek that the found coordinates match the uncut map
      if inflowMapCoordinates != inflowMapCoordinatesCut:
        warnings.warn("Mismatched inflow and inflow_cut: " + str(catchment))
      # # also check that the original and cut map have the same amount of non-nan points
      # DD Doesn't work because the inflowMapCut is corrupt and polluted by random floats
      # mask = pcr.setclone(inflowCut)
      # inflowMapCut = pcr.readmap(inflowCut)
      # inflowArrayCut = np.round(pcr.pcr2numpy(inflowMapCut, np.nan))
      # inflowArrayCut = inflowArrayCut[~np.isnan(inflowArrayCut)]
      # if len(inflowArray) != len(inflowArrayCut):
      #   print("Mismatched inflow map and its cut version: " + str(catchment))

    # Do the same for other maps in the map folder, but not outlets as they have only a single pixel in them anyway
    path_subcatch = os.path.join(SubCatchmentPath, str(catchment))
    path_subcatch_maps = os.path.join(path_subcatch, 'maps')
    cmd = "find " + path_subcatch_maps + " -name \"*.map\" -not -name \"*outlet*.map\""
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stdin=subprocess.PIPE, stderr=subprocess.STDOUT)
    pcrastermaps = p.communicate()[0].decode('utf-8').split("\n")
    p.wait()

    for b in pcrastermaps:
      if os.path.isfile(b) and os.path.getsize(b) > 0:
        p = subprocess.Popen("map2col \"" + b + "\" inflowMapCoordinates.txt && cat inflowMapCoordinates.txt && rm -rf inflowMapCoordinates.txt", shell=True, stdout=subprocess.PIPE, stdin=subprocess.PIPE, stderr=subprocess.STDOUT)
        buffer = p.communicate()[0].decode('utf-8').split("\n")
        p.wait()
        coordinates = []
        for i in buffer[1:-1]:
          line = i.split(" ")
          coordinates.append((int(float(line[2])), int(float(line[4]))))
        if inflowMapCoordinates != coordinates:
          warnings.warn("Mismatched inflow and " + b + " map: " + str(catchment))





