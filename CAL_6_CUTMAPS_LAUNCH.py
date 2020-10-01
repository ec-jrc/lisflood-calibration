import os
import sys
from pcraster import *
import pdb
import time
import struct
ver = sys.version
ver = ver[:ver.find('(')-1]
if ver.find('3.') > -1:
  from configparser import ConfigParser # Python 3.8
else:
  from ConfigParser import SafeConfigParser # Python 2.7-15import shutil
from pcrasterCommand import pcrasterCommand, getPCrasterPath
from pcraster import *
from netCDF4 import Dataset
import subprocess
import binFileIO
import traceback

def printout(text):
  print( text)
  sys.stdout.flush()

iniFile = os.path.normpath(sys.argv[1])
file_CatchmentsToProcess = os.path.normpath(sys.argv[2])
print("=================== START ===================")
print(">> Reading settings file ("+sys.argv[1]+")...")

if ver.find('3.') > -1:
  parser = ConfigParser()  # python 3.8
else:
  parser = SafeConfigParser()  # python 2.7-15
parser.read(iniFile)

Root = parser.get('DEFAULT', 'Root')
python_cmd = parser.get('Path', 'PYTHONCMD')
sys.exit(0)
try:
    with open(file_CatchmentsToProcess, "r") as f:
        catchments = f.readlines()
        catchmentFiles = ""
        for ic in catchments:
            catchmentFile = Root + "/logs/EFASCalib/CatchmentsToProcess_" + str(ic.replace("\n", "")) + "s.txt"
            catchmentFiles += " " + catchmentFile
            if not os.path.isfile(catchmentFile) or os.path.getsize(catchmentFile) == 0:
                f = open(catchmentFile, 'w')
                f.write(ic)
                f.close()

    if len(catchments) > 1:
        cmd = "module load gnuparallel && time parallel --progress --verbose -S 8/nidhogg,8/elli,8/phoebus,8/cade,8/arawm \"module load pcraster/4.1.0 && module load gdal/2.4.0 && " + python_cmd + " " + Root + "/../src/" + "/CAL_6_CUT_MAPS.py " + sys.argv[1] + " {}\" ::: " + catchmentFiles
    else: 
        cmd = "module load pcraster/4.1.0 && module load gdal/2.4.0 && " + python_cmd + " " + Root + "/../src/" + "/CAL_6_CUT_MAPS.py " + sys.argv[1] + " " + catchmentFiles
    printout(">> Calling \""+cmd+"\"")

    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stdin=subprocess.PIPE, stderr=subprocess.STDOUT)
    cutmapsOut = p.communicate()[0]
    printout(cutmapsOut)
    p.wait()

    # Transform into numpy binary
    binFileIO.main(iniFile, file_CatchmentsToProcess)

except:
    traceback.print_exc()
    raise Exception("ERROR in binFileIO.py")
