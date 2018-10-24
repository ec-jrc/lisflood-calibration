import os
import sys
from pcraster import *
import pdb
import time
import struct
from ConfigParser import SafeConfigParser
import shutil
from pcrasterCommand import pcrasterCommand, getPCrasterPath
from pcraster import *
from netCDF4 import Dataset
import subprocess



iniFile = os.path.normpath(sys.argv[1])
file_CatchmentsToProcess = os.path.normpath(sys.argv[2])
print "=================== START ==================="
print ">> Reading settings file ("+sys.argv[1]+")..."

parser = SafeConfigParser()
parser.read(iniFile)

Root = parser.get('DEFAULT', 'Root')
python_cmd = parser.get('Path', 'PYTHONCMD')

#print iniFile
try:
    f=open(Root+'/Cutmaps.sh','w')
    print 'open'
    cmd = python_cmd+' '+Root+'/CAL_6_CUT_MAPS.py '+sys.argv[1]+" "+sys.argv[2]
    f.write("#!/bin/sh \n")
    f.write("module load gdal \n")
    f.write(cmd)
    f.close()
    cmd="qsub -l nodes=1:ppn=32 -q long -N CUT_MAPS_calib "+Root+"/Cutmaps.sh"
    print ">> Calling \""+cmd+"\""
    os.system(cmd)
except:
    print "Empty or wrongly formatted Cal_Start, skipping..."
    #continue

#cmd = ' '.join['qsub -q long -N cutmaps',python_path,'CAL_6_CUT_MAPS.py',iniFile,file_CatchmentsToProcess]

#' '.join(['gdal_translate -projwin',str(x_ul),str(y_ul),str(x_lr),str(y_lr),'-projwin_srs EPSG:4326 -of netCDF  --config GDAL_CACHEMAX 512 -co WRITE_BOTTOMUP=NO'])
#                fullCmd=' '.join([cmd,filenc,fileout])
#print cmd

    #            
#subprocess.Popen(fullCmd,shell=True)