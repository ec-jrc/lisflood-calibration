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

#print iniFile
try:
    #f = open(Root+'/Cutmaps.sh','w')
    #print 'open'
    #cmd = python_cmd+' '+Root+"/../src/"+'/CAL_6_CUT_MAPS.py '+sys.argv[1]+" "+sys.argv[2]
    #f.write("#!/bin/ksh \n")
    ## f.write("export PYTHONPATH=$PYTHONPATH:/usr/local/apps/pcraster/4.0.1/python \n")
    ## f.write("export PATH=$PATH:/usr/local/apps/pcraster/4.0.1/python \n")
    #f.write("module load gdal \n")
    #f.write("module load gnuparallel \n")
    #f.write("module load pcraster \n")
    #f.write(cmd)
    #f.close()
    #os.system("chmod a+x " + Root + "/Cutmaps.sh")
    ##cmd="qsub -l nodes=1:ppn=32 -q long -N CUT_MAPS_calib "+Root+"/Cutmaps.sh" # Using qsub on HPC
    ##cmd="module load gnuparallel && parallel -S 8/elli,8/nidhogg,4/yorrick echo \"Number {}: Running on \`hostname\`\" ::: 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20" # Using gnuparallel on multiple hosts
    #cmd="parallel -S 8/cicero,8/nidhogg,8/ariadne,8/elli "+Root+"/Cutmaps.sh ::: 1" # Using gnuparallel on multiple hosts

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
        cmd = "module load gnuparallel && time parallel --progress --verbose -S 2/nidhogg,2/elli,2/phoebus,2/cade,2/arawm \"module load pcraster/4.1.0 && module load gdal/2.4.0 && " + python_cmd + " " + Root + "/../src/" + "/CAL_6_CUT_MAPS.py " + sys.argv[1] + " {}\" ::: " + catchmentFiles
    else: 
        cmd = "module load pcraster/4.1.0 && module load gdal/2.4.0 && " + python_cmd + " " + Root + "/../src/" + "/CAL_6_CUT_MAPS.py " + sys.argv[1] + " " + catchmentFiles
    printout(">> Calling \""+cmd+"\"")

    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stdin=subprocess.PIPE, stderr=subprocess.STDOUT)
    cutmapsOut = p.communicate()[0] #.split()[-1]
    printout(cutmapsOut)
    p.wait()

    # Transform into numpy binary
    binFileIO.main(iniFile, file_CatchmentsToProcess)

except:
    traceback.print_exc()
    raise Exception("ERROR in binFileIO.py")

#cmd = ' '.join['qsub -q long -N cutmaps',python_path,'CAL_6_CUT_MAPS.py',iniFile,file_CatchmentsToProcess]

#' '.join(['gdal_translate -projwin',str(x_ul),str(y_ul),str(x_lr),str(y_lr),'-projwin_srs EPSG:4326 -of netCDF  --config GDAL_CACHEMAX 512 -co WRITE_BOTTOMUP=NO'])
#                fullCmd=' '.join([cmd,filenc,fileout])
#print cmd

    #            
#subprocess.Popen(fullCmd,shell=True)
