# -*- coding: utf-8 -*-
"""Please refer to quick_guide.pdf for usage instructions"""

import os
import sys
import numpy as np
import pandas
import re
import pdb
import time
from datetime import datetime
ver = sys.version
ver = ver[:ver.find('(')-1]
if ver.find('3.') > -1:
  from configparser import ConfigParser # Python 3.8
else:
  from ConfigParser import SafeConfigParser # Python 2.7-15
import glob
import datetime
import subprocess
import random

# USAGE python CAL_7_PERFORM_CAL.py workflow_settings.txt CatchmentsToProcess_XX.txt

########################################################################
#   Read settings file
########################################################################

iniFile = os.path.normpath(sys.argv[1])

file_CatchmentsToProcess = os.path.normpath(sys.argv[2])

if ver.find('3.') > -1:
    parser = ConfigParser()  # python 3.8
else:
    parser = SafeConfigParser()  # python 2.7-15
parser.read(iniFile)

src_root = parser.get('Main', 'src_root')
nmax = int(parser.get('Main','No_of_calibration_nodes'))

stations_data_path = parser.get("Stations", "stations_data")
stations_links_path = parser.get('Stations', 'stations_links')

SubCatchmentPath = parser.get('Path','subcatchment_path')
numCPUs = parser.get('DEAP','numCPUs')

python_cmd = parser.get('Path', 'PYTHONCMD')


########################################################################
#   Loop through catchments and perform calibration
########################################################################

print(">> Reading stations_data file...")
stationdata = pandas.read_csv(stations_data_path, sep=",", index_col='ObsID')
stationdata_sorted = stationdata.sort_values(by=['DrainingArea.km2.LDD'],ascending=True)

CatchmentsToProcess = pandas.read_csv(file_CatchmentsToProcess,sep=",",header=None)

for index, row in stationdata_sorted.iterrows():
    catchment = index

    Series = CatchmentsToProcess[0]
    if len(Series[Series==catchment]) == 0: # Only process catchments whose ID is in the CatchmentsToProcess.txt file
        continue
    print("=================== "+str(catchment)+" ====================")
    path_subcatch = os.path.join(SubCatchmentPath,str(catchment))
    if os.path.exists(os.path.join(path_subcatch,"out","streamflow_simulated_best.csv")):
        print("streamflow_simulated_best.csv already exists! Moving on...")
        continue
    print(">> Starting calibration of catchment "+str(catchment)+", size "+str(row['DrainingArea.km2.LDD'])+" km2...")

    # Copy simulated streamflow from upstream catchments
    # Change inlet map by replacing the numeric ID's with 1, 2, ...
    print("Upstream station(s): ")
    stations_links = pandas.read_csv(stations_links_path,sep=",",index_col=0)

    subcatchment_list = [int(i) for i in stations_links.loc[catchment].values if not np.isnan(i)]

    for subcatchment in subcatchment_list: 
        subcatchment = str(subcatchment)
        print(subcatchment+" ")
                    
        Qsim_tss = os.path.join(SubCatchmentPath,subcatchment,"out","chanq_simulated_best.tss")
        
        #loop here till previous catchment on the list is done
        timer = 0
        while not os.path.exists(Qsim_tss) and timer<=720000:
            time.sleep(1)
            timer+=1
                            
        print('got it')
    print("\n")
    # Performing calibration with external call, to avoid multiprocessing problems
    try:
        sbc=str(catchment)
        catch=file_CatchmentsToProcess[-6:-4]
        # create sh scripts in scripts folder (scripts folder should already exists)
        path_scripts = os.path.join(src_root,'scripts')
        if not os.path.exists(path_scripts):
            print('Error: folder ' + path_scripts + ' not found')
            raise Exception('Error: folder ' + path_scripts + ' not found')

        # use different cache paths for numba compiled binaries for each catchments, to reduce chances of overlapped lock files 
        path_numba_cache_dirs = os.path.join(src_root,'numba_cache_dirs')
        if not os.path.exists(path_numba_cache_dirs):
            print('Error: folder ' + path_numba_cache_dirs + ' not found')
            raise Exception('Error: folder ' + path_numba_cache_dirs + ' not found')
        
        # max numbers of subfolders to use
        max_cache_subfolders = 60
        path_current_numba_cache_dir = os.path.join(path_numba_cache_dirs,str(int(random.random()*max_cache_subfolders)))
        if not os.path.exists(path_current_numba_cache_dir):
            os.mkdir(path_current_numba_cache_dir)
 
        script_name=os.path.join(path_scripts,'runLF_' +catch+'_'+sbc+'.sh')

        f=open(script_name,'w')
        f.write("#!/bin/sh \n")
        f.write("source activate liscal \n")
        f.write("export NUMBA_THREADING_LAYER='tbb' \n")
        f.write("export NUMBA_NUM_THREADS=1 \n")
        f.write("export NUMBA_CACHE_DIR=\"" + path_current_numba_cache_dir + "\" \n")
        cmd = python_cmd+' '+ os.path.join(src_root,'bin/CAL_7A_CALIBRATION.py') + ' '+ os.path.join(SubCatchmentPath,str(index),'settings.txt') + ' ' + str(index) + ' ' + str(numCPUs) + '\n'
        f.write(cmd)
        cmd = python_cmd+' '+ os.path.join(src_root,'bin/CAL_7B_LONGTERM_RUN.py') + ' '+ os.path.join(SubCatchmentPath,str(index),'settings.txt') + ' ' + str(index) + '\n'
        f.write(cmd)
        f.close()
        cmd="qsub -l nodes=1:ppn=32 -q long -N LF_cal_"+catch+"_"+sbc+" "+script_name
        
        timerqsub = 0
        
        while int(subprocess.Popen('qstat | grep LF_cal_ | wc -l',shell=True,stdout=subprocess.PIPE).stdout.read()) >= int(nmax) and timerqsub<=900000:
            #print 'submitted',int(subprocess.Popen('qstat | grep LF_calib | wc -l',shell=True,stdout=subprocess.PIPE).stdout.read())
            rand_time = int(random.random()*10)
            time.sleep(rand_time)
            timerqsub+=rand_time
        
        if timerqsub>900000:
            print('3 days waiting for job submission, something is wrong')
            raise Exception('too much time')
        
        print(">> Calling \""+cmd+"\"")
        os.system(cmd)
        
        #wait random time to let other queues to access nodes
        rand_time = int(random.random()*5)
        time.sleep(rand_time)
    except:
        print("Something went wrong with queue submission skipping...")
        continue

print("==================== END ====================")
