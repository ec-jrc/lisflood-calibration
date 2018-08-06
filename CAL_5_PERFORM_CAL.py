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
from ConfigParser import SafeConfigParser
from pcrasterCommand import pcrasterCommand, getPCrasterPath
import glob
import datetime


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

pcraster_path = parser.get('Path', 'PCRHOME')

forcing_start=parser.get('DEFAULT', 'ForcingStart')

config = {}
for execname in ["pcrcalc","map2asc","asc2map","col2map","map2col","mapattr","resample"]:
    config[execname] = getPCrasterPath(pcraster_path,execname)

pcrcalc = config["pcrcalc"]
col2map = config["col2map"]
map2col = config["map2col"]
resample = config["resample"]


########################################################################
#   Loop through catchments and perform calibration
########################################################################

print ">> Reading Qgis2.csv file..."
stationdata = pandas.read_csv(os.path.join(path_result,"Qgis2.csv"),sep=",",index_col=0)
stationdata_sorted = stationdata.sort_index(by=['CatchmentArea'],ascending=True)

CatchmentsToProcess = pandas.read_csv(file_CatchmentsToProcess,sep=",",header=None)

for index, row in stationdata_sorted.iterrows():
    Series = CatchmentsToProcess.ix[:,0]
    print 'cal_start',row['Cal_Start']
    if len(Series[Series==str(row["ID"])]) == 0: # Only process catchments whose ID is in the CatchmentsToProcess.txt file
        continue
    print "=================== "+row['ID']+" ===================="
    path_subcatch = os.path.join(SubCatchmentPath,row['ID'])
    if os.path.exists(os.path.join(path_subcatch,"streamflow_simulated_best.csv")):
        print "streamflow_simulated_best.csv already exists! Moving on..."
        continue
    print ">> Starting calibration of catchment "+row['ID']+", size "+str(row['CatchmentArea'])+" pixels..."

    # Delete all .csv files that were created after previous optimizations
    #for filename in glob.glob(os.path.join(path_subcatch,"*.csv")):
    #	os.remove(filename)

    # For some reason this version of LISFLOOD doesn't work with outlet map,
    # hence have to supply gauge coordinates
    gaugeloc_txt = os.path.join(path_subcatch,"maps","gaugeloc.txt")
    pcrasterCommand(map2col + " F0 F1"  , {"F0": os.path.join(path_subcatch,"maps","outlet.map"), "F1":gaugeloc_txt})

    # Copy simulated streamflow from upstream catchments
    # Change inlet map by replacing the numeric ID's with 1, 2, ...
    sys.stdout.write("Upstream station(s): ")
    direct_links = pandas.read_csv(os.path.join(path_result,"direct_links.csv"),sep=",",index_col=0)
    inflow_tss = os.path.join(path_subcatch,"inflow","inflow.tss")
    try: del big_one
    except: pass
    try: os.remove(inflow_tss)
    except: pass
    a = direct_links.loc[row['ID']].values
    cnt = 1
    subcatchinlets_map = os.path.join(path_subcatch,"inflow","inflow.map")
    subcatchinlets_new_map = os.path.join(path_subcatch,"inflow","inflow_new.map")
    subcatchinlets_new2_map = os.path.join(path_subcatch,"inflow","inflow_new2.map")
    subcatchinlets_new3_map = os.path.join(path_subcatch,"inflow","inflow_new3.map")
    smallsubcatchmask_map = os.path.join(path_subcatch,"maps","masksmall.map")
    pcrasterCommand(pcrcalc + " 'F0 = F1*0.0'", {"F0":subcatchinlets_new_map,"F1":subcatchinlets_map})
    header = ""
    for subcatchment in a:

        if type(subcatchment) is str:
            sys.stdout.write(subcatchment+" ")
                        
            Qsim_tss = os.path.join(SubCatchmentPath,subcatchment,"out","streamflow_simulated_best.tss")
                        #loop here till previous  catchment on the list is done
            timer = 0
            while not os.path.exists(Qsim_tss) and timer<=42000:
                time.sleep(1)
                timer+=1
            shift_time = datetime.datetime.strptime(row['Cal_Start'], "%d/%m/%Y %H:%M") - datetime.datetime.strptime(forcing_start, "%d/%m/%Y %H:%M")  # difference in days between forcingStart and the cal_start
            print 'inflow.tss'
            try: simulated_streamflow_tmp = pandas.read_table(Qsim_tss,sep=r"\s+",index_col=False,skiprows=4+shift_time.days-1,header=None,usecols=[1])
            except:
                print "Could not find streamflow_simulated_best.tss for upstream catchment "+subcatchment+", hence cannot run this catchment..."
                raise Exception("Stopping...")
            simulated_streamflow=simulated_streamflow_tmp[1:]
            print 'got it'
            if cnt==1: big_one = simulated_streamflow  # type: object
            else: big_one[str(cnt)] = simulated_streamflow.values
            numeric_only = re.compile(r'[^\d.]+')
            hhh = str(int(numeric_only.sub('',subcatchment)))
            pcrasterCommand(pcrcalc + " 'F0 = F0+scalar(F1=="+hhh+")*"+str(cnt)+"'", {"F0": subcatchinlets_new_map,"F1":subcatchinlets_map})
            cnt += 1
            header = header+subcatchment+"\n"

    pcrasterCommand(pcrcalc + " 'F1 = if(scalar(boolean(F0))>0,nominal(F0))'", {"F0": subcatchinlets_new_map,"F1": subcatchinlets_new2_map})
    pcrasterCommand(resample + " F0 F1 --clone F2 " , {"F0": subcatchinlets_new2_map, "F1":subcatchinlets_new3_map, "F2":smallsubcatchmask_map})
    #print "(note that despite memory error, inflow_new3.map is being created, strange...)"


    inflowflag = str(0)
    if ("big_one" in globals()) or ("big_one" in locals()):

        big_one.to_csv(inflow_tss,sep=' ',header=False)
        f = open(inflow_tss,'r+')
        content = f.read()
        content = 'timeseries scalar\n'+str(cnt)+'\n'+'timestep\n'+header+content
        f.seek(0,0)
        f.write(content)
        f.close()
        inflowflag = str(1)
    else:
        sys.stdout.write("none")
    sys.stdout.write("\n")

    # Performing calibration with external call, to avoid multiprocessing problems
    try:
        datetime.datetime.strptime(row['Cal_Start'],"%d/%m/%Y %H:%M")
        f=open('/FLOODS/lisflood/calibration_tool/runLF.sh','w')
        print 'open'
        cmd = "/ADAPTATION/usr/anaconda2/bin/python  /FLOODS/lisflood/calibration_tool/cal_single_objfun.py "+sys.argv[1]+" "+str(index)
        f.write("#!/bin/sh \n")
        f.write(cmd)
        f.close()
        cmd="qsub -l nodes=1:ppn=32 -q high -N LF_calib /FLOODS/lisflood/calibration_tool/runLF.sh"
        print ">> Calling \""+cmd+"\""
        os.system(cmd)
    except:
        print "Empty or wrongly formatted Cal_Start, skipping..."
        continue
    #if not row['Cal_Start']: # check if Cal_Start field of Qgis2.csv file is empty
    #    print "Empty Cal_Start, skipping..."
    #    continue


print "==================== END ===================="
