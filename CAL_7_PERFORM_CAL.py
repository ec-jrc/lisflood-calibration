# -*- coding: utf-8 -*-
"""Please refer to quick_guide.pdf for usage instructions"""
import gdal
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
from pcrasterCommand import pcrasterCommand, getPCrasterPath
import glob
import datetime
import subprocess
import traceback


#@profile    
def main():
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

    path_result = parser.get('Path', 'Result')

    SubCatchmentPath = parser.get('Path','SubCatchmentPath')

    pcraster_path = parser.get('Path', 'PCRHOME')

    ForcingStart = datetime.datetime.strptime(parser.get('DEFAULT','ForcingStart'),"%d/%m/%Y %H:%M")  # Start of forcing
    ForcingEnd = datetime.datetime.strptime(parser.get('DEFAULT','ForcingEnd'),"%d/%m/%Y %H:%M")  # Start of forcing

    Root = parser.get('DEFAULT', 'Root')
    nmax = int(parser.get('DEFAULT','No_of_calibration_lists'))

    python_cmd = parser.get('Path', 'PYTHONCMD')

    config = {}
    for execname in ["pcrcalc","map2asc","asc2map","col2map","map2col","mapattr","resample","readmap"]:
        config[execname] = getPCrasterPath(pcraster_path, iniFile, alias=execname)

    pcrcalc = config["pcrcalc"]
    col2map = config["col2map"]
    map2col = config["map2col"]
    resample = config["resample"]
    readmap = config["readmap"]


    ########################################################################
    #   Loop through catchments and perform calibration
    ########################################################################

    print(">> Reading Qmeta2.csv file...")
    stationdata = pandas.read_csv(os.path.join(path_result,"Qmeta2.csv"),sep=",",index_col=0)
    stationdata_sorted = stationdata.sort_values(by=['CatchmentArea'],ascending=True)

    CatchmentsToProcess = pandas.read_csv(file_CatchmentsToProcess,sep=",",header=None)

    for index, row in stationdata_sorted.iterrows():
        
        Series = CatchmentsToProcess.loc[:,0]
        #print('cal_start',row['Cal_Start'])
        if len(Series[Series==index]) == 0: # Only process catchments whose ObsID is in the CatchmentsToProcess.txt file
            continue
        print("=================== "+str(index)+" ====================")
        path_subcatch = os.path.join(SubCatchmentPath,str(index))
        if os.path.exists(os.path.join(path_subcatch,"streamflow_simulated_best.csv")):
            print("streamflow_simulated_best.csv already exists! Moving on...")
            continue
        print(">> Starting calibration of catchment "+str(index)+", size "+str(row['CatchmentArea'])+" pixels...")

        # For some reason this version of LISFLOOD doesn't work with outlet map,
        # hence have to supply gauge coordinates
        gaugeloc_txt = os.path.join(path_subcatch,"maps","gaugeloc.txt")
        pcrasterCommand(map2col + " F0 F1"  , {"F0": os.path.join(path_subcatch,"maps","outlet.map"), "F1":gaugeloc_txt})

        # Copy simulated streamflow from upstream catchments
        # Change inlet map by replacing the numeric ID's with 1, 2, ...
        sys.stdout.write("Upstream station(s): ")
        direct_links = pandas.read_csv(os.path.join(path_result,"direct_links.csv"),sep=",",index_col=0)
        #inflow_tss is created according to the cal_start cal_end parameyters, script removes steps before and after and it reindex the steps
        
        inflow_tss = os.path.join(path_subcatch,"inflow","chanq.tss")
        #inflow_tss_lastrun is for when after the optimal combination of parameters is found , when we run the full forcing period
        inflow_tss_last_run = os.path.join(path_subcatch,"inflow","chanq_last_run.tss")
        
        try: del big_one
        except: pass
        try: 
            os.remove(inflow_tss)
            os.remove(inflow_tss_last_run)
        except: pass
        a = [int(i) for i in direct_links.loc[index].values if not np.isnan(i)]
        cnt = 1
        subcatchinlets_map = os.path.join(path_subcatch,"inflow","inflow.map")
        subcatchinlets_cut_map = os.path.join(path_subcatch,"inflow","inflow_cut.map")
        smallsubcatchmask_map = os.path.join(path_subcatch,"maps","masksmall.map")
        
        header = ""
        for subcatchment in a:
            
            subcatchment = str(subcatchment)

            sys.stdout.write(subcatchment+" ")
                            
            Qsim_tss = os.path.join(SubCatchmentPath,subcatchment,"out","chanq_simulated_best.tss")
                    
            if not os.path.exists(Qsim_tss) or os.path.getsize(Qsim_tss) == 0:
                raise Exception("ERROR: Missing " + Qsim_tss)

            try:
                # DD The shift_time.days is not correctly read for 6-hourly. Using time stamps to make it timesteps invariant
                simulated_streamflow_tmp = pandas.read_csv(Qsim_tss, sep=r"\s+", index_col=False, skiprows=4, header=None, usecols=[1])
                simulated_streamflow_tmp.index = pandas.date_range(ForcingStart, periods=len(simulated_streamflow_tmp), freq='6H')
                # DD comment the following line if you want to make the inflow the complete period
                simulated_streamflow_tmp.index = [i+1 for i in range(len(simulated_streamflow_tmp))]
                simulated_streamflow_lastrun = pandas.read_csv(Qsim_tss,sep=r"\s+",index_col=0,skiprows=4,header=None)
            except:
                print("Could not find streamflow_simulated_best.tss for upstream catchment "+subcatchment+", hence cannot run this catchment...")
                raise Exception("Stopping...")
                    
            simulated_streamflow=simulated_streamflow_tmp
            print('got it')
            if cnt==1: 
                big_one = simulated_streamflow  # type: object
                big_one_lastrun = simulated_streamflow_lastrun
            else:
                big_one[str(cnt)] = simulated_streamflow.values
                big_one_lastrun[str(cnt)] = simulated_streamflow_lastrun.values
            cnt += 1
            header = header+subcatchment+"\n"

        # DD If the following commands give an error, then replace it with the proper method to cut pcraster maps without getting the error
        # In addition, there is no point in converting points to indices from 1 to 5 if they are later removed in inflow.py.
        # So instead, just clip the map with the original catchment numbers
        pcrasterCommand(resample + " --clone F2 F0 F1" , {"F0": subcatchinlets_map, "F1":subcatchinlets_cut_map, "F2":smallsubcatchmask_map})

        inflowflag = str(0)
        if ("big_one" in globals()) or ("big_one" in locals()):

            big_one_lastrun.to_csv(inflow_tss_last_run,sep=' ',header=False)
            f = open(inflow_tss_last_run,'r+')
            content = f.read()
            content = 'timeseries scalar\n'+str(cnt)+'\n'+'timestep\n'+header+content
            f.seek(0,0)
            f.write(content)
            f.close()
            
            big_one.to_csv(inflow_tss,sep=' ',header=False)
            f = open(inflow_tss,'r+')
            content = f.read()
            content = 'timeseries scalar\n'+str(cnt)+'\n'+'timestep\n'+header+content
            f.seek(0,0)
            f.write(content)
            f.close()
            
            inflowflag = str(1)
            
        else:
            sys.stdout.write("No upstream inflow needed\n")
        sys.stdout.write("\n")

        # Performing calibration with external call, to avoid multiprocessing problems
        try:
            import cal_single_objfun
            cal_single_objfun.main()
        except:
            traceback.print_exc()
            raise Exception("ERROR in CAL_7_PERFORM_CAL")



    print("==================== END ====================")

if __name__ == '__main__':
    main()

