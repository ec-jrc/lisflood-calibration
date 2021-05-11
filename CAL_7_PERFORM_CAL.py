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
from configparser import ConfigParser # Python 3.8
from pcrasterCommand import pcrasterCommand, getPCrasterPath
import glob
import datetime
import subprocess
#from memory_profiler import profile
#from guppy import hpy
import traceback


class Config():

    def __init__(self, settings_file):

        parser = ConfigParser()
        parser.read(settings_file)

        self.path_result = parser.get('Path', 'Result')
        self.subcatchment_path = parser.get('Path','SubCatchmentPath')

        self.forcing_start = datetime.datetime.strptime(parser.get('DEFAULT','ForcingStart'),"%d/%m/%Y %H:%M")  # Start of forcing
        self.forcing_end = datetime.datetime.strptime(parser.get('DEFAULT','ForcingEnd'),"%d/%m/%Y %H:%M")  # Start of forcing

        pcraster_path = parser.get('Path', 'PCRHOME')
        self.pcraster_cmd = {}
        for execname in ["pcrcalc","map2asc","asc2map","col2map","map2col","mapattr","resample","readmap"]:
            self.pcraster_cmd[execname] = getPCrasterPath(pcraster_path, settings_file, alias=execname)


def create_gauge_loc(cfg, path_subcatch):
    # For some reason this version of LISFLOOD doesn't work with outlet map,
    # hence have to supply gauge coordinates
    gaugeloc_txt = os.path.join(path_subcatch, "maps", "gaugeloc.txt")
    pcrasterCommand(cfg.pcraster_cmd['map2col'] + " F0 F1"  , {"F0": os.path.join(path_subcatch, "maps", "outlet.map"), "F1":gaugeloc_txt})


def prepare_inflows(cfg, path_subcatch, index):

    # Copy simulated streamflow from upstream catchments
    # Change inlet map by replacing the numeric ID's with 1, 2, ...
    print("Upstream station(s): ")
    direct_links = pandas.read_csv(os.path.join(cfg.path_result, "direct_links.csv"), sep=",", index_col=0)
    #inflow_tss is created according to the cal_start cal_end parameyters, script removes steps before and after and it reindex the steps
    
    inflow_tss = os.path.join(path_subcatch, "inflow","chanq.tss")
    #inflow_tss_lastrun is for when after the optimal combination of parameters is found , when we run the full forcing period
    inflow_tss_last_run = os.path.join(path_subcatch, "inflow", "chanq_last_run.tss")
    
    try: del big_one
    except: pass
    try: 
        os.remove(inflow_tss)
        os.remove(inflow_tss_last_run)
    except: pass
    upstream_catchments = [int(i) for i in direct_links.loc[index].values if not np.isnan(i)]
    cnt = 1
    subcatchinlets_map = os.path.join(path_subcatch, "inflow", "inflow.map")
    # subcatchinlets_new_map = os.path.join(path_subcatch,"inflow","inflow_new.map")
    subcatchinlets_cut_map = os.path.join(path_subcatch, "inflow", "inflow_cut.map")
    smallsubcatchmask_map = os.path.join(path_subcatch, "maps", "masksmall.map")
    
    # pcrasterCommand(pcrcalc + " 'F0 = F1*0.0'", {"F0":subcatchinlets_new_map,"F1":subcatchinlets_map})
    header = ""
    for subcatchment in upstream_catchments:
        
        subcatchment = str(subcatchment)

        print(subcatchment+" ")
                        
        Qsim_tss = os.path.join(cfg.subcatchment_path, subcatchment, "out","chanq_simulated_best.tss")
                
        if not os.path.exists(Qsim_tss) or os.path.getsize(Qsim_tss) == 0:
            raise Exception("ERROR: Missing " + Qsim_tss)

        try:
            # DD The shift_time.days is not correctly read for 6-hourly. Using time stamps to make it timesteps invariant
            simulated_streamflow_tmp = pandas.read_csv(Qsim_tss, sep=r"\s+", index_col=False, skiprows=4, header=None, usecols=[1])
            simulated_streamflow_tmp.index = pandas.date_range(cfg.forcing_start, periods=len(simulated_streamflow_tmp), freq='6H')
            # DD comment the following line if you want to make the inflow the complete period
            # simulated_streamflow_tmp = simulated_streamflow_tmp.loc[datetime.datetime.strptime(row['Cal_Start'], "%d/%m/%Y %H:%M"):datetime.datetime.strptime(row['Cal_End'], '%d/%m/%Y %H:%M')]
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
#            if cnt==1: big_one = simulated_streamflow  # type: object
#            else: big_one[str(cnt)] = simulated_streamflow.values
        # DD don't need this as it causes inflow points to be deleted in inflow.py
#             numeric_only = re.compile(r'[^\d.]+')
        # hhh = str(int(numeric_only.sub('',subcatchment)))
        # pcrasterCommand(pcrcalc + " 'F0 = F0+scalar(F1=="+hhh+")*"+str(cnt)+"'", {"F0": subcatchinlets_new_map,"F1":subcatchinlets_map})
        cnt += 1
        header = header+subcatchment+"\n"

    # DD If the following commands give an error, then replace it with the proper method to cut pcraster maps without getting the error
    # In addition, there is no point in converting points to indices from 1 to 5 if they are later removed in inflow.py.
    # So instead, just clip the map with the original catchment numbers
    # pcrasterCommand(pcrcalc + " 'F1 = if(scalar(boolean(F0))>0,nominal(F0))'", {"F0": subcatchinlets_new_map,"F1": subcatchinlets_new2_map})
    # pcrasterCommand(resample + " F0 F1 --clone F2 " , {"F0": subcatchinlets_new2_map, "F1":subcatchinlets_new3_map, "F2":smallsubcatchmask_map})
    #print("(note that despite memory error, inflow_new3.map is being created, strange...)")
    # pcrasterCommand(pcrcalc + " 'F1 = if(F0>=0,F0)'", {"F0": subcatchinlets_map,"F1": subcatchinlets_new_map})
    pcrasterCommand(cfg.pcraster_cmd['resample'] + " --clone F2 F0 F1" , {"F0": subcatchinlets_map, "F1":subcatchinlets_cut_map, "F2":smallsubcatchmask_map})
    # map = pcraster.readmap(subcatchinlets_cut_map)
    # mapNpyInt = int(pcraster.pcr2numpy(map, -9999))
    # mapN = pcraster.numpy2pcr(pcraster.Nominal, map, -9999)

    if ("big_one" in globals()) or ("big_one" in locals()):

        big_one_lastrun.to_csv(inflow_tss_last_run,sep=' ',header=False)
        #simulated_streamflow_lastrun.to_csv(inflow_tss_last_run,sep=' ',header=False)
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
    else:
        sys.stdout.write("No upstream inflow needed\n")
    sys.stdout.write("\n")


def calibrate_subcatchment(cfg, index):

    print("=================== "+str(index)+" ====================")
    path_subcatch = os.path.join(cfg.subcatchment_path, str(index))
    if os.path.exists(os.path.join(path_subcatch, "streamflow_simulated_best.csv")):
        print("streamflow_simulated_best.csv already exists! Moving on...")
        return
    print(">> Starting calibration of catchment "+str(index))

    create_gauge_loc(cfg, path_subcatch)

    prepare_inflows(cfg, path_subcatch, index)

    # Performing calibration with external call, to avoid multiprocessing problems
    try:     
        import cal_single_objfun
        cal_single_objfun.main()

    except:
        traceback.print_exc()
        raise Exception("ERROR in CAL_7_PERFORM_CAL")


def main(args):
    ########################################################################
    #   Read settings file
    ########################################################################
    if len(args) == 0:
        print(args)
        settings_file = os.path.normpath(sys.argv[1])
        catchments_list = os.path.normpath(sys.argv[2])
    else:
        print(sys.argv)
        settings_file = os.path.normpath(args[0])
        catchments_list = os.path.normpath(args[1])

    cfg = Config(settings_file)

    ########################################################################
    #   Loop through catchments and perform calibration
    ########################################################################

    print(">> Reading Qmeta2.csv file...")
    stationdata = pandas.read_csv(os.path.join(cfg.path_result,"Qmeta2.csv"), sep=",", index_col=0)
    stationdata_sorted = stationdata.sort_values(by=['DrainingArea.km2.LDD'], ascending=True)
    catchments = pandas.read_csv(catchments_list, sep=",", header=None)

    obs_ids = catchments.loc[:,0]

    for index, row in stationdata_sorted.iterrows():
        
        if len(obs_ids[obs_ids==index]) == 0: # Only process catchments whose ObsID is in the stationdata file
            continue

        calibrate_subcatchment(cfg, index)

    print("==================== END ====================")

if __name__ == '__main__':
    #h = hpy() 
    main()
    #print(h.heap())
