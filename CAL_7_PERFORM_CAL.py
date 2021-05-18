# -*- coding: utf-8 -*-
"""Please refer to quick_guide.pdf for usage instructions"""
import os
import sys
import numpy as np
import pandas
from datetime import datetime
from configparser import ConfigParser # Python 3.8
from pcrasterCommand import pcrasterCommand, getPCrasterPath
import glob
import datetime
import subprocess
import traceback

  
class DEAPParameters():

    def __init__(self, parser):
        self.use_multiprocessing = int(parser.get('DEAP','use_multiprocessing'))
        self.numCPUs = int(parser.get('DEAP','numCPUs'))
        self.minGen = int(parser.get('DEAP','minGen'))
        self.maxGen = int(parser.get('DEAP','maxGen'))
        self.pop = int(parser.get('DEAP','pop'))
        self.mu = int(parser.get('DEAP','mu'))
        self.lambda_ = int(parser.get('DEAP','lambda_'))

        self.cxpb = 0.6
        self.mutpb = 0.4

class Config():

    def __init__(self, settings_file):

        parser = ConfigParser()
        parser.read(settings_file)

        # paths
        self.path_result = parser.get('Path', 'Result')
        self.subcatchment_path = parser.get('Path','SubCatchmentPath')


        pcraster_path = parser.get('Path', 'PCRHOME')
        self.pcraster_cmd = {}
        for execname in ["pcrcalc", "map2asc", "asc2map", "col2map", "map2col", "mapattr", "resample", "readmap"]:
            self.pcraster_cmd[execname] = getPCrasterPath(pcraster_path, settings_file, alias=execname)

        # deap
        self.deap_param = DEAPParameters(parser)
        # Load param ranges file
        self.param_ranges = pandas.read_csv(parser.get('Path','ParamRanges'), sep=",", index_col=0)

        # template
        self.lisflood_template = parser.get('Templates','LISFLOODSettings')

        # Debug/test parameters
        self.test_convergence = bool(int(parser.get('DEFAULT', 'testConvergence')))
        self.fast_debug = bool(int(parser.get('DEFAULT', 'fastDebug')))
        self.numDigits = int(parser.get('DEFAULT', 'numDigitsTests'))

        # Date parameters
        self.ObservationsStart = datetime.strptime(parser.get('DEFAULT', 'ObservationsStart'), "%d/%m/%Y %H:%M")  # Start of forcing
        self.ObservationsEnd = datetime.strptime(parser.get('DEFAULT', 'ObservationsEnd'), "%d/%m/%Y %H:%M")  # Start of forcing
        self.forcing_start = datetime.datetime.strptime(parser.get('DEFAULT','ForcingStart'),"%d/%m/%Y %H:%M")  # Start of forcing
        self.forcing_end = datetime.datetime.strptime(parser.get('DEFAULT','ForcingEnd'),"%d/%m/%Y %H:%M")  # Start of forcing
        self.WarmupDays = int(parser.get('DEFAULT', 'WarmupDays'))
        self.calibration_freq = parser.get('DEFAULT', 'calibrationFreq')

        # observations
        self.Qtss_csv = parser.get('CSV', 'Qtss')




def create_gauge_loc(cfg, path_subcatch):
    # For some reason this version of LISFLOOD doesn't work with outlet map,
    # hence have to supply gauge coordinates
    gaugeloc_txt = os.path.join(path_subcatch, "maps", "gaugeloc.txt")
    with open(gaugeloc_txt,"r") as f:
        for line in f.readlines():
            (X,Y,value) = line.split()
    gaugeloc = str(float(X))+" "+str(float(Y))

    pcrasterCommand(cfg.pcraster_cmd['map2col'] + " F0 F1"  , {"F0": os.path.join(path_subcatch, "maps", "outlet.map"), "F1":gaugeloc_txt})

    return gaugeloc

def prepare_inflows(cfg, path_subcatch, index):

    # Copy simulated streamflow from upstream catchments
    # Change inlet map by replacing the numeric ID's with 1, 2, ...
    print("Upstream station(s): ")
    direct_links = pandas.read_csv(os.path.join(cfg.path_result, "direct_links.csv"), sep=",", index_col=0)
    #inflow_tss is created according to the cal_start cal_end parameyters, script removes steps before and after and it reindex the steps
    
    inflow_tss = os.path.join(path_subcatch, "inflow", "chanq.tss")
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
                        
        Qsim_tss = os.path.join(cfg.subcatchment_path, subcatchment, "out", "chanq_simulated_best.tss")
                
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
                
        simulated_streamflow = simulated_streamflow_tmp
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

    inflowflag = str(0)
    if os.path.isfile(inflow_tss):
        inflowflag = str(1)

    return inflowflag


class LisfloodSettingsTemplate():

    def __init__(self, cfg, path_subcatch, gaugeloc, inflowflag):

        self.outfix = os.path.join(path_subcatch, os.path.basename(cfg.lisflood_template[:-4]))
        self.lisflood_template = cfg.lisflood_template
        with open(os.path.join('templates', cfg.lisflood_template), "r") as f:
            template_xml = f.read()
    
        template_xml = template_xml.replace('%gaugeloc', gaugeloc) # Gauge location
        template_xml = template_xml.replace('%inflowflag', inflowflag)
        template_xml = template_xml.replace('%ForcingStart', cfg.forcing_start.strftime('%Y-%m-%d %H:%M')) # Date of forcing start
        template_xml = template_xml.replace('%SubCatchmentPath', path_subcatch)

        self.template_xml = template_xml

    def settings_path(self, suffix):
        return self.outfix+suffix+run_rand_id+'.xml'

    def write_template(self, obsid, run_rand_id, cal_start_local, cal_end_local, param_ranges, parameters):

        out_xml = self.template_xml

        out_xml = out_xml.replace('%run_rand_id', run_rand_id)
        out_xml = out_xml.replace('%CalStart', cal_start_local) # Date of Cal starting
        out_xml = out_xml.replace('%CalEnd', cal_end_local)  # Time step of forcing at which to end simulation

        for ii in range(len(param_ranges)):
            ## DD Special Rule for the SAVA
            if obsid == '851' and (param_ranges.index[ii] == "adjust_Normal_Flood" or param_ranges.index[ii] == "ReservoirRnormqMult"):
                out_xml = out_xml.replace('%adjust_Normal_Flood',"0.8")
                out_xml = out_xml.replace('%ReservoirRnormqMult',"1.0")
            out_xml = out_xml.replace("%"+param_ranges.index[ii],str(parameters[ii]))

        out_xml_prerun = out_xml
        out_xml_prerun = out_xml_prerun.replace('%InitLisflood',"1")
        with open(self.outfix+'-PreRun'+run_rand_id+'.xml', "w") as f:
            f.write(out_xml_prerun)

        out_xml_run = out_xml
        out_xml_run = out_xml_run.replace('%InitLisflood',"0")
        with open(self.outfix+'-Run'+run_rand_id+'.xml', "w") as f:
            f.write(out_xml_run)


def calibrate_subcatchment(cfg, obsid, station_data):

    print("=================== "+str(obsid)+" ====================")
    path_subcatch = os.path.join(cfg.subcatchment_path, str(obsid))
    if os.path.exists(os.path.join(path_subcatch, "streamflow_simulated_best.csv")):
        print("streamflow_simulated_best.csv already exists! Moving on...")
        return
    print(">> Starting calibration of catchment "+str(obsid))

    gaugeloc = create_gauge_loc(cfg, path_subcatch)

    inflowflag = prepare_inflows(cfg, path_subcatch, obsid)

    lis_template = LisfloodSettingsTemplate(cfg, path_subcatch, obsid, gaugeloc, inflowflag)

    lock_mgr = LockManager()

    model = HydrologicalModel(cfg, obsid, station_data, lis_template, lock_mgr)

    # Performing calibration with external call, to avoid multiprocessing problems
    import cal_single_objfun

    if os.path.exists(os.path.join(path_subcatch,"pareto_front.csv"))==False:
        cal_single_objfun.run_calibration(cfg, obsid, station_data, model, lock_mgr)

    cal_single_objfun.generate_outlet_streamflow(cfg, obsid, station_data, model)


def calibrate_system(args):
    ########################################################################
    #   Read settings file
    ########################################################################
    if len(args) == 0:
        print(args)
        settings_file = os.path.normpath(sys.argv[1])
        subcatchments_list = os.path.normpath(sys.argv[2])
    else:
        print(sys.argv)
        settings_file = os.path.normpath(args[0])
        subcatchments_list = os.path.normpath(args[1])

    cfg = Config(settings_file)
    
    # Read full list of stations, index is obsid
    print(">> Reading Qmeta2.csv file...")
    stations = pandas.read_csv(os.path.join(cfg.path_result,"Qmeta2.csv"), sep=",", index_col=0)

    # Read list of stations we want to calibrate
    subcatchments = pandas.read_csv(subcatchments_list, sep=",", header=None)
    obsid_list = subcatchments.loc[:,0]

    ########################################################################
    #   Loop through subcatchments and perform calibration
    ########################################################################
    for obsid in obsid_list:

        try:
            station_data = stations.loc[obsid]
        except KeyError as e:
            raise Exception('Station {} not found in stations file'.format(obsid))

        calibrate_subcatchment(cfg, obsid, station_data)

    print("==================== END ====================")

if __name__ == '__main__':
    #h = hpy() 
    calibrate_system()
    #print(h.heap())
