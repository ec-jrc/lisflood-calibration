import os
import sys
import pandas
import numpy as np
from datetime import datetime

from liscal import pcr_utils


class SubCatchment():

    def __init__(self, cfg, obsid, station_data, initialise=True):

        self.obsid = obsid
        self.data = station_data
        self.path = os.path.join(cfg.subcatchment_path, str(obsid))

        if initialise:
            cal_start, cal_end = self.calibration_start_end(cfg)
            self.cal_start = cal_start
            self.cal_end = cal_end

            self.gaugeloc = self.create_gauge_loc(cfg)

            self.inflowflag = self.prepare_inflows(cfg)

    def calibration_start_end(self, cfg):
        if cfg.fast_debug:
            # Turn this on for debugging faster. You can speed up further by setting maxGen = 1
            cal_start = self.data['Cal_Start']
            cal_end = (datetime.strptime(cal_start, '%Y-%m-%d %H:%M') + timedelta(days=1121)).strftime('%Y-%m-%d %H:%M')
            # !!!! rewrite cfg parameters
            cfg.forcing_start = datetime.strptime(cal_start, '%Y-%m-%d %H:%M')
            cfg.forcing_end = datetime.strptime(cal_end, '%Y-%m-%d %H:%M')
            cfg.WarmupDays = 0
        else:
            # Compute the time steps at which the calibration should start and end
            cal_start = self.data['Cal_Start']
            cal_end = self.data['Cal_End'] # DD original

        return cal_start, cal_end

    def create_gauge_loc(self, cfg):
        # For some reason this version of LISFLOOD doesn't work with outlet map,
        # hence have to supply gauge coordinates
        gaugeloc_txt = os.path.join(self.path, "maps", "gaugeloc.txt")
        with open(gaugeloc_txt,"r") as f:
            for line in f.readlines():
                (X,Y,value) = line.split()
        gaugeloc = str(float(X))+" "+str(float(Y))

        pcr_utils.pcrasterCommand(cfg.pcraster_cmd['map2col'] + " F0 F1"  , {"F0": os.path.join(self.path, "maps", "outlet.map"), "F1":gaugeloc_txt})

        return gaugeloc

    def prepare_inflows(self, cfg):

        # Copy simulated streamflow from upstream catchments
        # Change inlet map by replacing the numeric ID's with 1, 2, ...
        print("Upstream station(s): ")
        direct_links = pandas.read_csv(os.path.join(cfg.path_result, "direct_links.csv"), sep=",", index_col=0)
        #inflow_tss is created according to the cal_start cal_end parameyters, script removes steps before and after and it reindex the steps
        
        inflow_tss = os.path.join(self.path, "inflow", "chanq.tss")
        #inflow_tss_lastrun is for when after the optimal combination of parameters is found , when we run the full forcing period
        inflow_tss_last_run = os.path.join(self.path, "inflow", "chanq_last_run.tss")
        
        try: del big_one
        except: pass
        try: 
            os.remove(inflow_tss)
            os.remove(inflow_tss_last_run)
        except: pass
        upstream_catchments = [int(i) for i in direct_links.loc[self.obsid].values if not np.isnan(i)]
        cnt = 1
        subcatchinlets_map = os.path.join(self.path, "inflow", "inflow.map")
        # subcatchinlets_new_map = os.path.join(path_subcatch,"inflow","inflow_new.map")
        subcatchinlets_cut_map = os.path.join(self.path, "inflow", "inflow_cut.map")
        smallsubcatchmask_map = os.path.join(self.path, "maps", "masksmall.map")
        
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
        pcr_utils.pcrasterCommand(cfg.pcraster_cmd['resample'] + " --clone F2 F0 F1" , {"F0": subcatchinlets_map, "F1":subcatchinlets_cut_map, "F2":smallsubcatchmask_map})
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
