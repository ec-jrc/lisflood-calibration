import os
import sys
import pandas
import numpy as np
import pcraster as pcr
from datetime import datetime

from liscal import pcr_utils, utils


class SubCatchment():

    def __init__(self, cfg, obsid, station_data, initialise=True):

        self.obsid = obsid
        self.data = station_data
        self.path = os.path.join(cfg.subcatchment_path, str(obsid))
        self.path_out = os.path.join(self.path, 'out')

        if initialise:

            os.makedirs(self.path_out, exist_ok=True)

            cal_start, cal_end = self.calibration_start_end(cfg)
            self.cal_start = cal_start
            self.cal_end = cal_end

            outlet_file = os.path.join(self.path, "maps", "outletsmall.map")
            self.gaugeloc = self.extract_gauge_loc(outlet_file)
            print('Gauge location {}'.format(self.gaugeloc))

            self.inflowflag, n_inflows = self.prepare_inflows(cfg)
            print('Found {} inflows'.format(n_inflows))
            self.resample_inflows(cfg)


    def calibration_start_end(self, cfg):
        # Compute the time steps at which the calibration should start and end
        cal_start = datetime.strptime(self.data['Cal_Start'], '%d/%m/%Y %H:%M').strftime('%Y-%m-%d %H:%M')
        cal_end = datetime.strptime(self.data['Cal_End'], '%d/%m/%Y %H:%M').strftime('%Y-%m-%d %H:%M')

        if cfg.fast_debug:
            # Turn this on for debugging faster. You can speed up further by setting maxGen = 1
            cal_end = (datetime.strptime(cal_start, '%Y-%m-%d %H:%M') + timedelta(days=1121)).strftime('%Y-%m-%d %H:%M')
            # !!!! rewrite cfg parameters
            cfg.forcing_start = datetime.strptime(cal_start, '%Y-%m-%d %H:%M')
            cfg.forcing_end = datetime.strptime(cal_end, '%Y-%m-%d %H:%M')
            cfg.WarmupDays = 0

        return cal_start, cal_end

    def extract_gauge_loc(self, outlet_file):

        outlet = pcr.readmap(outlet_file)
        x = pcr.pcr2numpy(pcr.xcoordinate(outlet == 1), mv=-1)
        y = pcr.pcr2numpy(pcr.ycoordinate(outlet == 1), mv=-1)
        gaugeloc = str(float(x[x!=-1]))+" "+str(float(y[y!=-1]))
        return gaugeloc

    def resample_inflows(self, cfg):
        subcatchinlets_map = os.path.join(self.path, "inflow", "inflow.map")
        subcatchinlets_cut_map = os.path.join(self.path, "inflow", "inflow_cut.map")
        smallsubcatchmask_map = os.path.join(self.path, "maps", "masksmall.map")

        if not os.path.isfile(subcatchinlets_map):
            raise FileNotFoundError('inflow map missing: {}'.format(subcatchinlets_map))
        if not os.path.isfile(smallsubcatchmask_map):
            raise FileNotFoundError('mask map missing: {}'.format(smallsubcatchmask_map))
        pcr_utils.pcrasterCommand(cfg.pcraster_cmd['resample'] + " --clone F2 F0 F1" , {"F0": subcatchinlets_map, "F1":subcatchinlets_cut_map, "F2":smallsubcatchmask_map})


    def prepare_inflows(self, cfg):

        # Copy simulated streamflow from upstream catchments
        # Change inlet map by replacing the numeric ID's with 1, 2, ...
        print("Upstream station(s): ")
        direct_links = pandas.read_csv(os.path.join(cfg.path_result, "direct_links.csv"), sep=",", index_col=0)
        inflow_tss = os.path.join(self.path, "inflow", "chanq.tss")
        if os.path.isfile(inflow_tss):
            os.remove(inflow_tss)

        upstream_catchments = [int(i) for i in direct_links.loc[self.obsid].values if not np.isnan(i)]
        cnt = 1
        
        all_inflows = None
        n_inflows = 0
        header = ""
        for subcatchment in upstream_catchments:
            
            subcatchment = str(subcatchment)

            print('Retrieving inflow for subcatchment {}'.format(subcatchment))
                            
            Qsim_tss = os.path.join(cfg.subcatchment_path, subcatchment, "out", "chanq_simulated_best.tss")

            if not os.path.exists(Qsim_tss) or os.path.getsize(Qsim_tss) == 0:
                raise Exception("ERROR: Missing " + Qsim_tss)

            try:
                simulated_streamflow = utils.read_tss(Qsim_tss)
            except:
                print("Could not find streamflow_simulated_best.tss for upstream catchment "+subcatchment+", hence cannot run this catchment...")
                raise Exception("Stopping...")

            simulated_streamflow.index = pandas.date_range(cfg.forcing_start, periods=len(simulated_streamflow), freq='6H')
            if simulated_streamflow.index[-1] != cfg.forcing_end:
                raise Exception('Forcing start and end dates not coherent with inflow data, expecting {}, got {}',
                    cfg.forcing_end, simulated_streamflow.index[-1])
            simulated_streamflow.index = [i+1 for i in range(len(simulated_streamflow))]
                    
            if cnt==1: 
                all_inflows = simulated_streamflow  # type: object
            else:
                all_inflows[str(cnt)] = simulated_streamflow.values
            cnt += 1
            header = header+subcatchment+"\n"
            n_inflows += 1
            print('Found inflow for subcatchment {}'.format(subcatchment))

        if all_inflows is not None:
            all_inflows.to_csv(inflow_tss,sep=' ',header=False)
            f = open(inflow_tss,'r+')
            content = f.read()
            content = 'timeseries scalar\n'+str(cnt)+'\n'+'timestep\n'+header+content
            f.seek(0,0)
            f.write(content)
            f.close()
        else:
            print("No upstream inflow needed\n")
        sys.stdout.write("\n")

        inflowflag = str(0)
        if os.path.isfile(inflow_tss):
            inflowflag = str(1)

        return inflowflag, n_inflows
