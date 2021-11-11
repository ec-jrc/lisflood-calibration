import os
import sys
import pandas
import numpy as np
import pcraster as pcr
from datetime import datetime, timedelta

from liscal import pcr_utils, utils, stations


class SubCatchment():

    def __init__(self, cfg, obsid, station_data=None, initialise=True, create_links=True):

        self.obsid = obsid
        self.path = os.path.join(cfg.subcatchment_path, str(obsid))
        self.path_out = os.path.join(self.path, 'out')
        self.path_station = os.path.join(self.path, 'station')
        self.create_links = create_links

        if station_data is None:
            # Read full list of stations, index is obsid
            station_file = os.path.join(self.path_station, 'station_data.csv')
            print(">> Reading stations_data file {}".format(station_file))
            self.data = pandas.read_csv(station_file, sep=",", index_col=0)
            self.data = self.data[str(self.obsid)]
        else:
            self.data = station_data
        print('Station data:')
        print('---------------------------------------')
        print(self.data)
        print('---------------------------------------')

        if initialise:

            os.makedirs(self.path_out, exist_ok=True)

            outlet_file = os.path.join(self.path, "maps", "outletsmall.map")
            self.gaugeloc = self.extract_gauge_loc(outlet_file)
            print('Gauge location {}'.format(self.gaugeloc))

            self.inflowflag, n_inflows = self.prepare_inflows(cfg)
            print('Found {} inflows'.format(n_inflows))
            self.resample_inflows(cfg)

    def extract_gauge_loc(self, outlet_file):
        x = self.data['LisfloodX']
        y = self.data['LisfloodY']
        gaugeloc = str(float(x))+" "+str(float(y))
        
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

        count = 1
        all_inflows = None

        if self.create_links:
            # Copy simulated streamflow from upstream catchments
            # Change inlet map by replacing the numeric ID's with 1, 2, ...
            print("Upstream station(s): ")
            if not os.path.exists(cfg.stations_links) or os.path.getsize(cfg.stations_links) == 0:
                raise FileNotFoundError("stations_links missing: {}".format(cfg.stations_links))
            stations_links = pandas.read_csv(cfg.stations_links, sep=",", index_col=0)
            inflow_tss = os.path.join(self.path, "inflow", "chanq.tss")
            if os.path.isfile(inflow_tss):
                os.remove(inflow_tss)

            upstream_catchments = [int(i) for i in stations_links.loc[self.obsid].values if not np.isnan(i)]

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
                        
                if count == 1: 
                    all_inflows = simulated_streamflow  # type: object
                else:
                    all_inflows[str(count)] = simulated_streamflow.values
                count += 1
                header = header+subcatchment+"\n"
                print('Found inflow for subcatchment {}'.format(subcatchment))

        n_inflows =  count - 1

        # hack csv into tss (duh)
        if all_inflows is not None:
            all_inflows.to_csv(inflow_tss, sep=' ', header=False)
            f = open(inflow_tss, 'r+')
            content = f.read()
            content = 'timeseries scalar\n'+str(count)+'\n'+'timestep\n'+header+content
            f.seek(0,0)
            f.write(content)
            f.close()
            inflowflag = str(1)
        else:
            print("No upstream inflow needed\n")
            inflowflag = str(0)

        return inflowflag, n_inflows
