# -*- coding: utf-8 -*-
"""Please refer to quick_guide.pdf for usage instructions"""
import os
import sys
import numpy as np
import pandas
from datetime import datetime
from configparser import ConfigParser # Python 3.8
import glob
import subprocess
import traceback

from liscal import pcr_utils, templates, calibration


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
            self.pcraster_cmd[execname] = pcr_utils.getPCrasterPath(pcraster_path, settings_file, alias=execname)

        # deap
        self.deap_param = calibration.DEAPParameters(parser)
        # Load param ranges file
        self.param_ranges = pandas.read_csv(parser.get('Path','ParamRanges'), sep=",", index_col=0)

        # template
        self.lisflood_template = parser.get('Templates','LISFLOODSettings')

        # Debug/test parameters
        self.fast_debug = bool(int(parser.get('DEFAULT', 'fastDebug')))

        # Date parameters
        self.ObservationsStart = datetime.strptime(parser.get('DEFAULT', 'ObservationsStart'), "%d/%m/%Y %H:%M")  # Start of forcing
        self.ObservationsEnd = datetime.strptime(parser.get('DEFAULT', 'ObservationsEnd'), "%d/%m/%Y %H:%M")  # Start of forcing
        self.forcing_start = datetime.strptime(parser.get('DEFAULT','ForcingStart'),"%d/%m/%Y %H:%M")  # Start of forcing
        self.forcing_end = datetime.strptime(parser.get('DEFAULT','ForcingEnd'),"%d/%m/%Y %H:%M")  # Start of forcing
        self.WarmupDays = int(parser.get('DEFAULT', 'WarmupDays'))
        self.calibration_freq = parser.get('DEFAULT', 'calibrationFreq')

        # observations
        self.Qtss_csv = parser.get('CSV', 'Qtss')


def calibrate_subcatchment(cfg, obsid, station_data):

    print("=================== "+str(obsid)+" ====================")
    path_subcatch = os.path.join(cfg.subcatchment_path, str(obsid))
    if os.path.exists(os.path.join(path_subcatch, "streamflow_simulated_best.csv")):
        print("streamflow_simulated_best.csv already exists! Moving on...")
        return
    print(">> Starting calibration of catchment "+str(obsid))

    gaugeloc = pcr_utils.create_gauge_loc(cfg, path_subcatch)

    inflowflag = pcr_utils.prepare_inflows(cfg, path_subcatch, obsid)

    lis_template = templates.LisfloodSettingsTemplate(cfg, path_subcatch, obsid, gaugeloc, inflowflag)

    lock_mgr = calibration.LockManager()

    model = hydro_model.HydrologicalModel(cfg, obsid, path_subcatch, station_data, lis_template, lock_mgr)

    # Performing calibration with external call, to avoid multiprocessing problems
    if os.path.exists(os.path.join(path_subcatch,"pareto_front.csv"))==False:
        calibration.run_calibration(cfg, obsid, path_subcatch, station_data, model, lock_mgr)

    hydro_model.generate_outlet_streamflow(cfg, obsid, path_subcatch, station_data, lis_template)


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
