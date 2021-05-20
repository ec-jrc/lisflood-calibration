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

from liscal import pcr_utils, templates, calibration, config


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

    cfg = config.Config(settings_file)
    
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
