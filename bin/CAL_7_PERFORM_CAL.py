# -*- coding: utf-8 -*-
"""Please refer to quick_guide.pdf for usage instructions"""
import os
import sys
import argparse
import numpy as np
import pandas
from configparser import ConfigParser # Python 3.8
import glob
import subprocess
import traceback

from liscal import templates, calibration, config, subcatchment, objective, hydro_model


def calibrate_subcatchment(cfg, obsid, station_data):

    print("=================== "+str(obsid)+" ====================")
    subcatch = subcatchment.SubCatchment(cfg, obsid, station_data)
    if os.path.exists(os.path.join(subcatch.path, "streamflow_simulated_best.csv")):
        print("streamflow_simulated_best.csv already exists! Moving on...")
        return

    if os.path.exists(os.path.join(subcatch.path,"pareto_front.csv"))==False:
        print(">> Starting calibration of catchment "+str(obsid))

        lis_template = templates.LisfloodSettingsTemplate(cfg, subcatch)

        lock_mgr = calibration.LockManager(cfg.deap_param.num_cpus)

        obj = objective.ObjectiveDischarge(cfg, subcatch)

        model = hydro_model.HydrologicalModel(cfg, subcatch, lis_template, lock_mgr, obj)

        # load forcings and input maps in cache
        # required in front of processing pool
        # otherwise each child will reload the maps
        model.init_run()

        calib_deap = calibration.CalibrationDeap(cfg, model.run)
        calib_deap.run(subcatch.path, lock_mgr)

        obj.process_results()

    hydro_model.generate_outlet_streamflow(cfg, subcatch, lis_template)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('settings_file', help='Calibration settings file')
    parser.add_argument('stations', help='List of stations to process')
    args = parser.parse_args()

    settings_file = args.settings_file
    stations_list = args.stations

    cfg = config.Config(settings_file)

    # Read full list of stations, index is obsid
    print(">> Reading Qmeta2.csv file...")
    stations_meta = pandas.read_csv(cfg.Qmeta_csv, sep=",", index_col=0)

    # Read list of stations we want to calibrate
    stations = pandas.read_csv(stations_list, sep=",", header=None)
    obsid_list = stations.loc[:, 0]

    ########################################################################
    #   Loop through subcatchments and perform calibration
    ########################################################################
    for obsid in obsid_list:

        try:
            station_data = stations_meta.loc[obsid]
        except KeyError as e:
            raise Exception('Station {} not found in stations file'.format(obsid))

        calibrate_subcatchment(cfg, obsid, station_data)

    print("==================== END ====================")
