#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Please refer to quick_guide.pdf for usage instructions"""
import os
import sys
import argparse
import random
import numpy as np
import pandas as pd
from configparser import ConfigParser # Python 3.8
import glob
import subprocess
import traceback

from liscal import templates, calibration, config, subcatchment, objective, hydro_model



def calibrate_subcatchment(cfg, obsid, subcatch):

    print("=================== "+str(obsid)+" ====================")
    if os.path.exists(os.path.join(subcatch.path, "out", "streamflow_simulated_best.csv")):
        print("streamflow_simulated_best.csv already exists! Moving on...")
        return

    lis_template = templates.LisfloodSettingsTemplate(cfg, subcatch)
    
    if os.path.exists(os.path.join(subcatch.path,"pareto_front.csv"))==False:
        print(">> Starting calibration of catchment "+str(obsid))

        lock_mgr = calibration.LockManager(cfg.num_cpus)

        obj = objective.ObjectiveKGE(cfg, subcatch)

        model = hydro_model.HydrologicalModel(cfg, subcatch, lis_template, lock_mgr, obj)

        # load forcings and input maps in cache
        # required in front of processing pool
        # otherwise each child will reload the maps
        model.init_run()

        # Adjust param_ranges list if lakes or reservoirs are not included into the current catchment
        cfg.original_param_ranges = cfg.param_ranges.copy()
        if model.lissettings.options['simulateLakes']==False:
            if 'LakeMultiplier' in cfg.param_ranges.index:
                cfg.param_ranges.drop("LakeMultiplier", inplace=True)
        if model.lissettings.options['simulateReservoirs']==False:
            if 'ReservoirFloodStorage' in cfg.param_ranges.index:
                cfg.param_ranges.drop("ReservoirFloodStorage", inplace=True)
            if 'ReservoirFloodOutflowFactor' in cfg.param_ranges.index:
                cfg.param_ranges.drop("ReservoirFloodOutflowFactor", inplace=True)
        if model.lissettings.options['MCTRouting']==False:
            if 'CalChanMan3' in cfg.param_ranges.index:
                cfg.param_ranges.drop("CalChanMan3", inplace=True)

        # Adjust param_ranges list if min Daily Avg Temp > 1 so that SnowMelt coefficient should not be calibrated for the current catchment
        station_data_file=os.path.join(os.path.join(subcatch.path_station,'station_data.csv'))
        StationDataFile=pd.read_csv(station_data_file,index_col=0)
        if float(StationDataFile.loc["min_TAvgS"]) > float(model.lissettings.binding['TempSnow']):
            if 'SnowMeltCoef' in cfg.param_ranges.index:
                cfg.param_ranges.drop("SnowMeltCoef", inplace=True)

        calib_deap = calibration.CalibrationDeap(cfg, model.run, obj.weights, cfg.seed)
        calib_deap.run(subcatch.path, lock_mgr)

        obj.process_results()
    else:
        print("pareto_front.csv already exists! Moving on...")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('settings_file', help='Calibration settings file')
    parser.add_argument('station', help='Station OBSID to process')
    parser.add_argument('n_cpus', help='Number of cpus')
    parser.add_argument('--seed', help='Seed value for random numbers generation in deap')
    args = parser.parse_args()

    settings_file = args.settings_file

    print('Running calibration using {} cpus'.format(args.n_cpus))

    cfg = config.ConfigCalibration(settings_file, args.n_cpus, args.seed)

    # Calibrate lisflood fo specified station
    obsid = int(args.station)

    subcatch = subcatchment.SubCatchment(cfg, obsid)

    calibrate_subcatchment(cfg, obsid, subcatch)

    print("==================== END ====================")
