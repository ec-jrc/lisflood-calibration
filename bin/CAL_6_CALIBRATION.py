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

        if cfg.deap_param.apply_multiobjective_calibration:
            obj.set_custom_multiobjective_weights(cfg.deap_param.objective_KGE,
                                                  cfg.deap_param.objective_corr,
                                                  cfg.deap_param.objective_bias,
                                                  cfg.deap_param.objective_y,
                                                  cfg.deap_param.objective_sae )

        model = hydro_model.HydrologicalModel(cfg, subcatch, lis_template, lock_mgr, obj)

        # load forcings and input maps in cache
        # required in front of processing pool
        # otherwise each child will reload the maps
        model.init_run()

        cfg.filter_param_ranges_after_init(model_initialized=model)

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
