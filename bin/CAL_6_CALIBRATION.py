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

        cfg.filter_param_ranges_after_init(model_initialized=model, split_lake_params=cfg.deap_param.split_lake_params)
        
        # additional check on the calibration status
        rerun_with_KGE = False
        calibstatus_file_path = os.path.join(subcatch.path,'CalibrationStatus_2nd_run_KGE.txt')
        if os.path.exists(calibstatus_file_path)==True:
            rerun_with_KGE = True
        else:
            calib_deap = calibration.CalibrationDeap(cfg, model.run, obj.weights, cfg.seed)
            calib_deap.run(subcatch.path, lock_mgr)
            calib_status, reason = obj.process_results()
            if calib_status is False:
                # in case of failed calibration, check the reason and restart 
                if reason == "KGEJSD_Failed":
                    message = "KGEJSD calibration failed, restarting calibration with KGE..."
                    print(message)
                    
                    # Open the file in write mode
                    with open(calibstatus_file_path, 'w') as file:
                        # Write the message to the file
                        file.write(message)

                    rerun_with_KGE = True
                else:
                    raise Exception(f'Error on first calibration run: calib_status is {calib_status}, reason is {reason}\n')

        if rerun_with_KGE is True:
            # recreate settings folder and xml files if not created yet in second run
            lis_template = templates.LisfloodSettingsTemplate(cfg, subcatch)
            
            # change objective
            cfg.deap_param.objectives_list = ['KGE']

            model = hydro_model.HydrologicalModel(cfg, subcatch, lis_template, lock_mgr, obj)

            # load forcings and input maps in cache
            # required in front of processing pool
            # otherwise each child will reload the maps
            model.init_run()

            cfg.filter_param_ranges_after_init(model_initialized=model, split_lake_params=cfg.deap_param.split_lake_params)

            obj = objective.ObjectiveKGE(cfg, subcatch)
            calib_deap = calibration.CalibrationDeap(cfg, model.run, obj.weights, cfg.seed)
            calib_deap.run(subcatch.path, lock_mgr)

            calib_status, reason = obj.process_results(compare_KGSJSD=True)
            if calib_status is False:
                raise Exception(f'Error on second calibration run: calib_status is {calib_status}, reason is {reason}\n')
        

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
