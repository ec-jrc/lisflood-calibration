#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Please refer to quick_guide.pdf for usage instructions"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
from configparser import ConfigParser # Python 3.8
import glob
import subprocess
import traceback

from liscal import templates, calibration, config, subcatchment, objective, hydro_model, stations



def longtermrun_subcatchment(cfg, obsid, station_data):

    print("=================== "+str(obsid)+" ====================")
    if os.path.exists(os.path.join(subcatch.path, "out", "streamflow_simulated_best.csv")) or \
        os.path.exists(os.path.join(subcatch.path, "out", "streamflow_simulated_best_STOPForLowKGE.csv")):
        print("streamflow_simulated_best.csv already exists! Moving on...")
        return

    lis_template = templates.LisfloodSettingsTemplate(cfg, subcatch)
    
    if os.path.exists(os.path.join(subcatch.path,"pareto_front.csv"))==True:

        lock_mgr = calibration.LockManager(cfg.num_cpus)

        obj = objective.ObjectiveKGE(cfg, subcatch)

        model = hydro_model.HydrologicalModel(cfg, subcatch, lis_template, lock_mgr, obj)

        # load forcings and input maps in cache
        # required in front of processing pool
        # otherwise each child will reload the maps
        model.init_run()

        cfg.filter_param_ranges_after_init(model_initialized=model, split_lake_params=cfg.deap_param.split_lake_params)  
        dt=cfg.timestep/60      # in the long term run we use dt to set the shift of the starting period, instead of counting the observation period lenght.
                                # Thus this value is now taken from the configuration timestep.
        subperiods = None
        filtered_reservoir_events = None
        if model is not None:
            if cfg.reservoir_events is not None:
                if os.path.exists(cfg.reservoir_events):
                    reservoir_events_df = pd.read_csv(cfg.reservoir_events)
                    run_start = cfg.forcing_start.strftime('%d/%m/%Y %H:%M')
                    run_end = cfg.forcing_end.strftime('%d/%m/%Y %H:%M')
                    subperiods, filtered_reservoir_events = stations.process_reservoir_periods(model, reservoir_events_df, dt, None, run_start, run_end, min_years=None, isLongRun=True)
                else:
                    print("WARNING: reservoir_events csv file not found. Observations will not be filtered by reservoir events")

            hydro_model.generate_outlet_streamflow(cfg, subcatch, lis_template, subperiods, filtered_reservoir_events)
            calibstatus_file_path_KGEJSDLow = os.path.join(subcatch.path,'CalibrationStatus_1st_run_KGEJSDLow.txt')
            if os.path.exists(calibstatus_file_path_KGEJSDLow)==True:
                out_dir = subcatch.path_out
                os.rename(os.path.join(out_dir,"streamflow_simulated_best.csv"), os.path.join(out_dir,"streamflow_simulated_best_STOPForLowKGE.csv"))
                os.rename(os.path.join(out_dir,"streamflow_simulated_best.tss"), os.path.join(out_dir,"streamflow_simulated_best_STOPForLowKGE.tss"))
                os.rename(os.path.join(out_dir,"chanq_simulated_best.csv"), os.path.join(out_dir,"chanq_simulated_best_STOPForLowKGE.csv"))
                os.rename(os.path.join(out_dir,"chanq_simulated_best.tss"), os.path.join(out_dir,"chanq_simulated_best_STOPForLowKGE.tss"))
            

        else:
            raise Exception('Could not find initialize model.')
    else:
        raise Exception('Could not find optimnal parameters for long term run. Please calibrate to generate pareto_front.csv first.')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('settings_file', help='Calibration settings file')
    parser.add_argument('station', help='Station OBSID to process')
    args = parser.parse_args()

    settings_file = args.settings_file

    cfg = config.ConfigCalibration(settings_file)

    # Long term run for specified station
    obsid = int(args.station)

    subcatch = subcatchment.SubCatchment(cfg, obsid)

    longtermrun_subcatchment(cfg, obsid, subcatch)

    print("==================== END ====================")
