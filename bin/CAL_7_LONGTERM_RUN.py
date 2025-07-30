#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Please refer to quick_guide.pdf for usage instructions"""
import os
import argparse
import numpy as np
import pandas as pd

from liscal import templates, calibration, config, subcatchment, objective, hydro_model, stations, utils



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
                    subperiods, filtered_reservoir_events = stations.process_reservoir_periods(model, reservoir_events_df, dt, None, run_start, run_end, Min_calib_days=None, isLongRun=True)
                else:
                    print("WARNING: reservoir_events csv file not found. Observations will not be filtered by reservoir events")

            hydro_model.generate_outlet_streamflow(cfg, subcatch, lis_template, subperiods, filtered_reservoir_events)
            # additional Checks to STOP downstream catchments calibration 
            # 1) KGElow  KGE<-0.41 (Already applyed in CAL6)
            # 2) HighWaterRemoval TransSub>0.12 & GwLoss >0.9 & GwPerc>1 & b_Xinanjiang<1 & PowerPrefFlow>5 & LowerZoneTimeConstant>500
            # 3) HighTL ratio cumsum(TransmLosslong_term_run.tss)[end]/ (cumsum(rainUpslong_term_run.tss[end)[end]+cumsum(snowUpslong_term_run.tss)[end]) >0.40
            if cfg.deap_param.stop_on_low_kgejsd == True:
                calibstatus_file_path_KGEJSDLow = os.path.join(subcatch.path,'CalibrationStatus_1st_run_KGEJSDLow.txt')
                # check for KGElow  KGE<-0.41 (Already applyed in CAL6)
                if os.path.exists(calibstatus_file_path_KGEJSDLow)==True:
                    out_dir = subcatch.path_out
                    os.rename(os.path.join(out_dir,"streamflow_simulated_best.csv"), os.path.join(out_dir,"streamflow_simulated_best_STOPForLowKGE.csv"))
                    os.rename(os.path.join(out_dir,"streamflow_simulated_best.tss"), os.path.join(out_dir,"streamflow_simulated_best_STOPForLowKGE.tss"))
                    os.rename(os.path.join(out_dir,"chanq_simulated_best.csv"), os.path.join(out_dir,"chanq_simulated_best_STOPForLowKGE.csv"))
                    os.rename(os.path.join(out_dir,"chanq_simulated_best.tss"), os.path.join(out_dir,"chanq_simulated_best_STOPForLowKGE.tss"))
            
                # check for HighWaterRemoval
                try:
                    parameters = hydro_model.read_parameters(subcatch.path)
                    TransSub = parameters[cfg.param_ranges.index.get_loc('TransSub')]
                    GwLoss = parameters[cfg.param_ranges.index.get_loc('GwLoss')]
                    GwPerc = parameters[cfg.param_ranges.index.get_loc('GwPercValue')]
                    b_Xinanjiang = parameters[cfg.param_ranges.index.get_loc('b_Xinanjiang')]
                    PowerPrefFlow = parameters[cfg.param_ranges.index.get_loc('PowerPrefFlow')]
                    LowerZoneTimeConstant = parameters[cfg.param_ranges.index.get_loc('LowerZoneTimeConstant')]

                    if TransSub > 0.12 and GwLoss > 0.9 and GwPerc > 1 and b_Xinanjiang < 1 and PowerPrefFlow > 5 and LowerZoneTimeConstant > 500:
                        try:
                            calibstatus_file_path_HighWaterRemoval = os.path.join(subcatch.path,'CalibrationStatus_HighWaterRemoval.txt')
                            message = "Calibration failed, HighWaterRemoval, STOP here after longterm run..."
                            print(message)
                            
                            # Open the file in write mode
                            with open(calibstatus_file_path_HighWaterRemoval, 'w') as file:
                                # Write the message to the file
                                file.write(message)

                            out_dir = subcatch.path_out
                            os.rename(os.path.join(out_dir,"streamflow_simulated_best.csv"), os.path.join(out_dir,"streamflow_simulated_best_STOPForHighWaterRemoval.csv"))
                            os.rename(os.path.join(out_dir,"streamflow_simulated_best.tss"), os.path.join(out_dir,"streamflow_simulated_best_STOPForHighWaterRemoval.tss"))
                            os.rename(os.path.join(out_dir,"chanq_simulated_best.csv"), os.path.join(out_dir,"chanq_simulated_best_STOPForHighWaterRemoval.csv"))
                            os.rename(os.path.join(out_dir,"chanq_simulated_best.tss"), os.path.join(out_dir,"chanq_simulated_best_STOPForHighWaterRemoval.tss"))
                        except:
                            print("Warning: issue in HighWaterRemoval Stop condition")
                            pass                        
                except:
                    print("Warning: Could not find all parameter for the HighWaterRemoval check, the check has not been applyed")
                    pass
                # check for HighTL
                try:
                    # need to take [1] or we get 2d array
                    transm_loss_data = utils.read_tss(os.path.join(subcatch.path_out, "long_term_run", 'TransmLosslong_term_run.tss'))[1]  
                    rain_data = utils.read_tss(os.path.join(subcatch.path_out, "long_term_run", 'rainUpslong_term_run.tss'))[1]  
                    snow_data = utils.read_tss(os.path.join(subcatch.path_out, "long_term_run", 'snowUpslong_term_run.tss'))[1]  
                    
                    # PCRaster will put 1e31 instead of NaN, set to NaN to catch errors
                    transm_loss_data[transm_loss_data==1e31] = np.nan  
                    rain_data[rain_data==1e31] = np.nan  
                    snow_data[snow_data==1e31] = np.nan  
                    
                    # Compute the cumulative sums
                    cumsum_transm_loss = transm_loss_data.sum()
                    cumsum_rain = rain_data.sum()
                    cumsum_snow = snow_data.sum()

                    # Calculate the ratio
                    ratio = cumsum_transm_loss / (cumsum_rain + cumsum_snow)

                    if ratio > 0.30:
                        try:
                            calibstatus_file_path_HighTL = os.path.join(subcatch.path,'CalibrationStatus_HighTL.txt')
                            message = f"Calibration failed, HighTL (ratio={ratio}), STOP here after longterm run..."
                            print(message)
                            
                            # Open the file in write mode
                            with open(calibstatus_file_path_HighTL, 'w') as file:
                                # Write the message to the file
                                file.write(message)

                            out_dir = subcatch.path_out
                            os.rename(os.path.join(out_dir,"streamflow_simulated_best.csv"), os.path.join(out_dir,"streamflow_simulated_best_STOPForHighTL.csv"))
                            os.rename(os.path.join(out_dir,"streamflow_simulated_best.tss"), os.path.join(out_dir,"streamflow_simulated_best_STOPForHighTL.tss"))
                            os.rename(os.path.join(out_dir,"chanq_simulated_best.csv"), os.path.join(out_dir,"chanq_simulated_best_STOPForHighTL.csv"))
                            os.rename(os.path.join(out_dir,"chanq_simulated_best.tss"), os.path.join(out_dir,"chanq_simulated_best_STOPForHighTL.tss"))
                        except:
                            print("Warning: issue in HighTL Stop condition")
                            pass                        
                except:
                    print("Warning: Could not find all tss files for the HighTL check, the check has not been applyed")
                    pass
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
