#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Please refer to quick_guide.pdf for usage instructions"""
import os
import argparse

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
        rerun_with_KGEJSD = False
        original_seed = cfg.seed
        calibstatus_file_path_KGEJSD = os.path.join(subcatch.path,'CalibrationStatus_2nd_run_KGEJSD.txt')
        calibstatus_file_path_KGE = os.path.join(subcatch.path,'CalibrationStatus_3rd_run_KGE.txt')
        if os.path.exists(calibstatus_file_path_KGEJSD)==True and os.path.exists(calibstatus_file_path_KGE)==False:
            rerun_with_KGEJSD = True
        else:
            if os.path.exists(calibstatus_file_path_KGE)==True:
                rerun_with_KGE = True
            else:
                calib_deap = calibration.CalibrationDeap(cfg, model.run, obj.weights, cfg.seed)
                calib_deap.run(subcatch.path, lock_mgr)
                calib_status, reason = obj.process_results()
                if calib_status is False:
                    # in case of failed calibration, check the reason and restart 
                    if reason == "KGEJSD_Failed":
                        message = "KGEJSD 1st calibration failed, restarting calibration with KGEJSD new seed..."
                        print(message)
                        
                        # Open the file in write mode
                        with open(calibstatus_file_path_KGEJSD, 'w') as file:
                            # Write the message to the file
                            file.write(message)

                        rerun_with_KGEJSD = True
                        del calib_deap
                        del model
                    else:
                        if reason == "KGEJSD_Low":
                            # Execute longterm run and stop
                            rerun_with_KGEJSD = False
                            del calib_deap
                            del model
                        else:
                            raise Exception(f'Error on first calibration run: calib_status is {calib_status}, reason is {reason}\n')
                        
        if rerun_with_KGEJSD is True:
            # recreate settings folder and xml files if not created yet in second run
            lis_template = templates.LisfloodSettingsTemplate(cfg, subcatch)
            
            # change seed, keep KGEJSD
            cfg.seed = 233

            obj = objective.ObjectiveKGE(cfg, subcatch)

            model = hydro_model.HydrologicalModel(cfg, subcatch, lis_template, lock_mgr, obj)

            # load forcings and input maps in cache
            # required in front of processing pool
            # otherwise each child will reload the maps
            model.init_run()

            calib_deap = calibration.CalibrationDeap(cfg, model.run, obj.weights, cfg.seed)
            calib_deap.run(subcatch.path, lock_mgr)

            calib_status, reason = obj.process_results(runType = "KGEJSD_2nd")
            if calib_status is False:
                # in case of failed calibration, check the reason and restart 
                if reason == "KGEJSD_Failed":
                    message = "KGEJSD 2nd calibration failed, restarting calibration with KGE..."
                    print(message)
                    
                    # Open the file in write mode
                    with open(calibstatus_file_path_KGE, 'w') as file:
                        # Write the message to the file
                        file.write(message)

                    rerun_with_KGE = True
                    del calib_deap
                    del model
                else:
                    raise Exception(f'Error on first calibration run: calib_status is {calib_status}, reason is {reason}\n')

        if rerun_with_KGE is True:
            # recreate settings folder and xml files if not created yet in second run
            lis_template = templates.LisfloodSettingsTemplate(cfg, subcatch)
            
            # change objective, keep original seed
            cfg.deap_param.objectives_list = ['KGE']
            cfg.seed = original_seed

            obj = objective.ObjectiveKGE(cfg, subcatch)

            model = hydro_model.HydrologicalModel(cfg, subcatch, lis_template, lock_mgr, obj)

            # load forcings and input maps in cache
            # required in front of processing pool
            # otherwise each child will reload the maps
            model.init_run()

            calib_deap = calibration.CalibrationDeap(cfg, model.run, obj.weights, cfg.seed)
            calib_deap.run(subcatch.path, lock_mgr)

            calib_status, reason = obj.process_results(compare_KGSJSD=True)
            if calib_status is False:
                raise Exception(f'Error on third calibration run: calib_status is {calib_status}, reason is {reason}\n')
        

    else:
        print("pareto_front.csv already exists! Moving on...")


def main(settings_file, station, n_cpus=1, seed=None):
    """Run calibration for a specified station.

    Parameters
    ----------
    settings_file : str
        Path to calibration settings file.
    station : str or int
        Station OBSID to process.
    n_cpus : int, optional
        Number of CPUs to use (default 1).
    seed : int or str or None, optional
        Seed value for random numbers generation in DEAP.
    """

    print('Running calibration using {} cpus'.format(n_cpus))

    cfg = config.ConfigCalibration(settings_file, n_cpus, seed)

    obsid = int(station)

    subcatch = subcatchment.SubCatchment(cfg, obsid)

    calibrate_subcatchment(cfg, obsid, subcatch)

    print("==================== END ====================")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('settings_file', help='Calibration settings file')
    parser.add_argument('station', help='Station OBSID to process')
    parser.add_argument('n_cpus', help='Number of cpus')
    parser.add_argument('--seed', help='Seed value for random numbers generation in deap')
    args = parser.parse_args()

    main(args.settings_file, args.station, args.n_cpus, args.seed)
