#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Please refer to quick_guide.pdf for usage instructions"""
import os
import argparse
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from liscal import templates, calibration, config, subcatchment, objective, hydro_model


@dataclass
class CalibrationQualityResult:
    """Internal data transfer object for calibration quality evaluation results."""
    success: bool
    reason: str  # "", "KGEJSD_Low", "KGEJSD_Failed"
    pHistory_ranked: pd.DataFrame
    # For threshold evaluation context:
    KGE_best: Optional[float]
    CORR_best: Optional[float]
    JSD_best: Optional[float]
    maxCORR: Optional[float]
    maxKGE: Optional[float]


def rename_calibration_run(subcatch_path, path_out, suffix):
    """Rename calibration run outputs (out folder, settings, CSVs) with a suffix.

    Parameters
    ----------
    subcatch_path : str
        Path to the subcatchment directory.
    path_out : str
        Path to the output directory (e.g., subcatch.path_out).
    suffix : str
        Suffix to append (e.g., "KGEJSD_1st", "KGE").

    Returns
    -------
    str
        The renamed path_out directory path.
    """
    renamed_pathout = path_out + "_" + suffix
    os.rename(path_out, renamed_pathout)
    settings_dir = os.path.join(subcatch_path, 'settings')
    renamed_settings_dir = settings_dir + "_" + suffix
    os.rename(settings_dir, renamed_settings_dir)
    os.rename(os.path.join(subcatch_path, "pHistoryWRanks.csv"), os.path.join(subcatch_path, "pHistoryWRanks_" + suffix + ".csv"))
    os.rename(os.path.join(subcatch_path, "paramsHistory.csv"), os.path.join(subcatch_path, "paramsHistory_" + suffix + ".csv"))
    os.rename(os.path.join(subcatch_path, "front_history.csv"), os.path.join(subcatch_path, "front_history_" + suffix + ".csv"))
    os.rename(os.path.join(subcatch_path, "runs_log.csv"), os.path.join(subcatch_path, "runs_log_" + suffix + ".csv"))
    return renamed_pathout


def resume_kgejsd_run(subcatch_path, path_out, bestKGEJSDtoResume):
    """Resume KGEJSD calibration run after KGE comparison.

    Renames the KGEJSD_{bestKGEJSDtoResume} folders/files back to the active names.

    Parameters
    ----------
    subcatch_path : str
        Path to the subcatchment directory.
    path_out : str
        Path to the output directory (e.g., subcatch.path_out).
    bestKGEJSDtoResume : str
        Which KGEJSD run to resume ("1st" or "2nd").
    """
    resumed_pathout = path_out + "_KGEJSD_" + bestKGEJSDtoResume
    os.rename(resumed_pathout, path_out)
    settings_dir = os.path.join(subcatch_path, 'settings')
    resumed_settings_dir = settings_dir + "_KGEJSD_" + bestKGEJSDtoResume
    os.rename(resumed_settings_dir, settings_dir)

    os.rename(os.path.join(subcatch_path, "pareto_front_KGEJSD_" + bestKGEJSDtoResume + ".csv"), os.path.join(subcatch_path, "pareto_front.csv"))
    os.rename(os.path.join(subcatch_path, "pHistoryWRanks_KGEJSD_" + bestKGEJSDtoResume + ".csv"), os.path.join(subcatch_path, "pHistoryWRanks.csv"))
    os.rename(os.path.join(subcatch_path, "paramsHistory_KGEJSD_" + bestKGEJSDtoResume + ".csv"), os.path.join(subcatch_path, "paramsHistory.csv"))
    os.rename(os.path.join(subcatch_path, "front_history_KGEJSD_" + bestKGEJSDtoResume + ".csv"), os.path.join(subcatch_path, "front_history.csv"))
    os.rename(os.path.join(subcatch_path, "runs_log_KGEJSD_" + bestKGEJSDtoResume + ".csv"), os.path.join(subcatch_path, "runs_log.csv"))


def select_best_calib(JSD_1st, JSD_2nd, CORR_1st, CORR_2nd, KGE_1st, KGE_2nd,
                      CORR_bestKGE, KGE_bestKGE):
    """Decide whether KGE or KGEJSD calibration is better.

    Pure function extracted from Objective.select_best_calib. Compares
    the two KGEJSD runs and the KGE run to determine which should be used
    for the long-term run.

    Parameters
    ----------
    JSD_1st : float
        JSD value from the 1st KGEJSD calibration run.
    JSD_2nd : float
        JSD value from the 2nd KGEJSD calibration run.
    CORR_1st : float
        Correlation from the 1st KGEJSD calibration run.
    CORR_2nd : float
        Correlation from the 2nd KGEJSD calibration run.
    KGE_1st : float
        KGE from the 1st KGEJSD calibration run.
    KGE_2nd : float
        KGE from the 2nd KGEJSD calibration run.
    CORR_bestKGE : float
        Correlation from the KGE calibration run.
    KGE_bestKGE : float
        KGE from the KGE calibration run.

    Returns
    -------
    tuple of (bool, str, float, float, float)
        (select_KGE, bestKGEJSDtoResume, KGE_bestKGEJSD, CORR_bestKGEJSD, JSD_bestKGEJSD)
    """
    # Determine which KGEJSD run to select
    if ((JSD_1st <= 0.1) and (JSD_2nd <= 0.1)) or ((JSD_1st > 0.1) and (JSD_2nd > 0.1)):
        if (CORR_1st - CORR_2nd > 0.05) or ((abs(CORR_1st - CORR_2nd) <= 0.05) and (KGE_1st > KGE_2nd)):
            KGE_bestKGEJSD = KGE_1st
            CORR_bestKGEJSD = CORR_1st
            JSD_bestKGEJSD = JSD_1st
            bestKGEJSDtoResume = "1st"
        else:
            KGE_bestKGEJSD = KGE_2nd
            CORR_bestKGEJSD = CORR_2nd
            JSD_bestKGEJSD = JSD_2nd
            bestKGEJSDtoResume = "2nd"
    else:
        if JSD_1st <= 0.1:
            KGE_bestKGEJSD = KGE_1st
            CORR_bestKGEJSD = CORR_1st
            JSD_bestKGEJSD = JSD_1st
            bestKGEJSDtoResume = "1st"
        else:
            KGE_bestKGEJSD = KGE_2nd
            CORR_bestKGEJSD = CORR_2nd
            JSD_bestKGEJSD = JSD_2nd
            bestKGEJSDtoResume = "2nd"

    # Determine if the long run should execute using KGE
    select_KGE = ((JSD_bestKGEJSD > 0.1) and ((CORR_bestKGE > CORR_bestKGEJSD) or (KGE_bestKGE > KGE_bestKGEJSD))) or \
                 ((JSD_bestKGEJSD <= 0.1) and ((CORR_bestKGE - CORR_bestKGEJSD > 0.05) or (KGE_bestKGE - KGE_bestKGEJSD > 0.05)))

    return select_KGE, bestKGEJSDtoResume, KGE_bestKGEJSD, CORR_bestKGEJSD, JSD_bestKGEJSD


def evaluate_calibration_quality(obj, run_type="KGEJSD_1st", compare_KGEJSD=False):
    """Evaluate quality of a completed calibration run.

    Extracted from Objective.process_results. Reads param history, computes
    ranked solution, checks quality thresholds, writes calibration status
    files and handles all orchestration decisions.

    Parameters
    ----------
    obj : ObjectiveKGEBase
        The objective instance (provides read_param_history, write_ranked_solution,
        write_pareto_front, write_summary_file, is_kge_jsd, cfg, subcatch).
    run_type : str
        Type of the current run (e.g., "KGEJSD_1st", "KGEJSD_2nd").
    compare_KGEJSD : bool
        Whether to compare KGE vs KGEJSD results (3rd run comparison).

    Returns
    -------
    tuple of (bool, str)
        (success, reason) where success=False means calibration failed.
    """
    pHistory = obj.read_param_history()
    pHistory_ranked = obj.write_ranked_solution(pHistory)

    isKGE_JSD = obj.is_kge_jsd

    if isKGE_JSD:   # in this case we perform the check on Correlation
        maxCORRhistory = pHistory["Correlation"].max()
        maxKGEhistory = pHistory["Kling Gupta Efficiency"].max()

        bestParetoIndex = pHistory_ranked["paretoRank"].nsmallest(1).index
        CORR_bestKGEJSD = pHistory_ranked.loc[bestParetoIndex]["Correlation"].values[0]
        KGE_bestKGEJSD = pHistory_ranked.loc[bestParetoIndex]["Kling Gupta Efficiency"].values[0]
        JSD_bestKGEJSD = pHistory_ranked.loc[bestParetoIndex]["JSD"].values[0]

        if KGE_bestKGEJSD < -0.41 and run_type == "KGEJSD_1st" and obj.cfg.deap_param.stop_on_low_kgejsd > 0:
            calibstatus_file_path_KGEJSDLow = os.path.join(obj.subcatch.path, 'CalibrationStatus_1st_run_KGEJSDLow.txt')
            message = "KGEJSD 1st calibration failed, low KGE, running longterm run and STOP here..."
            print(message)
            with open(calibstatus_file_path_KGEJSDLow, 'w') as file:
                file.write(message)

        if KGE_bestKGEJSD < -0.41 and run_type == "KGEJSD_1st" and obj.cfg.deap_param.stop_on_low_kgejsd == 1:
            obj.write_pareto_front(pHistory_ranked, isKGE_JSD)
            obj.write_summary_file(selObjFun=run_type)
            return False, "KGEJSD_Low"
        else:
            if (maxCORRhistory - CORR_bestKGEJSD > 0.095) or (maxKGEhistory - KGE_bestKGEJSD > 0.095) or (JSD_bestKGEJSD > 0.1):
                paretofront_filename = "pareto_front_" + run_type + ".csv"
                message = f"WARNING!\n" \
                    f"Max correlation value in param history = {maxCORRhistory}\n" \
                    f"Selected kge-jsd calibrated item correlation value = {CORR_bestKGEJSD}\n" \
                    f"Max KGE value in param history = {maxKGEhistory}\n" \
                    f"Selected kge-jsd calibrated item KGE value = {KGE_bestKGEJSD}\n" \
                    f"Selected kge-jsd calibrated item JSD value = {JSD_bestKGEJSD}\n" \
                    f"Repeating the calibration with 'KGE objective function'\n" \
                    f"(pareto_front written in {obj.subcatch.path} -> {paretofront_filename})\n"
                print(message)
                obj.write_pareto_front(pHistory_ranked, isKGE_JSD, None, paretofront_filename)
                return False, "KGEJSD_Failed"
    else:
        if compare_KGEJSD:
            pHistory_ranked_KGSJSD_1st = pd.read_csv(os.path.join(obj.subcatch.path, "pHistoryWRanks_KGEJSD_1st.csv"), sep=",")
            pHistory_ranked_KGSJSD_2nd = pd.read_csv(os.path.join(obj.subcatch.path, "pHistoryWRanks_KGEJSD_2nd.csv"), sep=",")

            bestParetoIndex = pHistory_ranked["paretoRank"].nsmallest(1).index
            CORR_bestKGE = pHistory_ranked.loc[bestParetoIndex]["Correlation"].values[0]
            KGE_bestKGE = pHistory_ranked.loc[bestParetoIndex]["Kling Gupta Efficiency"].values[0]

            bestParetoIndex_KGSJSD_1st = pHistory_ranked_KGSJSD_1st["paretoRank"].nsmallest(1).index
            CORR_bestKGEJSD_1st = pHistory_ranked_KGSJSD_1st.loc[bestParetoIndex_KGSJSD_1st]["Correlation"].values[0]
            KGE_bestKGEJSD_1st = pHistory_ranked_KGSJSD_1st.loc[bestParetoIndex_KGSJSD_1st]["Kling Gupta Efficiency"].values[0]
            JSD_bestKGEJSD_1st = pHistory_ranked_KGSJSD_1st.loc[bestParetoIndex_KGSJSD_1st]["JSD"].values[0]

            bestParetoIndex_KGSJSD_2nd = pHistory_ranked_KGSJSD_2nd["paretoRank"].nsmallest(1).index
            CORR_bestKGEJSD_2nd = pHistory_ranked_KGSJSD_2nd.loc[bestParetoIndex_KGSJSD_2nd]["Correlation"].values[0]
            KGE_bestKGEJSD_2nd = pHistory_ranked_KGSJSD_2nd.loc[bestParetoIndex_KGSJSD_2nd]["Kling Gupta Efficiency"].values[0]
            JSD_bestKGEJSD_2nd = pHistory_ranked_KGSJSD_2nd.loc[bestParetoIndex_KGSJSD_2nd]["JSD"].values[0]

            select_KGE, bestKGEJSDtoResume, KGE_bestKGEJSD, CORR_bestKGEJSD, JSD_bestKGEJSD = select_best_calib(
                JSD_bestKGEJSD_1st, JSD_bestKGEJSD_2nd,
                CORR_bestKGEJSD_1st, CORR_bestKGEJSD_2nd,
                KGE_bestKGEJSD_1st, KGE_bestKGEJSD_2nd,
                CORR_bestKGE, KGE_bestKGE)

            if select_KGE:
                obj.write_pareto_front(pHistory_ranked, isKGE_JSD)
                obj.write_summary_file(selObjFun="KGE")
                return True, ""
            else:
                paretofront_filename = "pareto_front_KGE.csv"
                message = f"WARNING!\n" \
                    f"Selected kge calibrated item correlation value = {CORR_bestKGE}\n" \
                    f"Selected kge calibrated item  KGE value = {KGE_bestKGE}\n" \
                    f"Selected kge-jsd 1st calibrated item correlation value = {CORR_bestKGEJSD_1st}\n" \
                    f"Selected kge-jsd 1st calibrated item KGE value = {KGE_bestKGEJSD_1st}\n" \
                    f"Selected kge-jsd 1st calibrated item JSD value = {JSD_bestKGEJSD_1st}\n" \
                    f"Selected kge-jsd 2nd calibrated item correlation value = {CORR_bestKGEJSD_2nd}\n" \
                    f"Selected kge-jsd 2nd calibrated item KGE value = {KGE_bestKGEJSD_2nd}\n" \
                    f"Selected kge-jsd 2nd calibrated item JSD value = {JSD_bestKGEJSD_2nd}\n" \
                    f"Resuming the calibration with 'KGEJSD objective function', {bestKGEJSDtoResume} attempt\n" \
                    f"(pareto_front written in {obj.subcatch.path} -> {paretofront_filename})\n"
                print(message)
                obj.write_pareto_front(pHistory_ranked, isKGE_JSD, None, paretofront_filename)
                obj.write_summary_file(selObjFun="KGEJSD_" + bestKGEJSDtoResume)
                return True, "resume_KGEJSD_" + bestKGEJSDtoResume

    obj.write_pareto_front(pHistory_ranked, isKGE_JSD)
    obj.write_summary_file(selObjFun="KGE" if isKGE_JSD == False else "KGEJSD")
    return True, ""


def calibrate_subcatchment(cfg, obsid, subcatch):

    print("=================== "+str(obsid)+" ====================")
    if os.path.exists(os.path.join(subcatch.path, "out", "streamflow_simulated_best.csv")):
        print("streamflow_simulated_best.csv already exists! Moving on...")
        return

    lis_template = templates.LisfloodSettingsTemplate(cfg, subcatch)
    
    if os.path.exists(os.path.join(subcatch.path,"pareto_front.csv"))==False:
        print(">> Starting calibration of catchment "+str(obsid))

        lock_mgr = calibration.LockManager(cfg.num_cpus)

        obj = objective.create_objective(cfg, subcatch)

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
                calib_status, reason = evaluate_calibration_quality(obj)
                if calib_status is False:
                    # in case of failed calibration, check the reason and restart 
                    if reason == "KGEJSD_Failed":
                        # Rename run outputs before restarting with new seed
                        renamed_pathout = rename_calibration_run(subcatch.path, subcatch.path_out, "KGEJSD_1st")
                        # Write warning message to the renamed out folder
                        file_path = os.path.join(renamed_pathout, "CorrelationIssue_Warning_messageKGEJSD_1st.txt")
                        with open(file_path, 'w') as file:
                            file.write("KGEJSD 1st calibration failed due to threshold violation.\n")

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

            obj = objective.create_objective(cfg, subcatch)

            model = hydro_model.HydrologicalModel(cfg, subcatch, lis_template, lock_mgr, obj)

            # load forcings and input maps in cache
            # required in front of processing pool
            # otherwise each child will reload the maps
            model.init_run()

            calib_deap = calibration.CalibrationDeap(cfg, model.run, obj.weights, cfg.seed)
            calib_deap.run(subcatch.path, lock_mgr)

            calib_status, reason = evaluate_calibration_quality(obj, run_type="KGEJSD_2nd")
            if calib_status is False:
                # in case of failed calibration, check the reason and restart 
                if reason == "KGEJSD_Failed":
                    # Rename run outputs before restarting with KGE
                    renamed_pathout = rename_calibration_run(subcatch.path, subcatch.path_out, "KGEJSD_2nd")
                    # Write warning message to the renamed out folder
                    file_path = os.path.join(renamed_pathout, "CorrelationIssue_Warning_messageKGEJSD_2nd.txt")
                    with open(file_path, 'w') as file:
                        file.write("KGEJSD 2nd calibration failed due to threshold violation.\n")

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

            obj = objective.create_objective(cfg, subcatch)

            model = hydro_model.HydrologicalModel(cfg, subcatch, lis_template, lock_mgr, obj)

            # load forcings and input maps in cache
            # required in front of processing pool
            # otherwise each child will reload the maps
            model.init_run()

            calib_deap = calibration.CalibrationDeap(cfg, model.run, obj.weights, cfg.seed)
            calib_deap.run(subcatch.path, lock_mgr)

            calib_status, reason = evaluate_calibration_quality(obj, compare_KGEJSD=True)
            if calib_status is False:
                raise Exception(f'Error on third calibration run: calib_status is {calib_status}, reason is {reason}\n')
            
            # Handle the resume_KGEJSD case: KGE was not selected, switch back to KGEJSD
            if reason.startswith("resume_KGEJSD_"):
                bestKGEJSDtoResume = reason.replace("resume_KGEJSD_", "")
                # Rename current KGE run outputs
                renamed_pathout = rename_calibration_run(subcatch.path, subcatch.path_out, "KGE")
                # Write warning message to the renamed out folder
                file_path = os.path.join(renamed_pathout, "ResumedKGEJSD_Warning_message_" + bestKGEJSDtoResume + ".txt")
                with open(file_path, 'w') as file:
                    file.write("KGE calibration not selected; resuming KGEJSD " + bestKGEJSDtoResume + " attempt.\n")
                # Resume KGEJSD calibration run
                resume_kgejsd_run(subcatch.path, subcatch.path_out, bestKGEJSDtoResume)
        

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
