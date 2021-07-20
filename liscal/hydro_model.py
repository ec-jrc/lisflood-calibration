import os
import shutil
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

import subprocess
import traceback
import random

# lisflood
import lisf1

from liscal import utils


class HydrologicalModel():

    def __init__(self, cfg, subcatch, lis_template, lock_mgr, objective):

        self.cfg = cfg
        self.subcatch = subcatch

        self.lis_template = lis_template

        self.lock_mgr = lock_mgr

        self.objective = objective

        if cfg.fast_debug:
            self.obs_start = datetime.strptime(subcatch.data['Split_date'],"%d/%m/%Y %H:%M").strftime('%d/%m/%Y %H:%M')
            self.obs_end = (datetime.strptime(self.obs_start,"%d/%m/%Y %H:%M") + timedelta(days=120)).strftime('%d/%m/%Y %H:%M')
            self.cal_start = self.obs_start
            self.cal_end = self.obs_end
        else:
            spinup = int(subcatch.data['Spinup_days'])
            self.obs_start = datetime.strptime(subcatch.data['Split_date'],"%d/%m/%Y %H:%M").strftime('%d/%m/%Y %H:%M')
            self.obs_end = datetime.strptime(subcatch.data['Obs_end'],"%d/%m/%Y %H:%M").strftime('%d/%m/%Y %H:%M')
            self.cal_start = (datetime.strptime(self.obs_start,"%d/%m/%Y %H:%M") - timedelta(days=spinup)).strftime('%d/%m/%Y %H:%M')
            self.cal_end = datetime.strptime(subcatch.data['Obs_end'],"%d/%m/%Y %H:%M").strftime('%d/%m/%Y %H:%M')

    def init_run(self):

        # dummy Individual, doesn't matter here
        param_ranges = self.cfg.param_ranges
        Individual = 0.5*np.ones(len(param_ranges))

        cfg = self.cfg

        run_id = str(0)

        parameters = self.objective.get_parameters(Individual)

        self.lis_template.write_template(run_id, self.cal_start, self.cal_end, cfg.param_ranges, parameters)

        prerun_file = self.lis_template.settings_path('-PreRun', run_id)

        # -i option to exit after initialisation, we just load the inputs map in memory
        try:
            lisf1.main(prerun_file, '-i')
        except:
            traceback.print_exc()
            raise Exception("")

    def run(self, Individual):

        cfg = self.cfg

        gen = self.lock_mgr.get_gen()
        run = self.lock_mgr.increment_run()
        print('Generation {}, run {}'.format(gen, run))

        run_id = '{}_{}'.format(gen, run)

        parameters = self.objective.get_parameters(Individual)

        self.lis_template.write_template(run_id, self.cal_start, self.cal_end, cfg.param_ranges, parameters)

        prerun_file = self.lis_template.settings_path('-PreRun', run_id)
        run_file = self.lis_template.settings_path('-Run', run_id)

        try:
            lisf1.main(prerun_file, '-v')
            lisf1.main(run_file, '-v')
        except:
            traceback.print_exc()
            raise Exception("")

        simulated_streamflow = self.objective.read_simulated_streamflow(run_id, self.cal_start, self.cal_end)
        objectives = self.objective.compute_objectives(run_id, self.obs_start, self.obs_end, simulated_streamflow)

        with self.lock_mgr.lock:
            self.objective.update_parameter_history(run_id, parameters, objectives, gen, run)

        return objectives  # If using just one objective function, put a comma at the end!!!


def read_parameters(path_subcatch):

    paramvals = pd.read_csv(os.path.join(path_subcatch, "pareto_front.csv"),sep=",")

    name_params= paramvals.columns
    names=name_params[3:]
    print('names',names)
    parameters=list()
    for indx in range(0,len(names)):
        print('name[idx]', names[indx],'paramvals',paramvals[names[indx]])
        parameters.append(paramvals[names[indx]].values[0])

    print('param', parameters)

    return parameters


def simulated_best_tss2csv(cfg, subcatch, run_id, forcing_start, dataname, outname):

    tss_file = os.path.join(subcatch.path_out, dataname + run_id + '.tss')

    tss = utils.read_tss(tss_file)

    tss[1][tss[1]==1e31] = np.nan
    tss_values = tss[1].values

    if cfg.calibration_freq == '6-hourly':
        freq = '6H'
    elif cfg.calibration_freq == 'daily':
        freq = 'D'
    else:
        raise Exception('Calibration freq {} not supported'.format(cfg.calibration_freq))

    index = pd.date_range(forcing_start, periods=len(tss_values), freq=freq).strftime('%d/%m/%Y %H:%M')
    df = pd.DataFrame(data=tss_values, index=index)
    df.columns = [str(subcatch.obsid)]
    df.index.name = 'Timestamp'
    df.to_csv(os.path.join(subcatch.path_out, outname+"_simulated_best.csv"))

    try:
        os.remove(os.path.join(subcatch.path_out, outname+"_simulated_best.tss"))
    except:
        pass
    shutil.copy(tss_file, os.path.join(subcatch.path_out, outname+"_simulated_best.tss"))


def stage_inflows(path_subcatch):

    inflow_tss = os.path.join(path_subcatch, "inflow", "chanq.tss")
    inflow_tss_last_run = os.path.join(path_subcatch, "inflow", "chanq_last_run.tss")
    inflow_tss_cal = os.path.join(path_subcatch, "inflow", "chanq_cal.tss")
    if os.path.isfile(inflow_tss) or os.path.isfile(inflow_tss_cal):
        print(inflow_tss)
        print(inflow_tss_cal)
        print(inflow_tss_last_run)
        os.rename(inflow_tss, inflow_tss_cal)
        os.rename(inflow_tss_last_run, inflow_tss)


def generate_outlet_streamflow(cfg, subcatch, lis_template):

    # stage_inflows(subcatch.path)

    print(">> Running LISFLOOD using the \"best\" parameter set")
    parameters = read_parameters(subcatch.path)

    run_id = 'X'

    run_start = cfg.forcing_start.strftime('%d/%m/%Y %H:%M')
    run_end = cfg.forcing_end.strftime('%d/%m/%Y %H:%M')
    lis_template.write_template(run_id, run_start, run_end, cfg.param_ranges, parameters)

    # FIRST LISFLOOD RUN
    prerun_file = lis_template.settings_path('-PreRun', run_id)
    lisf1.main(prerun_file, '-v')

    # DD JIRA issue https://efascom.smhi.se/jira/browse/ECC-1210 to avoid overwriting the bestrun avgdis.end.nc
    cmd = 'cp {0}/out/avgdis{1}end.nc {0}/out/avgdis{1}.simulated_bestend.nc'.format(subcatch.path, run_id)
    utils.run_cmd(cmd)
    cmd = 'cp {0}/out/lzavin{1}end.nc {0}/out/lzavin{1}.simulated_bestend.nc'.format(subcatch.path, run_id)
    utils.run_cmd(cmd)

    # SECOND LISFLOOD RUN
    run_file = lis_template.settings_path('-Run', run_id)
    lisf1.main(run_file, 'q')

    # DD JIRA issue https://efascom.smhi.se/jira/browse/ECC-1210 restore the backup
    cmd = 'rm {0}/out/avgdis{1}end.nc {0}/out/lzavin{1}end.nc'.format(subcatch.path, run_id)
    utils.run_cmd(cmd)

    simulated_best_tss2csv(cfg, subcatch, run_id, cfg.forcing_start, 'dis', 'streamflow')
    simulated_best_tss2csv(cfg, subcatch, run_id, cfg.forcing_start, 'chanq', 'chanq')


def generate_benchmark(cfg, subcatch, lis_template, param_target, outfile, start, end):

    run_id = 'X'

    param_ranges = cfg.param_ranges
    parameters = [None] * len(param_ranges)
    for ii in range(len(param_ranges)):
        parameters[ii] = param_target[ii] * (float(param_ranges.iloc[ii, 1]) - float(param_ranges.iloc[ii, 0])) + float(param_ranges.iloc[ii, 0])
        
    lis_template.write_template(run_id, start, end, param_ranges, parameters)

    prerun_file = lis_template.settings_path('-PreRun', run_id)
    run_file = lis_template.settings_path('-Run', run_id)

    lisf1.main(prerun_file, '-v')
    lisf1.main(run_file, '-q')

    # Outputing synthetic observed discharge
    print( ">> Saving simulated streamflow with default parameters in {}".format(outfile))
    Qsim_tss = os.path.join(subcatch.path, "out", 'dis' + run_id + '.tss')
    simulated_streamflow = utils.read_tss(Qsim_tss)
    simulated_streamflow[1][simulated_streamflow[1] == 1e31] = np.nan
    Qsim = simulated_streamflow[1].values
    index = pd.to_datetime(pd.date_range(start, end, freq='6H'), format='%d/%m/%Y %H:%M', errors='raise')
    Qsim = pd.DataFrame(data=Qsim, index=index)
    Qsim.columns = [str(subcatch.obsid)]
    Qsim.index.name = 'Timestamp'
    Qsim.to_csv(outfile, ',', date_format='%d/%m/%Y %H:%M')

    # required for downstream catchments
    simulated_best_tss2csv(cfg, subcatch, run_id, start, 'dis', 'streamflow')
    simulated_best_tss2csv(cfg, subcatch, run_id, start, 'chanq', 'chanq')
