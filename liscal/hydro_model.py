import os
import shutil
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import time

import subprocess
import traceback
import random

# lisflood
import lisf1
from lisflood.global_modules.decorators import Cache

from liscal import utils


class HydrologicalModel():

    def __init__(self, cfg, subcatch, lis_template, objective):

        self.subcatch = subcatch

        self.lis_template = lis_template

        self.objective = objective
        self.cfg = cfg

        if cfg.fast_debug:
            self.obs_start = datetime.strptime(subcatch.data['Split_date'],"%d/%m/%Y %H:%M").strftime('%d/%m/%Y %H:%M')
            self.obs_end = (datetime.strptime(self.obs_start,"%d/%m/%Y %H:%M") + timedelta(days=120)).strftime('%d/%m/%Y %H:%M')
            self.cal_start = self.obs_start
            self.cal_end = self.obs_end
        else:
            spinup = int(float(subcatch.data['Spinup_days']))
            self.obs_start = datetime.strptime(subcatch.data['Split_date'],"%d/%m/%Y %H:%M").strftime('%d/%m/%Y %H:%M')
            self.obs_end = datetime.strptime(subcatch.data['Obs_end'],"%d/%m/%Y %H:%M").strftime('%d/%m/%Y %H:%M')
            self.cal_start = (datetime.strptime(self.obs_start,"%d/%m/%Y %H:%M") - timedelta(days=spinup)).strftime('%d/%m/%Y %H:%M')
            self.cal_end = datetime.strptime(subcatch.data['Obs_end'],"%d/%m/%Y %H:%M").strftime('%d/%m/%Y %H:%M')

    def init_run(self, run_id):

        # dummy Individual, doesn't matter here
        param_ranges = self.objective.param_ranges
        Individual = 0.5*np.ones(len(param_ranges))

        parameters = self.objective.get_parameters(Individual)

        prerun_file, run_file = self.lis_template.write_init(run_id, self.cfg.forcing_start.strftime('%d/%m/%Y %H:%M'), self.cfg.forcing_end.strftime('%d/%m/%Y %H:%M'), self.cal_start, self.cal_end, param_ranges, parameters)

        # -i option to exit after initialisation, we just load the inputs map in memory
        lisf1.main(prerun_file, '-i')
        
        lisf1.main(run_file, '-i')

        # store lisflood cache size to make sure we don't load anything else after that
        self.lisflood_cache_size = Cache.size()

    def run(self, individual):

        gen = individual['gen']
        run = individual['id'] 
        run_id = '{}_{}'.format(gen, run)
        print('Generation {}, run {}'.format(gen, run))

        parameters = self.objective.get_parameters(individual['value'])

        self.lis_template.write_template(run_id, self.cal_start, self.cal_end, self.objective.param_ranges, parameters)

        prerun_file = self.lis_template.settings_path('-PreRun', run_id)
        run_file = self.lis_template.settings_path('-Run', run_id)

        try:
            lisf1.main(prerun_file, '-v')
            lisf1.main(run_file, '-v')
        except:
            traceback.print_exc()
            raise Exception("Lisflood failed!")

        # check lisflood cache size to make sure we don't load the same map multiple times
        cache_size = Cache.size()
        assert cache_size == self.lisflood_cache_size

        simulated_streamflow = self.objective.read_simulated_streamflow(run_id, self.cal_start, self.cal_end)
        objectives = self.objective.compute_objectives(run_id, self.obs_start, self.obs_end, simulated_streamflow)

        return objectives  # If using just one objective function, put a comma at the end!!!


def read_parameters(path_subcatch):

    paramvals = pd.read_csv(os.path.join(path_subcatch, "pareto_front.csv"),sep=",")

    name_params= paramvals.columns
    names=name_params[3:]
    parameters=list()
    for indx in range(0,len(names)):
        print(names[indx], paramvals[names[indx]].values[0])
        parameters.append(paramvals[names[indx]].values[0])

    return parameters


def simulated_best_tss2csv(cfg, subcatch, run_id, forcing_start, dataname, outname):

    tss_file = os.path.join(subcatch.path_out, dataname + run_id + '.tss')

    tss = utils.read_tss(tss_file)

    tss[1][tss[1]==1e31] = np.nan
    tss_values = tss[1].values

    freq = '{}min'.format(cfg.timestep)

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


def generate_timing(cfg, subcatch, lis_template, param_target, outfile, start, end):

    run_id = 'T'

    param_ranges = cfg.param_ranges
    parameters = [None] * len(param_ranges)
    for ii in range(len(param_ranges)):
        parameters[ii] = param_target[ii] * (float(param_ranges.iloc[ii, 1]) - float(param_ranges.iloc[ii, 0])) + float(param_ranges.iloc[ii, 0])
        
    lis_template.write_template(run_id, start, end, param_ranges, parameters)

    prerun_file = lis_template.settings_path('-PreRun', run_id)
    run_file = lis_template.settings_path('-Run', run_id)

    # cache first
    f = open("timings.csv", "w")
    f.write('obsID, cache, prerun, run\n{},'.format(subcatch.obsid))
    f.close()
    t0 = time.time()
    lisf1.main(prerun_file, '-i')
    t1 = time.time()
    print('\ncaching done in {}\n'.format(t1-t0)) 
    f = open("timings.csv", "a")
    f.write('{},'.format(t1-t0))
    f.close()
    t2 = time.time()
    lisf1.main(prerun_file, '-v')
    t3 = time.time()
    print('\nprerun done in {}\n'.format(t3-t2)) 
    f = open("timings.csv", "a")
    f.write('{},'.format(t3-t2))
    f.close()
    t4 = time.time()
    lisf1.main(run_file, '-q')
    t5 = time.time()
    print('\nrun done in {}\n'.format(t5-t4)) 
    f = open("timings.csv", "a")
    f.write('{}\n'.format(t5-t4))
    f.close()

def generate_benchmark(cfg, subcatch, lis_template, param_target, outfile, start, end):

    run_id = 'Z'

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
