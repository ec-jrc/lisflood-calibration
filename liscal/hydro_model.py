import os
import numpy as np
import pandas

import subprocess
import traceback
import random

# lisflood
import lisf1


class HydrologicalModel():

    def __init__(self, cfg, subcatch, lis_template, lock_mgr, objective):

        self.cfg = cfg
        self.subcatch = subcatch

        self.lis_template = lis_template

        self.lock_mgr = lock_mgr

        self.objective = objective

    def init_run(self):

        # dummy Individual, doesn't matter here
        param_ranges = self.cfg.param_ranges
        Individual = 0.5*np.ones(len(param_ranges))

        cfg = self.cfg

        run_rand_id = str(int(random.random()*1e10)).zfill(12)

        parameters = self.objective.get_parameters(Individual)

        self.lis_template.write_template(run_rand_id, self.subcatch.cal_start, self.subcatch.cal_end, cfg.param_ranges, parameters)

        prerun_file = self.lis_template.settings_path('-PreRun', run_rand_id)

        # -i option to exit after initialisation, we just load the inputs map in memory
        try:
            lisf1.main(prerun_file, '-i', '-v')
        except:
            traceback.print_exc()
            raise Exception("")

    def run(self, Individual):

        cfg = self.cfg

        self.lock_mgr.increment_run()
        gen = self.lock_mgr.get_gen()
        run = self.lock_mgr.get_run()

        run_rand_id = str(int(random.random()*1e10)).zfill(12)

        parameters = self.objective.get_parameters(Individual)

        self.lis_template.write_template(run_rand_id, self.subcatch.cal_start, self.subcatch.cal_end, cfg.param_ranges, parameters)

        prerun_file = self.lis_template.settings_path('-PreRun', run_rand_id)
        run_file = self.lis_template.settings_path('-Run', run_rand_id)

        try:
            lisf1.main(prerun_file, '-v') #os.path.realpath(__file__),
            lisf1.main(run_file, '-v')  # os.path.realpath(__file__),
        except:
            traceback.print_exc()
            raise Exception("")

        fKGEComponents = self.objective.compute_objectives(run_rand_id)

        KGE = fKGEComponents[0]

        with self.lock_mgr.lock:
            self.objective.update_parameter_history(run_rand_id, parameters, fKGEComponents, gen, run)

        return fKGEComponents  # If using just one objective function, put a comma at the end!!!


def read_parameters(path_subcatch):

    paramvals = pandas.read_csv(os.path.join(path_subcatch, "pareto_front.csv"),sep=",")

    name_params= paramvals.columns
    names=name_params[3:]
    print('names',names)
    parameters=list()
    for indx in range(0,len(names)):
        print('name[idx]', names[indx],'paramvals',paramvals[names[indx]])
        parameters.append(paramvals[names[indx]].values[0])

    print('param', parameters)

    return parameters


def simulated_best_tss2csv(path_subcatch, run_rand_id, forcing_start, dataname, outname):

    tss_file = os.path.join(path_subcatch, "out", dataname + run_rand_id + '.tss')

    tss = read_tss(tss_file)

    tss[1][tss[1]==1e31] = np.nan
    tss_values = tss[1].values

    df = pandas.DataFrame(data=tss_values, index=pandas.date_range(forcing_start, periods=len(tss_values), freq='6H'))
    df.to_csv(os.path.join(path_subcatch, outname+"_simulated_best.csv"), ',', header="")

    try:
        os.remove(os.path.join(path_subcatch, outname+"_simulated_best.tss"))
    except:
        pass
    os.rename(tss_file, os.path.join(path_subcatch, outname+"_simulated_best.tss"))


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

    stage_inflows(subcatch.path)

    print(">> Running LISFLOOD using the \"best\" parameter set")
    parameters = read_parameters(subcatch.path)

    run_rand_id = str(int(random.random()*1e10)).zfill(12)

    run_start = cfg.forcing_start.strftime('%Y-%m-%d %H:%M')
    run_end = cfg.forcing_end.strftime('%Y-%m-%d %H:%M')
    lis_template.write_template(run_rand_id, run_start, run_end, cfg.param_ranges, parameters)

    # FIRST LISFLOOD RUN
    prerun_file = lis_template.settings_path('-PreRun', run_rand_id)
    lisf1.main(prerun_file, '-v')

    # DD JIRA issue https://efascom.smhi.se/jira/browse/ECC-1210 to avoid overwriting the bestrun avgdis.end.nc
    cmd = "cp " + subcatch.path + "/out/avgdis" + run_rand_id + "end.nc " + subcatch.path + "/out/avgdis" + run_rand_id + ".simulated_bestend.nc"
    utils.run_cmd(cmd)
    cmd = "cp " + subcatch.path + "/out/lzavin" + run_rand_id + "end.nc " + subcatch.path + "/out/lzavin" + run_rand_id + ".simulated_bestend.nc"
    utils.run_cmd(cmd)

    # SECOND LISFLOOD RUN
    run_file = lis_template.settings_path('-Run', run_rand_id)
    lisf1.main(run_file)

    # DD JIRA issue https://efascom.smhi.se/jira/browse/ECC-1210 restore the backup
    cmd = "rm " + subcatch.path + "/out/avgdis" + run_rand_id + "end.nc " + subcatch.path + "/out/lzavin" + run_rand_id + "end.nc"
    utils.run_cmd(cmd)

    simulated_best_tss2csv(subcatch.path, run_rand_id, cfg.forcing_start, 'dis', 'streamflow')
    simulated_best_tss2csv(subcatch.path, run_rand_id, cfg.forcing_start, 'chanq', 'chanq')


def generate_benchmark(cfg, subcatch, lis_template):

    run_rand_id = str(int(random.random()*1e10)).zfill(12)

    param_ranges = cfg.param_ranges
    parameters = [None] * len(param_ranges)
    for ii in range(len(param_ranges)):
        parameters[ii] = 0.5 * (float(param_ranges.iloc[ii, 1]) - float(param_ranges.iloc[ii, 0])) + float(param_ranges.iloc[ii, 0])

    lis_template.write_template(run_rand_id, param_ranges, parameters)

    prerun_file = lis_template('-Prerun')
    run_file = lis_template('-Run')

    lisf1.main(prerun_file, '-v')
    lisf1.main(run_file, '-v')

    Qsim_tss = os.path.join(subcatch.path, "out", 'dis' + run_rand_id + '.tss')
    simulated_streamflow = read_tss(Qsim_tss)
    simulated_streamflow[1][simulated_streamflow[1] == 1e31] = np.nan
    Qsim = simulated_streamflow[1].values
    print( ">> Saving simulated streamflow with default parameters(convergenceTester.csv)")
    # DD DEBUG try shorter time series for testing convergence
    Qsim = pandas.DataFrame(data=Qsim, index=pandas.date_range(subcatch.cal_start, periods=len(Qsim), freq='6H'))
    #Qsim = pandas.DataFrame(data=Qsim, index=pandas.date_range(ForcingStart, periods=len(Qsim), freq='6H'))
    Qsim.to_csv(os.path.join(subcatch.path, "convergenceTester.csv"), ',', header="")

    return Qsim
