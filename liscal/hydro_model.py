import os
import numpy as np
import pandas

import subprocess
from datetime import datetime, timedelta
import traceback
import random

# lisflood
import lisf1

from liscal import hydro_stats


class ObjectiveDischarge():

    def __init__(self, cfg, subcatch):
        self.cfg = cfg
        self.subcatch = subcatch
        self.param_ranges = cfg.param_ranges

        self.observed_streamflow = self.read_observed_streamflow()

    def get_parameters(self, Individual):
        param_ranges = self.param_ranges
        Parameters = [None] * len(param_ranges)
        for ii in range(len(param_ranges)):
            Parameters[ii] = Individual[ii]*(float(param_ranges.iloc[ii,1])-float(param_ranges.iloc[ii,0]))+float(param_ranges.iloc[ii,0])

        return parameters

    def read_observed_streamflow(self):
        cfg = self.cfg
        obsid = self.subcatch.obsid
        # Load observed streamflow # DD Much faster IO with npy despite being more complicated (<1s vs 22s)
        if os.path.exists(cfg.Qtss_csv.replace(".csv", ".npy")) and os.path.getsize(cfg.Qtss_csv) > 0:
            streamflow_data = pandas.DataFrame(np.load(cfg.Qtss_csv.replace(".csv", ".npy"), allow_pickle=True))
            streamflow_datetimes = np.load(cfg.Qtss_csv.replace(".csv", "_dates.npy"), allow_pickle=True).astype('string_')
            try:
                streamflow_data.index = [datetime.strptime(i.decode('utf-8'), "%d/%m/%Y %H:%M") for i in streamflow_datetimes]
            except ValueError:
                try:
                    streamflow_data.index = [datetime.strptime(i.decode('utf-8'), "%Y-%m-%d %H:%M:%S") for i in streamflow_datetimes]
                except ValueError:
                    streamflow_data.index = [datetime.strptime(i.decode('utf-8'), "%Y-%m-%d") for i in streamflow_datetimes]
            streamflow_data.columns = np.load(cfg.Qtss_csv.replace(".csv", "_catchments.npy"), allow_pickle=True)
        else:
            streamflow_data = pandas.read_csv(cfg.Qtss_csv, sep=",", index_col=0)
            # streamflow_data.index = pandas.date_range(start=ObservationsStart, end=ObservationsEnd, periods=len(streamflow_data))
            # streamflow_data = pandas.read_csv(Qtss_csv, sep=",", index_col=0, parse_dates=True) # DD WARNING buggy unreliable parse_dates! Don't use it!
            np.save(cfg.Qtss_csv.replace(".csv", ".npy"), streamflow_data)
            np.save(cfg.Qtss_csv.replace(".csv", "_dates.npy"), streamflow_data.index)
            np.save(cfg.Qtss_csv.replace(".csv", "_catchments.npy"), streamflow_data.columns.values)
        observed_streamflow = streamflow_data[str(obsid)]
        observed_streamflow = observed_streamflow[cfg.forcing_start.strftime('%Y-%m-%d %H:%M'):cfg.forcing_end.strftime('%Y-%m-%d %H:%M')] # Keep only the part for which we run LISFLOOD
        observed_streamflow = observed_streamflow[Cal_Start:Cal_End]

        return observed_streamflow

    def read_simulated_streamflow(self, run_rand_id):
        Qsim_tss = os.path.join(self.subcatch.path, "out", 'dis'+run_rand_id+'.tss')
        if os.path.isfile(Qsim_tss)==False:
            print("run_rand_id: "+str(run_rand_id))
            raise Exception("No simulated streamflow found. Probably LISFLOOD failed to start? Check the log files of the run!")
        simulated_streamflow = read_tss(Qsim_tss)
        simulated_streamflow[1][simulated_streamflow[1]==1e31] = np.nan
        simulated_streamflow.index = [datetime.strptime(self.subcatch.cal_start, "%Y-%m-%d %H:%M") + timedelta(hours=6*i) for i in range(len(simulated_streamflow.index))]

        return simulated_streamflow

    def resample_streamflows(self, simulated_streamflow, observed_streamflow):

        cfg = self.cfg
        cal_start = self.subcatch.cal_start
        cal_end = self.subcatch.cal_end

        # synchronise simulated and observed streamflows - NaN when missing obs
        Q = pandas.concat({"Sim": simulated_streamflow[1], "Obs": self.observed_streamflow}, axis=1)  # .reset_index()

        # Finally, extract equal-length arrays from it
        Qobs = np.array(Q['Obs'][cal_start:cal_end]) #.values+0.001
        Qsim = np.array(Q['Sim'][cal_start:cal_end])

        if cfg.calibration_freq == r"6-hourly":
            # DD: Check if daily or 6-hourly observed streamflow is available
            # DD: Aggregate 6-hourly simulated streamflow to daily ones
            if self.subcatch.data["CAL_TYPE"].find("_24h") > -1:
                # DD: Overwrite index with date range so we can use Pandas' resampling + mean function to easily average 6-hourly to daily data
                Qsim = simulated_streamflow[cal_start:cal_end]
                Qsim.index = pandas.date_range(cal_start, cal_end, freq="360min")
                Qsim = Qsim.resample('24H', label="right", closed="right").mean()
                Qsim = np.array(Qsim) #[1].values + 0.001
                # Same for Qobs
                Qobs = observed_streamflow[cal_start:cal_end]
                Qobs.index = pandas.date_range(cal_start, cal_end, freq="360min")
                Qobs = Qobs.resample('24H', label="right", closed="right").mean()
                Qobs = np.array(Qobs) #[1].values + 0.001

        elif cfg.calibration_freq == r"daily":
            # DD Untested code! DEBUG TODO
            Qobs = observed_streamflow[cal_start:cal_start]
            Qobs.index = pandas.date_range(cal_start, cal_start, freq="360min")
            Qobs = Qobs.resample('24H', label="right", closed="right").mean()
            Qobs = np.array(Qobs) #[1].values + 0.001

        # Trim nans
        Qsim = Qsim[~np.isnan(Qobs)]
        Qobs = Qobs[~np.isnan(Qobs)]

        return Qsim, Qobs

    def compute_KGE(self, Qsim, Qobs):
        cfg = self.cfg

        # Compute objective function score
        # # DD A few attempts with filtering of peaks and low flows
        if cfg.calibration_freq == r"6-hourly":
            # DD: Check if daily or 6-hourly observed streamflow is available
            # DD: Aggregate 6-hourly simulated streamflow to daily ones
            if self.subcatch.data["CAL_TYPE"].find("_24h") > -1:
                fKGEComponents = hydro_stats.fKGE(s=Qsim, o=Qobs, warmup=cfg.WarmupDays, weightedLogWeight=0.0, lowFlowPercentileThreshold=0.0, usePeaksOnly=False)
            else:
                fKGEComponents = hydro_stats.fKGE(s=Qsim, o=Qobs, warmup=4*cfg.WarmupDays, weightedLogWeight=0.0, lowFlowPercentileThreshold=0.0, usePeaksOnly=False)
        elif cfg.calibration_freq == r"daily":
            fKGEComponents = hydro_stats.fKGE(s=Qsim, o=Qobs, warmup=cfg.WarmupDays, weightedLogWeight=0.0, lowFlowPercentileThreshold=0.0, usePeaksOnly=False)

        return fKGEComponents

    def update_parameter_history(self, run_rand_id, parameters, fKGEComponents, gen, run):

        cfg = self.cfg

        KGE = fKGEComponents[0]

        with open(os.path.join(self.subcatch.path, "runs_log.csv"), "a") as myfile:
            myfile.write(str(run_rand_id)+","+str(KGE)+"\n")

        # DD We want to check that the parameter space is properly sampled. Write them out to file now
        paramsHistoryFilename = os.path.join(self.subcatch.path, "paramsHistory.csv")
        if not os.path.exists(paramsHistoryFilename) or os.path.getsize(paramsHistoryFilename) == 0:
            paramsHistoryFile = open(paramsHistoryFilename, "w")
            # Headers
            paramsHistory = "randId,"
            for i in [str(ip) + "," for ip in self.param_ranges.index.values]:
                paramsHistory += i
            for i in [str(ip) + "," for ip in ["Kling Gupta Efficiency", "Correlation", "Signal ratio (s/o) (Bias)", "Noise ratio (s/o) (Spread)", "sae", "generation", "runNumber"]]:
                paramsHistory += i
            paramsHistory += "\n"
            # Minimal values
            paramsHistory += str(self.param_ranges.head().columns.values[0]) + ","
            for i in [str(ip) + "," for ip in self.param_ranges[str(self.param_ranges.head().columns.values[0])].values]:
                paramsHistory += i
            paramsHistory += "\n"
            # Default values
            paramsHistory += str(self.param_ranges.head().columns.values[2]) + ","
            for i in [str(ip) + "," for ip in self.param_ranges[str(self.param_ranges.head().columns.values[2])].values]:
                paramsHistory += i
            paramsHistory += "\n"
            # Maximal values
            paramsHistory += str(self.param_ranges.head().columns.values[1]) + ","
            for i in [str(ip) + "," for ip in self.param_ranges[str(self.param_ranges.head().columns.values[1])].values]:
                paramsHistory += i
            paramsHistory += "\n\n"
        else:
            paramsHistoryFile = open(paramsHistoryFilename, "a")
            paramsHistory = ""
        paramsHistory += str(run_rand_id) + ","
        for i in [str(ip) + "," for ip in parameters]:
            paramsHistory += i
        for i in [str(ip) + "," for ip in fKGEComponents]:
            paramsHistory += i
        paramsHistory += str(gen) + ","
        paramsHistory += str(run)
        paramsHistory += "\n"
        paramsHistoryFile.write(paramsHistory)
        paramsHistoryFile.close()

    def compute_objectives(self, run_rand_id):
        # DD Extract simulation
        simulated_streamflow = self.read_simulated_streamflow(run_rand_id)

        Qsim, Qobs = self.resample_streamflows(simulated_streamflow, self.observed_streamflow)
        if len(Qobs) != len(Qsim):
            raise Exception("run_rand_id: "+str(run_rand_id)+": observed and simulated streamflow arrays have different number of elements ("+str(len(Qobs))+" and "+str(len(Qsim))+" elements, respectively)")

        fKGEComponents = self.compute_KGE(Qsim, Qobs)

        print("   run_rand_id: "+str(run_rand_id)+", KGE: "+"{0:.3f}".format(fKGEComponents[0]))

        return fKGEComponents

class ObjectiveDischargeTest(ObjectiveDischarge):

    def __init__(self, cfg, subcatch, tol):

        self.tol = tol

        super().__init__(cfg, subcatch)

    def read_observed_streamflow(self):
        cfg = self.cfg
        obsid = self.subcatch.obsid

        # Load observed streamflow # DD Much faster IO with npy despite being more complicated (<1s vs 22s)
        streamflow_data = pandas.read_csv(cfg.subcatchment_path + "/" + str(obsid) + "/convergenceTester.csv", sep=",", index_col=0, header=None)
        # streamflow_data.index = pandas.date_range(start=ObservationsStart, end=ObservationsEnd, periods=len(streamflow_data))
        #streamflow_data.index = pandas.date_range(start=ForcingStart, end=ForcingEnd, periods=len(streamflow_data))
        streamflow_data.index = pandas.date_range(start=streamflow_data.index[0], end=streamflow_data.index[-1], periods=len(streamflow_data))
        observed_streamflow = streamflow_data[cfg.forcing_start:cfg.forcing_end]
        return observed_streamflow

    def get_parameters(self, Individual):
        param_ranges = self.cfg.param_ranges
        parameters = [None] * len(param_ranges)
        for ii in range(len(param_ranges)):
          ref = 0.5 * (float(param_ranges.iloc[ii, 1]) - float(param_ranges.iloc[ii, 0])) + float(param_ranges.iloc[ii, 0])
          parameters[ii] = ref * (1+self.tol)
        return parameters


def read_tss(tss_file):
    df = pandas.read_csv(tss_file, sep=r"\s+", index_col=0, skiprows=4, header=None, skipinitialspace=True)
    return df


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

        return KGE, # If using just one objective function, put a comma at the end!!!


class HydrologicalModelBenchmark(HydrologicalModel):
    
    def __init__(self, cfg, subcatch, lis_template, lock_mgr):
        super().__init__(cfg, subcatch, lis_template, lock_mgr)

    def get_parameters(self, Individual):
        param_ranges = self.cfg.param_ranges
        parameters = [None] * len(param_ranges)
        for ii in range(len(param_ranges)):
            parameters[ii] = 0.5 * (float(param_ranges.iloc[ii, 1]) - float(param_ranges.iloc[ii, 0])) + float(param_ranges.iloc[ii, 0])
        return parameters

    def run(self, Individual):

        cfg = self.cfg

        run_rand_id = str(int(random.random()*1e10)).zfill(12)

        parameters = self.get_parameters(Individual)

        self.lis_template.write_template(run_rand_id, cfg.param_ranges, parameters)

        prerun_file = self.lis_template('-Prerun')
        run_file = self.lis_template('-Run')

        lisf1.main(prerun_file, '-v')
        lisf1.main(run_file, '-v')

        Qsim_tss = os.path.join(self, self.subcatch.path, "out", 'dis' + run_rand_id + '.tss')
        simulated_streamflow = read_tss(Qsim_tss)
        simulated_streamflow[1][simulated_streamflow[1] == 1e31] = np.nan
        Qsim = simulated_streamflow[1].values
        print( ">> Saving simulated streamflow with default parameters(convergenceTester.csv)")
        # DD DEBUG try shorter time series for testing convergence
        Qsim = pandas.DataFrame(data=Qsim, index=pandas.date_range(self.subcatch.cal_start, periods=len(Qsim), freq='6H'))
        #Qsim = pandas.DataFrame(data=Qsim, index=pandas.date_range(ForcingStart, periods=len(Qsim), freq='6H'))
        Qsim.to_csv(os.path.join(self, self.subcatch.path, "convergenceTester.csv"), ',', header="")
        
        return Qsim


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


def generate_outlet_streamflow(cfg, subcatch, station_data, lis_template):

    # Select the "best" parameter set and run LISFLOOD for the entire forcing period
    #Parameters = paramvals[best,:]

    stage_inflows(subcatch.path)

    print(">> Running LISFLOOD using the \"best\" parameter set")
    parameters = read_parameters(subcatch.path)

    run_rand_id = str(int(random.random()*1e10)).zfill(12)

    run_start = cfg.forcing_start.strftime('%Y-%m-%d %H:%M')
    run_end = cfg.forcing_end.strftime('%Y-%m-%d %H:%M')
    lis_template.write_template(run_rand_id, run_start, run_end, cfg.param_ranges, parameters)

    ### SECOND LISFLOOD RUN ###

    prerun_file = lis_template.settings_path('-PreRun', run_rand_id)
    lisf1.main(prerun_file, '-v')

    # DD JIRA issue https://efascom.smhi.se/jira/browse/ECC-1210 to avoid overwriting the bestrun avgdis.end.nc
    cmd = "cp " + subcatch.path + "/out/avgdis" + run_rand_id + "end.nc " + subcatch.path + "/out/avgdis" + run_rand_id + ".simulated_bestend.nc"
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stdin=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(p.communicate()[0])
    p.wait()
    cmd = "cp " + subcatch.path + "/out/lzavin" + run_rand_id + "end.nc " + subcatch.path + "/out/lzavin" + run_rand_id + ".simulated_bestend.nc"
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stdin=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(p.communicate()[0])
    p.wait()

    ### SECOND LISFLOOD RUN ###

    run_file = lis_template.settings_path('-Run', run_rand_id)
    lisf1.main(run_file)

    # DD JIRA issue https://efascom.smhi.se/jira/browse/ECC-1210 restore the backup
    cmd = "rm " + subcatch.path + "/out/avgdis" + run_rand_id + "end.nc " + subcatch.path + "/out/lzavin" + run_rand_id + "end.nc"
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stdin=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(p.communicate()[0])
    p.wait()

    simulated_best_tss2csv(subcatch.path, run_rand_id, cfg.forcing_start, 'dis', 'streamflow')
    simulated_best_tss2csv(subcatch.path, run_rand_id, cfg.forcing_start, 'chanq', 'chanq')



def generate_benchmark(cfg):

    observed_streamflow = 0.0

    minParams = ParamRanges[str(ParamRanges.head().columns.values[0])].values
    maxParams = ParamRanges[str(ParamRanges.head().columns.values[1])].values
    defaultParams = ParamRanges[str(ParamRanges.head().columns.values[2])].values

    ## DD uncomment to generate a synthetic run with default parameters to converge to
    RunModel((defaultParams - minParams) / (maxParams - minParams), mapLoadOnly=False)
    print("Finished generating default run. Please relaunch the calibration. It will now try to converge to this default run.")

