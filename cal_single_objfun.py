
# -*- coding: utf-8 -*-
"""Please refer to quick_guide.pdf for usage instructions"""
import os
import sys
import HydroStats
import numpy as np
import pandas
import multiprocessing as mp
import time
from pcrasterCommand import pcrasterCommand, getPCrasterPath
from configparser import ConfigParser
from scoop import futures
import subprocess
from datetime import datetime, timedelta
import cProfile
import traceback
import matplotlib.pyplot as plt
import networkx
import math

# lisflood
import lisf1

# deap related packages
import array
import random
from deap import algorithms
from deap import base
from deap import creator
from deap import tools


    # #https://stackoverflow.com/questions/9754034/can-i-create-a-shared-multiarray-or-lists-of-lists-object-in-python-for-multipro?lq=1
    # # DD Use a multiprocessing shared Array type to keep track of other metrics of each run besides the KGE, e.g. for different early-stopping mechanism
    # totSumError = mp.Array('f', (maxGen+1) * max(pop,lambda_))
    # np.frombuffer(totSumError.get_obj(), 'f').reshape((maxGen+1), max(pop,lambda_))[:] = np.nan

def plotSolutionTree(history, pHistory):
    graph = networkx.DiGraph(history.genealogy_tree)
    graph = graph.reverse()  # Make the graph top-down
    colors = []
    for i in graph:
        ind = history.genealogy_history[i]
        Parameters = [None] * len(ParamRanges)
        for ii in range(len(ParamRanges)):
            Parameters[ii] = float(ind[ii] * (float(ParamRanges.iloc[ii, 1]) - float(ParamRanges.iloc[ii, 0])) + float(ParamRanges.iloc[ii, 0]))
        for index, row in pHistory.iterrows():
            params = row.iloc[1:len(ParamRanges) + 1]
            sParameters = pandas.Series(Parameters, dtype=float, index = params.index)
            if all(abs(params-sParameters)<1e-6):
                colors += [row['Kling Gupta Efficiency']]
    # colors = [toolbox.evaluate(history.genealogy_history[i])[0] for i in graph]
    networkx.draw(graph, node_color=colors)
    plt.show()
    plt.savefig(os.path.join(path_subcatch, "graph.svg"), format='svg')

# DD Function to profile the multiprocessing children
def profiler(Individual, mapLoadOnly=None):
    profNum = int(random.random()*1e6)
    print("run profiler " + str(profNum))
    ret = []
    cProfile.runctx('wrapper(ret, Individual, mapLoadOnly=None)', globals(), locals(), 'prof%d.cprof' %profNum)
    return ret[0]

# DD Wrapper function to retrieve result of the profiled function
def wrapper(ret, Individual, mapLoadOnly=None):
    ret.append(RunModel(Individual, mapLoadOnly=None))

def findBestSAERuns(numOfRuns, population, allowDuplicates=True):
    pHistory = pandas.read_csv(os.path.join(path_subcatch, "paramsHistory.csv"), sep=",")[3:]
    saes = []
    for ind in population:
        # Make sure we find at least one element in the table, incl. the fact we can only represent 12 digit numbers in, so try to find a match with decreasing precition until a match is found
        indIndex = pHistory["Kling Gupta Efficiency"].index[abs(pHistory["Kling Gupta Efficiency"] - ind.fitness.values[0]) <= 1e-11]
        if len(indIndex) == 0:
            indIndex = pHistory["Kling Gupta Efficiency"].index[abs(pHistory["Kling Gupta Efficiency"] - ind.fitness.values[0]) <= 1e-10]
        if len(indIndex) == 0:
            indIndex = pHistory["Kling Gupta Efficiency"].index[abs(pHistory["Kling Gupta Efficiency"] - ind.fitness.values[0]) <= 1e-9]
        if len(indIndex) == 0:
            indIndex = pHistory["Kling Gupta Efficiency"].index[abs(pHistory["Kling Gupta Efficiency"] - ind.fitness.values[0]) <= 1e-8]
        if len(indIndex) == 0:
            indIndex = pHistory["Kling Gupta Efficiency"].index[abs(pHistory["Kling Gupta Efficiency"] - ind.fitness.values[0]) <= 1e-7]
        if len(indIndex) == 0:
            indIndex = pHistory["Kling Gupta Efficiency"].index[abs(pHistory["Kling Gupta Efficiency"] - ind.fitness.values[0]) <= 1e-7]
        if len(indIndex) == 0:
            indIndex = pHistory["Kling Gupta Efficiency"].index[abs(pHistory["Kling Gupta Efficiency"] - ind.fitness.values[0]) <= 1e-6]
        if len(indIndex) == 0:
            indIndex = pHistory["Kling Gupta Efficiency"].index[abs(pHistory["Kling Gupta Efficiency"] - ind.fitness.values[0]) <= 1e-5]
        if len(indIndex) == 0:
            indIndex = pHistory["Kling Gupta Efficiency"].index[abs(pHistory["Kling Gupta Efficiency"] - ind.fitness.values[0]) <= 1e-4]
        if len(indIndex) == 0:
            indIndex = pHistory["Kling Gupta Efficiency"].index[abs(pHistory["Kling Gupta Efficiency"] - ind.fitness.values[0]) <= 1e-3]
        if len(indIndex) == 0:
            indIndex = pHistory["Kling Gupta Efficiency"].index[abs(pHistory["Kling Gupta Efficiency"] - ind.fitness.values[0]) <= 1e-2]
        if len(indIndex) == 0:
            indIndex = pHistory["Kling Gupta Efficiency"].index[abs(pHistory["Kling Gupta Efficiency"] - ind.fitness.values[0]) <= 1e-1]                
        if len(indIndex) == 0:
            print("BLAAT")
            print(ind.fitness.values)
            for ii, i in enumerate(pHistory["Kling Gupta Efficiency"]):
                print(ii, i, abs(i - ind.fitness.values[0]), abs(i - ind.fitness.values[0]) <= 1e-11, abs(i - ind.fitness.values[0]) <= 1e-10, abs(i - ind.fitness.values[0]) <= 1e-9, abs(i - ind.fitness.values[0]) <= 1e-8, abs(i - ind.fitness.values[0]) <= 1e-7, abs(i - ind.fitness.values[0]) <= 1e-6, abs(i - ind.fitness.values[0]) <= 1e-5)
            print( "MEEUUHH")
        indSae = pHistory['sae'][indIndex]
        indKGE = pHistory["Kling Gupta Efficiency"][indIndex]
        # Round the SAE as well to a precision of 1e4
        saes.append([round(indSae[indIndex[0]] * 1e4) / 1e4, indKGE[indIndex[0]], ind])
    if allowDuplicates:
        uniques = sorted([ind[0] for ind in saes])
    else:
        uniques = sorted(set([ind[0] for ind in saes]))
    saesNoDups = []
    for u in uniques:
        saesNoDups.append([ind for ind in saes if ind[0] == u])
    # by appending automatically, the values are sorted
    children = []
    for i in range(numOfRuns):
        children.append(saesNoDups[i][0][2])
    return children


########################################################################
#   Function for running the model, returns objective function scores
########################################################################
#@profile

def calibration_start_end(cfg, station_data):
    if cfg.fast_debug:
        # Turn this on for debugging faster. You can speed up further by setting maxGen = 1
        cal_start = datetime.strptime(station_data['Cal_Start'], '%d/%m/%Y %H:%M').strftime('%Y-%m-%d %H:%M')
        cal_end = (datetime.strptime(cal_start, '%Y-%m-%d %H:%M') + timedelta(days=1121)).strftime('%Y-%m-%d %H:%M')
        # !!!! rewrite cfg parameters
        cfg.forcing_start = datetime.strptime(cal_start, '%Y-%m-%d %H:%M')
        cfg.forcing_end = datetime.strptime(cal_end, '%Y-%m-%d %H:%M')
        cfg.WarmupDays = 0
    else:
        # Compute the time steps at which the calibration should start and end
        cal_start = datetime.strptime(station_data['Cal_Start'],'%d/%m/%Y %H:%M').strftime('%Y-%m-%d %H:%M')
        #al_End = str(datetime.datetime.strptime(row['Cal_End'],"%d/%m/%Y %H:%M"))
        cal_end = datetime.strptime(station_data['Cal_End'],'%d/%m/%Y %H:%M').strftime('%Y-%m-%d %H:%M') # DD original

    return cal_start, cal_end


class HydrologicalModelTest(HydrologicalModel):
    
    def __init__(self, cfg, obsid, station_data, lis_template):

        self.observed_streamflow = read_benchmark_streamflow(cfg, obsid)

        super().__init__(self, cfg, station_data, lis_template)

    def read_observed_streamflow(self):
        cfg = self.cfg
        obsid = self.obsid

        # Load observed streamflow # DD Much faster IO with npy despite being more complicated (<1s vs 22s)
        streamflow_data = pandas.read_csv(cfg.subcatchment_path + "/" + str(obsid) + "/convergenceTester.csv", sep=",", index_col=0, header=None)
        # streamflow_data.index = pandas.date_range(start=ObservationsStart, end=ObservationsEnd, periods=len(streamflow_data))
        #streamflow_data.index = pandas.date_range(start=ForcingStart, end=ForcingEnd, periods=len(streamflow_data))
        streamflow_data.index = pandas.date_range(start=streamflow_data.index[0], end=streamflow_data.index[-1], periods=len(streamflow_data))
        observed_streamflow = streamflow_data[cfg.forcing_start:cfg.forcing_end]
        return observed_streamflow

    def get_parameters(self):
        cfg = self.cfg
        parameters = [None] * len(cfg.param_ranges)
        for ii in range(len(cfg.param_ranges)):
          ref = 0.5 * (float(cfg.param_ranges.iloc[ii, 1]) - float(cfg.param_ranges.iloc[ii, 0])) + float(ParamRanges.iloc[ii, 0])
          parameters[ii] = ref * (1+10**-numDigits)
        return parameters

    def get_start_end_local(self):
        cfg = self.cfg
        cal_start_local = (cfg.forcing_end - timedelta(days=335+cfg.WarmupDays)).strftime('%Y-%m-%d %H:%M')
        cal_end_local = cfg.forcing_end.strftime('%Y-%m-%d %H:%M')
        return cal_start_local, cal_end_local

    def resample_streamflows(self, simulated_streamflow, observed_streamflow, cal_start_local, cal_end_local):

        cfg = self.cfg

        Q = pandas.concat({"Sim": simulated_streamflow[1], "Obs": self.observed_streamflow}, axis=1)  # .reset_index()

        # Finally, extract equal-length arrays from it
        Qobs = np.array(Q['Obs'][self.cal_start:self.cal_end]) #.values+0.001
        Qsim = np.array(Q['Sim'][self.cal_start:self.cal_end])

        if self.station_data["CAL_TYPE"].find("_24h") > -1:
            # When testing convergence, we replace the obs by the synthetic obs generated by lisflood with an arbitrary set of params
            Qsim = simulated_streamflow[cal_start_local:cal_end_local]
            Qobs = observed_streamflow[cal_start_local:cal_end_local]
            # apply only to 24h station to aggregate to daily
            Qsim.index = pandas.date_range(cal_start_local, cal_end_local, freq="360min")
            Qsim = Qsim.resample('24H', label="right", closed="right").mean()
            # Same for Qobs
            Qobs.index = pandas.date_range(cal_start_local, cal_end_local, freq="360min")
            Qobs = Qobs.resample('24H', label="right", closed="right").mean()

            Qsim = np.array(Qsim)  # [1].values + 0.001
            Qobs = np.array(Qobs)  # [1].values + 0.001

        # Trim nans
        Qsim = Qsim[~np.isnan(Qobs)]
        Qobs = Qobs[~np.isnan(Qobs)]

        return Qsim, Qobs


class HydrologicalModelBenchmark(HydrologicalModel):
    
    def __init__(self, cfg, obsid, station_data, lis_template):
        super().__init__(self, cfg, obsid, station_data, lis_template)

    def get_parameters(self):
        cfg = self.cfg
        parameters = [None] * len(cfg.param_ranges)
        for ii in range(len(ParamRanges)):
            Parameters[ii] = 0.5 * (float(ParamRanges.iloc[ii, 1]) - float(ParamRanges.iloc[ii, 0])) + float(ParamRanges.iloc[ii, 0])
        return parameters

    def get_start_end_local(self):
        cfg = self.cfg
        cal_start_local = (cfg.forcing_end - timedelta(days=335+cfg.WarmupDays)).strftime('%Y-%m-%d %H:%M')
        cal_end_local = cfg.forcing_end.strftime('%Y-%m-%d %H:%M')
        return cal_start_local, cal_end_local

    def run(self, Individual, mapLoadOnly=None):

        cfg = self.cfg

        run_rand_id = str(int(random.random()*1e10)).zfill(12)

        cal_start_local, cal_end_local = self.get_start_end_local(mapLoadOnly)

        parameters = self.get_parameters()

        self.lis_template.write_template(self.obsid, run_rand_id, cal_start_local, cal_end_local, cfg.param_ranges, parameters):

        # DD Do not run lisflood twice in a bash script, instead, import lisflood and keep everything in memory to reduce IO
        currentdir = os.getcwd()

        prerun_file = self.lis_template('-Prerun')
        run_file = self.lis_template('-Run')

        lisf1.main(prerun_file) #os.path.realpath(__file__),
        lisf1.main(run_file, '-v', '-t')  # os.path.realpath(__file__),

        Qsim_tss = os.path.join(path_subcatch, "out", 'dis' + run_rand_id + '.tss')
        simulated_streamflow = pandas.read_csv(Qsim_tss, sep=r"\s+", index_col=0, skiprows=4, header=None, skipinitialspace=True)
        simulated_streamflow[1][simulated_streamflow[1] == 1e31] = np.nan
        Qsim = simulated_streamflow[1].values
        print( ">> Saving simulated streamflow with default parameters(convergenceTester.csv)")
        # DD DEBUG try shorter time series for testing convergence
        Qsim = pandas.DataFrame(data=Qsim, index=pandas.date_range(Cal_Start_Local, periods=len(Qsim), freq='6H'))
        #Qsim = pandas.DataFrame(data=Qsim, index=pandas.date_range(ForcingStart, periods=len(Qsim), freq='6H'))
        Qsim.to_csv(os.path.join(path_subcatch, "convergenceTester.csv"), ',', header="")
        
        return Qsim


class HydrologicalModel():

    def __init__(self, cfg, obsid, station_data, lis_template, lock_mgr):

        self.cfg = cfg
        self.obsid = obsid
        self.station_data = station_data
        self.lis_template = lis_template

        cal_start, cal_end = calibration_start_end(cfg, station_data)
        self.cal_start = cal_start
        self.cal_end = cal_end

        self.observed_streamflow = read_observed_streamflow(cfg, obsid)

        self.lock_mgr = lock_mgr

    def read_observed_streamflow(self):
        cfg = self.cfg
        obsid = self.obsid
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

    def get_start_end_local(self, mapLoadOnly):
        cfg = self.cfg
        cal_start_local = self.cal_start
        if mapLoadOnly:
            cal_end_local = datetime.strptime(self.cal_start, "%Y-%m-%d %H:%M") + timedelta(days=0.25) #datetime.strptime(row['Cal_Start'], '%d/%m/%Y %H:%M') + timedelta(days=0.25)
            cal_end_local = cal_end_local.strftime('%Y-%m-%d %H:%M')
        else:
            cal_end_local = self.cal_end

        return cal_start_local, cal_end_local

    def get_parameters(self, Individual):
        param_ranges = self.cfg.param_ranges
        Parameters = [None] * len(param_ranges)
        for ii in range(len(param_ranges)):
            Parameters[ii] = Individual[ii]*(float(param_ranges.iloc[ii,1])-float(param_ranges.iloc[ii,0]))+float(param_ranges.iloc[ii,0])

        return parameters

    def read_simulated_streamflow(self, run_rand_id, cal_start_local):
        Qsim_tss = os.path.join(self.cfg.path_subcatch, "out", 'dis'+run_rand_id+'.tss')
        if os.path.isfile(Qsim_tss)==False:
            print("run_rand_id: "+str(run_rand_id))
            raise Exception("No simulated streamflow found. Probably LISFLOOD failed to start? Check the log files of the run!")
        simulated_streamflow = pandas.read_csv(Qsim_tss, sep=r"\s+", index_col=0, skiprows=4, header=None, skipinitialspace=True)
        simulated_streamflow[1][simulated_streamflow[1]==1e31] = np.nan
        simulated_streamflow.index = [datetime.strptime(cal_start_local, "%Y-%m-%d %H:%M") + timedelta(hours=6*i) for i in range(len(simulated_streamflow.index))]

        return simulated_streamflow

    def resample_streamflows(self, simulated_streamflow, observed_streamflow, cal_start_local, cal_end_local):

        cfg = self.cfg

        Q = pandas.concat({"Sim": simulated_streamflow[1], "Obs": self.observed_streamflow}, axis=1)  # .reset_index()

        # Finally, extract equal-length arrays from it
        Qobs = np.array(Q['Obs'][self.cal_start:self.cal_end]) #.values+0.001
        Qsim = np.array(Q['Sim'][self.cal_start:self.cal_end])

        if cfg.calibration_freq == r"6-hourly":
            # DD: Check if daily or 6-hourly observed streamflow is available
            # DD: Aggregate 6-hourly simulated streamflow to daily ones
            if self.station_data["CAL_TYPE"].find("_24h") > -1:
                # DD: Overwrite index with date range so we can use Pandas' resampling + mean function to easily average 6-hourly to daily data
                Qsim = simulated_streamflow[self.cal_start:self.cal_end]
                Qsim.index = pandas.date_range(self.cal_start, self.cal_end, freq="360min")
                Qsim = Qsim.resample('24H', label="right", closed="right").mean()
                Qsim = np.array(Qsim) #[1].values + 0.001
                # Same for Qobs
                Qobs = observed_streamflow[self.cal_start:self.cal_end]
                Qobs.index = pandas.date_range(self.cal_start, self.cal_end, freq="360min")
                Qobs = Qobs.resample('24H', label="right", closed="right").mean()
                Qobs = np.array(Qobs) #[1].values + 0.001
                # Trim nans
                Qsim = Qsim[~np.isnan(Qobs)]
                Qobs = Qobs[~np.isnan(Qobs)]
        elif cfg.calibration_freq == r"daily":
            # DD Untested code! DEBUG TODO
            Qobs = observed_streamflow[self.cal_start:self.cal_end]
            Qobs.index = pandas.date_range(self.cal_start, self.cal_end, freq="360min")
            Qobs = Qobs.resample('24H', label="right", closed="right").mean()
            Qobs = np.array(Qobs) #[1].values + 0.001
            # Trim nans
            Qobs = Qobs[~np.isnan(Qobs)]

        return Qsim, Qobs

    def compute_KGE(self, Qsim, Qobs):
        cfg = self.cfg

        # Compute objective function score
        # # DD A few attempts with filtering of peaks and low flows
        if cfg.calibration_freq == r"6-hourly":
            # DD: Check if daily or 6-hourly observed streamflow is available
            # DD: Aggregate 6-hourly simulated streamflow to daily ones
            if self.station_data["CAL_TYPE"].find("_24h") > -1:
                fKGEComponents = HydroStats.fKGE(s=Qsim, o=Qobs, warmup=cfg.WarmupDays, weightedLogWeight=0.0, lowFlowPercentileThreshold=0.0, usePeaksOnly=False)
            else:
                fKGEComponents = HydroStats.fKGE(s=Qsim, o=Qobs, warmup=4*cfg.WarmupDays, weightedLogWeight=0.0, lowFlowPercentileThreshold=0.0, usePeaksOnly=False)
        elif cfg.calibration_freq == r"daily":
            fKGEComponents = HydroStats.fKGE(s=Qsim, o=Qobs, warmup=cfg.WarmupDays, weightedLogWeight=0.0, lowFlowPercentileThreshold=0.0, usePeaksOnly=False)

        return fKGEComponents

    def update_parameter_history(self, run_rand_id, parameters, fKGEComponents):

        cfg = self.cfg

        self.lock_mgr.lock()
        with open(os.path.join(cfg.path_subcatch, "runs_log.csv"), "a") as myfile:
            myfile.write(str(run_rand_id)+","+str(KGE)+"\n")

        # DD We want to check that the parameter space is properly sampled. Write them out to file now
        paramsHistoryFilename = os.path.join(cfg.path_subcatch, "paramsHistory.csv")
        if not os.path.exists(paramsHistoryFilename) or os.path.getsize(paramsHistoryFilename) == 0:
            paramsHistoryFile = open(paramsHistoryFilename, "w")
            # Headers
            paramsHistory = "randId,"
            for i in [str(ip) + "," for ip in cfg.param_ranges.index.values]:
                paramsHistory += i
            for i in [str(ip) + "," for ip in ["Kling Gupta Efficiency", "Correlation", "Signal ratio (s/o) (Bias)", "Noise ratio (s/o) (Spread)", "sae", "generation", "runNumber"]]:
                paramsHistory += i
            paramsHistory += "\n"
            # Minimal values
            paramsHistory += str(cfg.param_ranges.head().columns.values[0]) + ","
            for i in [str(ip) + "," for ip in cfg.param_ranges[str(cfg.param_ranges.head().columns.values[0])].values]:
                paramsHistory += i
            paramsHistory += "\n"
            # Default values
            paramsHistory += str(cfg.param_ranges.head().columns.values[2]) + ","
            for i in [str(ip) + "," for ip in cfg.param_ranges[str(cfg.param_ranges.head().columns.values[2])].values]:
                paramsHistory += i
            paramsHistory += "\n"
            # Maximal values
            paramsHistory += str(cfg.param_ranges.head().columns.values[1]) + ","
            for i in [str(ip) + "," for ip in cfg.param_ranges[str(cfg.param_ranges.head().columns.values[1])].values]:
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
        if cfg.use_multiprocessing:
            # paramsHistory += str(HydroStats.sae(s=Qsim, o=Qobs, warmup=WarmupDays)) + ","
            paramsHistory += str(self.lock_mgr.get_gen()) + ","
            paramsHistory += str(self.lock_mgr.get_run())
        paramsHistory += "\n"
        paramsHistoryFile.write(paramsHistory)
        paramsHistoryFile.close()
        self.lock_mgr.unlock()

    def init_run(self):

        # dummy Individual, doesn't matter here
        param_ranges = self.cfg.param_ranges
        Individual = 0.5*np.ones(len(param_ranges))

        cfg = self.cfg

        run_rand_id = str(int(random.random()*1e10)).zfill(12)

        cal_start_local, cal_end_local = self.get_start_end_local(mapLoadOnly)

        parameters = self.get_parameters(Individual)

        self.lis_template.write_template(self.obsid, run_rand_id, cal_start_local, cal_end_local, cfg.param_ranges, parameters):

        prerun_file = self.lis_template.settings_path('-Prerun')

        try:
            lisf1.main(prerun_file, '-i')
        except:
            traceback.print_exc()
            raise Exception("")

    def run(self, Individual):

        cfg = self.cfg

        run_rand_id = str(int(random.random()*1e10)).zfill(12)

        cal_start_local, cal_end_local = self.get_start_end_local(mapLoadOnly)

        parameters = self.get_parameters(Individual)

        self.lis_template.write_template(self.obsid, run_rand_id, cal_start_local, cal_end_local, cfg.param_ranges, parameters):

        prerun_file = self.lis_template.settings_path('-Prerun')
        run_file = self.lis_template.settings_path('-Run')

        try:
            lisf1.main(prerun_file) #os.path.realpath(__file__),
            lisf1.main(run_file)  # os.path.realpath(__file__),
        except:
            traceback.print_exc()
            raise Exception("")

        # DD Extract simulation
        simulated_streamflow = read_simulated_streamflow(run_rand_id, cal_start_local)

        Qsim, Qobs = self.resample_streamflows(simulated_streamflow, self.observed_streamflow)
        if len(Qobs) != len(Qsim):
            raise Exception("run_rand_id: "+str(run_rand_id)+": observed and simulated streamflow arrays have different number of elements ("+str(len(Qobs))+" and "+str(len(Qsim))+" elements, respectively)")

        fKGEComponents = compute_KGE(self, Qsim, Qobs)

        KGE = fKGEComponents[0]

        self.lock_mgr.increment_run()

        self.update_parameter_history(run_rand_id, parameters, fKGEComponents)

        print("   run_rand_id: "+str(run_rand_id)+", KGE: "+"{0:.3f}".format(KGE))

        return KGE, # If using just one objective function, put a comma at the end!!!


class LockManager():
    def __init__(self):
        
        self.counters = {}
        self.counters['gen'] = mp.Value('i')
        self.set_gen(-1)
        self.counters['run'] = mp.Value('i')
        self.set_run(-1)
        
        self.lock = mp.Lock()

    def lock(self):
        self.lock.acquire()

    def unlock(self):
        self.lock.release()

    def increment_gen(self):
        self._increment('gen')
        self._set('run', 0)

    def increment_run(self):
        self._increment('run')

    def set_gen(self, value):
        self._set('gen', value)
        self._set('run', 0)

    def set_run(self, value):
        self._set('run', value)

    def get_gen(self, value):
        self._value('gen')

    def get_run(self, value):
        self._value('run')

    def _increment(self, name):
        with self.counters[name].get_lock():
            self.counters[name].value += 1

    def _set(self, name, value):
        with self.counters[name].get_lock():
            self.counters[name].value = value

    def _value(self, name):
        return counters[name]


def initialise_deap(deap_param, model):

    creator.create("FitnessMin", base.Fitness, weights=(1.0,))
    creator.create("Individual", array.array, typecode='d', fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    # history = tools.History()

    # Attribute generator
    toolbox.register("attr_float", random.uniform, 0, 1)

    # Structure initializers
    toolbox.register("Individual", tools.initRepeat, creator.Individual, toolbox.attr_float, len(ParamRanges))
    toolbox.register("population", tools.initRepeat, list, toolbox.Individual)

    def checkBounds(min, max):
        def decorator(func):
            def wrappper(*args, **kargs):
                offspring = func(*args, **kargs)
                for child in offspring:
                    for i in range(len(child)):
                        if child[i] > max:
                            child[i] = max
                        elif child[i] < min:
                            child[i] = min
                return offspring
            return wrappper
        return decorator

    toolbox.register("evaluate", model.run) #profiler) #RunModel) #DD Debug for profiling
    toolbox.register("mate", tools.cxBlend, alpha=0.15)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.3, indpb=0.3)
    toolbox.register("select", tools.selNSGA2)
    toolbox.decorate("mate", checkBounds(0, 1))
    toolbox.decorate("mutate", checkBounds(0, 1))
    # toolbox.decorate("mate", history.decorator)
    # toolbox.decorate("mutate", history.decorator)

    return toolbox


def check_termination_conditions(gen, deap_parameters, effmax, conditions):

    min_gen = deap_parameters.minGen
    max_gen = deap_parameters.maxGen

    # Terminate the optimization after maxGen generations
    if gen >= max_gen:
        print(">> Termination criterion maxGen fulfilled.")
        conditions['maxGen'] = True

    if gen >= min_gen and (effmax[gen, 0] - effmax[gen - 3, 0]) < 0.003
        # # DD attempt to stop early with different criterion
        # if (effmax[gen.value,0]-effmax[gen.value-1,0]) < 0.001 and np.nanmin(np.frombuffer(totSumError.get_obj(), 'f').reshape((maxGen+1), max(pop,lambda_))[gen.value, :]) > np.nanmin(np.frombuffer(totSumError.get_obj(), 'f').reshape((maxGen+1), max(pop,lambda_))[gen.value - 1, :]):
        #     print(">> Termination criterion no-improvement sae fulfilled.")
        #     # conditions["StallFit"] = True
        print(">> Termination criterion no-improvement KGE fulfilled.")
        conditions["StallFit"] = True


def restore_calibration(deap_param, lock_mgr, toolbox, halloffame, history_file,
                        effmax, effmin, effavg, effstd, conditions):

    # Open the paramsHistory file from previous runs
    paramsHistory = pandas.read_csv(history_file, sep=",")[4:]
    print("Restoring previous calibration state")

    # Initiate the generations counter
    lock_mgr.set_gen(0)

    population = None
    # reconstruct the generational evoluation
    for igen in range(int(paramsHistory["generation"].iloc[-1])+1):
        # retrieve the generation's data
        parsHistory = paramsHistory[paramsHistory["generation"] == igen]
        # reconstruct the invalid individuals array
        valid_ind = updatePopulationFromHistory(parsHistory)
        # Update the hall of fame with the generation's parameters
        halloffame.update(valid_ind)
        # prepare for the next stage
        if population is not None:
            population[:] = toolbox.select(population + valid_ind, deap_param.mu)
        else:
            population = valid_ind

        gen = lock_mgr.get_gen()

        # Loop through the different objective functions and calculate some statistics from the Pareto optimal population
        for ii in range(1):
            effmax[gen, ii] = np.amax([halloffame[x].fitness.values[ii] for x in range(len(halloffame))])
            effmin[gen, ii] = np.amin([halloffame[x].fitness.values[ii] for x in range(len(halloffame))])
            effavg[gen, ii] = np.average([halloffame[x].fitness.values[ii] for x in range(len(halloffame))])
            effstd[gen, ii] = np.std([halloffame[x].fitness.values[ii] for x in range(len(halloffame))])
        print(">> gen: " + str(gen) + ", effmax_KGE: " + "{0:.3f}".format(effmax[gen, 0]))

        check_termination_conditions(gen, deap_param, effmax, conditions)

        lock_mgr.increment_gen()

    return population


def generate_population(deap_param, lock_mgr, toolbox, halloffame, effmax, effmin, effavg, effstd, conditions):
    # Start with a fresh population
    population = toolbox.population(n=deap_param.pop)

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid] # DD this filters the population or children for uncalculated fitnesses. We retain only the uncalculated ones to avoid recalculating those that already had a fitness. Potentially this can save time, especially if the algorithm tends to produce a child we already ran.
    
    lock_mgr.set_gen(0)

    # Run the first generation, first random sampling of the parameter space
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses): # DD this updates the fitness (=KGE) for the individuals in the global pool of individuals which we just calculated. ind are
        ind.fitness.values = fit

    halloffame.update(population) # DD this selects the best one

    gen = lock_mgr.get_gen()
    assert gen == 0

    # Loop through the different objective functions and calculate some statistics from the Pareto optimal population
    for ii in range(1):
        effmax[0,ii] = np.amax([halloffame[x].fitness.values[ii] for x in range(len(halloffame))])
        effmin[0,ii] = np.amin([halloffame[x].fitness.values[ii] for x in range(len(halloffame))])
        effavg[0,ii] = np.average([halloffame[x].fitness.values[ii] for x in range(len(halloffame))])
        effstd[0,ii] = np.std([halloffame[x].fitness.values[ii] for x in range(len(halloffame))])
    print(">> gen: "+str(gen)+", effmax_KGE: "+"{0:.3f}".format(effmax[gen,0]))

    # Update the generation to the first
    lock_mgr.increment_gen()

    return population


def generate_offspring(deap_param, lock_mgr, toolbox, halloffame, population, effmax, effmin, effavg, effstd, conditions):

    # Vary the population
    offspring = algorithms.varOr(population, toolbox, deap_param.lambda_, deap_param.cxpb, deap_param.mutpb)

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind) # DD this runs lisflood
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    # Update the hall of fame with the generated individuals
    if halloffame is not None:
        halloffame.update(offspring)

    # Select the next generation population
    population[:] = toolbox.select(population + offspring, deap_param.mu)

    gen = lock_mgr.get_gen()

    # Loop through the different objective functions and calculate some statistics
    # from the Pareto optimal population
    for ii in range(1):
        effmax[gen,ii] = np.amax([halloffame[x].fitness.values[ii] for x in range(len(halloffame))])
        effmin[gen,ii] = np.amin([halloffame[x].fitness.values[ii] for x in range(len(halloffame))])
        effavg[gen,ii] = np.average([halloffame[x].fitness.values[ii] for x in range(len(halloffame))])
        effstd[gen,ii] = np.std([halloffame[x].fitness.values[ii] for x in range(len(halloffame))])
    print(">> gen: "+str(gen)+", effmax_KGE: "+"{0:.3f}".format(effmax[gen,0]))

    check_termination_conditions(gen, deap_param, effmax, conditions)

    lock_mgr.increment_gen()


def write_front_history(cfg, gen, effmax, effmin, effavg, effstd):
    front_history = pandas.DataFrame()
    front_history['gen'] = range(gen)
    front_history['effmax_R'] = effmax[0:gen,0]
    front_history['effmin_R'] = effmin[0:gen,0]
    front_history['effstd_R'] = effstd[0:gen,0]
    front_history['effavg_R'] = effavg[0:gen,0]
    front_history.to_csv(os.path.join(cfg.path_subcatch,"front_history.csv"))


def write_ranked_solution(cfg):
    pHistory = pandas.read_csv(os.path.join(cfg.path_subcatch, "paramsHistory.csv"), sep=",")[3:]
    # Keep only the best 10% of the runs for the selection of the parameters for the next generation
    pHistory = pHistory.sort_values(by="Kling Gupta Efficiency", ascending=False)
    pHistory = pHistory.head(int(max(2,round(len(pHistory) * 0.1))))
    n = len(pHistory)
    minOffset = 0.1
    maxOffset = 1.0
    # Give ranking scores to corr
    pHistory = pHistory.sort_values(by="Correlation", ascending=False)
    pHistory["corrRank"] = [minOffset + float(i + 1) * (maxOffset - minOffset) / n for i, ii in enumerate(pHistory["Correlation"].values)]
    # Give ranking scores to sae
    pHistory = pHistory.sort_values(by="sae", ascending=True)
    pHistory["saeRank"] = [minOffset + float(i + 1) * (maxOffset - minOffset) / n for i, ii in enumerate(pHistory["sae"].values)]
    # Give ranking scores to KGE
    pHistory = pHistory.sort_values(by="Kling Gupta Efficiency", ascending=False)
    pHistory["KGERank"] = [minOffset + float(i + 1) * (maxOffset - minOffset) / n for i, ii in enumerate(pHistory["Kling Gupta Efficiency"].values)]
    # Give pareto score
    pHistory["paretoRank"] = pHistory["corrRank"].values * pHistory["saeRank"].values * pHistory["KGERank"].values
    pHistory = pHistory.sort_values(by="paretoRank", ascending=True)
    pHistory.to_csv(os.path.join(cfg.path_subcatch,"pHistoryWRanks.csv"),',')

    return pHistory


def write_pareto_front(cfg, pHistory):
    # Select the best pareto candidate
    bestParetoIndex = pHistory["paretoRank"].nsmallest(1).index
    # Save the pareto front
    paramvals = np.zeros(shape=(1,len(cfg.param_ranges)))
    paramvals[:] = np.NaN
    for ipar, par in enumerate(cfg.param_ranges.index):
        paramvals[0][ipar] = pHistory.loc[bestParetoIndex][par]
    pareto_front = pandas.DataFrame(
        {
            'effover': pHistory["Kling Gupta Efficiency"].loc[bestParetoIndex],
            'R': pHistory["Kling Gupta Efficiency"].    loc[bestParetoIndex]
        }, index=[0]
    )
    for ii in range(len(cfg.param_ranges)):
        pareto_front["param_"+str(ii).zfill(2)+"_"+cfg.param_ranges.index[ii]] = paramvals[0,ii]
    pareto_front.to_csv(os.path.join(cfg.path_subcatch,"pareto_front.csv"),',')


def run_calibration(cfg, obsid, station_data, model, lock_mgr):
       
    toolbox = initialise_deap(cfg.deap_param, model)

    observed_streamflow = model.read_observed_streamflow()

    t = time.time()

    # load forcings and input maps in cache
    # required in front of processing pool
    # otherwise each child will reload the maps
    model.init_run()

    if use_multiprocessing==True:
        pool_size = cfg.deap_param.numCPUs #mp.cpu_count() * 1 ## DD just restrict the number of CPUs to use manually
        pool = mp.Pool(processes=pool_size, initargs=(self.lock_mgr.lock,))
        toolbox.register("map", pool.map)

    #cxpb = 0.9 # For someone reason, if sum of cxpb and mutpb is not one, a lot less Pareto optimal solutions are produced
    # DD: These values are used as percentage probability, so they should add up to 1, to determine whether to mutate or cross. The former finds the parameter for the next generation by taking the average of two parameters. This could lead to convergence to a probability set by the first generation as a biproduct of low first-generation parameter spread (they are generated using a uniform-distribution random generator.
    #mutpb = 0.1

    # Initialise statistics arrays
    effmax = np.zeros(shape=(deap_parameters.maxGen + 1, 1)) * np.NaN
    effmin = np.zeros(shape=(deap_parameters.maxGen + 1, 1)) * np.NaN
    effavg = np.zeros(shape=(deap_parameters.maxGen + 1, 1)) * np.NaN
    effstd = np.zeros(shape=(deap_parameters.maxGen + 1, 1)) * np.NaN

    # Start generational process setting all stopping conditions to false
    conditions = {"maxGen": False, "StallFit": False}

    # Start a new hall of fame
    halloffame = tools.ParetoFront()

    # Attempt to open a previous parameter history
    history_file = os.path.join(cfg.path_subcatch, "paramsHistory.csv")
    if os.path.getsize(history_file) > 0 and os.path.isfile(history_file):
        population = restore_calibration(deap_parameters, lock_mgr, toolbox, halloffame, history_file, effmax, effmin, effavg, effstd, conditions)
    else:
        # No previous parameter history was found, so start from scratch
        population = generate_population(deap_parameters, lock_mgr, toolbox, halloffame, effmax, effmin, effavg, effstd, conditions)

    # Resume the generational process from wherever we left off
    while not any(conditions.values()):
        generate_offspring(deap_param, lock_mgr, toolbox, halloffame, population, effmax, effmin, effavg, effstd, conditions)

    # plotSolutionTree(history, paramsHistory)

    # Finito
    if use_multiprocessing == True:
        pool.close()
    elapsed = time.time() - t
    print(">> Time elapsed: "+"{0:.2f}".format(elapsed)+" s")

    ########################################################################
    #   Save calibration results
    ########################################################################

    # Save history of the change in objective function scores during calibration to csv file
    write_front_history(cfg, gen, effmax, effmin, effavg, effstd)

    pHistory = write_ranked_solutions(cfg)

    write_pareto_front(cfg, pHistory)

    return


def generate_benchmark(cfg):

    observed_streamflow = 0.0

    minParams = ParamRanges[str(ParamRanges.head().columns.values[0])].values
    maxParams = ParamRanges[str(ParamRanges.head().columns.values[1])].values
    defaultParams = ParamRanges[str(ParamRanges.head().columns.values[2])].values

    ## DD uncomment to generate a synthetic run with default parameters to converge to
    RunModel((defaultParams - minParams) / (maxParams - minParams), mapLoadOnly=False)
    print("Finished generating default run. Please relaunch the calibration. It will now try to converge to this default run.")


    def updatePopulationFromHistory(pHistory):
        n = len(pHistory)
        paramvals = np.zeros(shape=(n, len(ParamRanges)))
        paramvals[:] = np.NaN
        invalid_ind = []
        fitnesses = []
        for ind in range(n):
            for ipar, par in enumerate(ParamRanges.index):
                # # scaled to unscaled conversion
                # paramvals[ind][ipar] = pHistory.iloc[ind][par] * (float(ParamRanges.iloc[ipar,1]) - \
                #   float(ParamRanges.iloc[ipar,0]))+float(ParamRanges.iloc[ipar,0])
                # unscaled to scaled conversion
                paramvals[ind][ipar] = (pHistory.iloc[ind][par] - float(ParamRanges.iloc[ipar, 0])) / \
                  (float(ParamRanges.iloc[ipar, 1]) - float(ParamRanges.iloc[ipar, 0]))
            # Create a fresh individual with the restored parameters
            # newInd = toolbox.Individual() # creates an individual with random numbers for the parameters
            newInd = creator.Individual(list(paramvals[ind]))  # creates a totally empty individual
            invalid_ind.append(newInd)
            # WARNING: Change the following line when using multi-objective functions
            # also load the old KGE the individual had (works only for single objective function)
            fitnesses.append((pHistory.iloc[ind][len(ParamRanges) + 1],))
        # update the score of each
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        return invalid_ind


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


def read_tss(tss_file):
    df = pandas.read_csv(tss_file, sep=r"\s+", index_col=0, skiprows=4, header=None, skipinitialspace=True)
    return df


def simulated_best_tss2csv(path_subcatch, forcing_start, dataname, outname):

    tss_file = os.path.join(path_subcatch, "out", dataname + run_rand_id + '.tss')

    tss = read_tss(tss_file)

    tss[1][tss[1]==1e31] = np.nan
    tss_values = tss[1].values

    df = pandas.DataFrame(data=tss_values, index=pandas.date_range(forcing_start, periods=len(tss_values), freq='6H'))
    df.to_csv(os.path.join(path_subcatch, outname+"_simulated_best.csv"), ', ', header="")

    try:
        os.remove(os.path.join(path_subcatch, outname+"_simulated_best.tss"))
    except:
        pass
    os.rename(tss_file, os.path.join(path_subcatch, outname+"_simulated_best.tss"))


def generate_outlet_streamflow(cfg, obsid, station_data, lis_template):

    # Select the "best" parameter set and run LISFLOOD for the entire forcing period
    #Parameters = paramvals[best,:]

    stage_inflows(path_subcatch)

    print(">> Running LISFLOOD using the \"best\" parameter set")
    parameters = read_parameters(path_subcatch)

    run_rand_id = str(int(random.random()*1e10)).zfill(12)

    cal_start_local = cfg.forcing_start.strftime('%Y-%m-%d %H:%M')
    cal_end_local = cfg.forcing_end.strftime('%Y-%m-%d %H:%M')
    lis_template.write_template(obsid, run_rand_id, cal_start_local, cal_end_local, cfg.param_ranges, parameters):

    ### SECOND LISFLOOD RUN ###

    prerun_file = self.lis_template.settings_path('-Prerun')
    lisf1.main(prerun_file)

    # DD JIRA issue https://efascom.smhi.se/jira/browse/ECC-1210 to avoid overwriting the bestrun avgdis.end.nc
    cmd = "cp " + path_subcatch + "/out/avgdis" + run_rand_id + "end.nc " + path_subcatch + "/out/avgdis" + run_rand_id + ".simulated_bestend.nc"
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stdin=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(p.communicate()[0])
    p.wait()
    cmd = "cp " + path_subcatch + "/out/lzavin" + run_rand_id + "end.nc " + path_subcatch + "/out/lzavin" + run_rand_id + ".simulated_bestend.nc"
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stdin=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(p.communicate()[0])
    p.wait()

    ### SECOND LISFLOOD RUN ###

    run_file = self.lis_template.settings_path('-Run')
    lisf1.main(run_file)

    # DD JIRA issue https://efascom.smhi.se/jira/browse/ECC-1210 restore the backup
    cmd = "rm " + path_subcatch + "/out/avgdis" + run_rand_id + "end.nc " + path_subcatch + "/out/lzavin" + run_rand_id + "end.nc"
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stdin=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(p.communicate()[0])
    p.wait()

    simulated_best_tss2csv(path_subcatch, forcing_start, 'dis', 'streamflow')
    simulated_best_tss2csv(path_subcatch, forcing_start, 'chanq', 'chanq')
