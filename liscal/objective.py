import os
import numpy as np
import pandas
from datetime import datetime, timedelta

from liscal import hydro_stats, utils


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
        simulated_streamflow = utils.read_tss(Qsim_tss)
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
