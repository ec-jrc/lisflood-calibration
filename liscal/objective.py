import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from liscal import hydro_stats, utils


class ObjectiveKGE():

    def __init__(self, cfg, subcatch, read_observations=True):
        self.cfg = cfg
        self.subcatch = subcatch
        self.param_ranges = cfg.param_ranges
        self.weights = [1, 0, 0, 0, 0]

        if read_observations:
            self.observed_streamflow = self.read_observed_streamflow()


    def get_parameters(self, Individual):
        param_ranges = self.param_ranges
        parameters = [None] * len(param_ranges)
        for ii in range(len(param_ranges)):
            parameters[ii] = Individual[ii]*(float(param_ranges.iloc[ii,1])-float(param_ranges.iloc[ii,0]))+float(param_ranges.iloc[ii,0])

        return parameters

    def read_observed_streamflow(self):
        cfg = self.cfg
        start = self.subcatch.cal_start
        end = self.subcatch.cal_end
        
        streamflow_data = pd.read_csv(cfg.observed_discharges, sep=",", index_col=0)
        # check that date format is correct
        pd.to_datetime(streamflow_data.index, format='%d/%m/%Y %H:%M', errors='raise')
        
        observed_streamflow = streamflow_data[str(self.subcatch.obsid)]
        observed_streamflow = observed_streamflow[start:end]
        assert datetime.strptime(observed_streamflow.index[0], "%d/%m/%Y %H:%M") >= cfg.forcing_start
        assert datetime.strptime(observed_streamflow.index[-1], "%d/%m/%Y %H:%M") <= cfg.forcing_end

        return observed_streamflow


    def read_simulated_streamflow(self, run_id):
        start = self.subcatch.cal_start
        end = self.subcatch.cal_end

        Qsim_tss = os.path.join(self.subcatch.path_out, 'dis'+run_id+'.tss')
        if os.path.isfile(Qsim_tss)==False:
            print("run_id: "+str(run_id))
            raise Exception("No simulated streamflow found. Probably LISFLOOD failed to start? Check the log files of the run!")

        simulated_streamflow = utils.read_tss(Qsim_tss)[1]  # need to take [1] or we get 2d array
        simulated_streamflow[simulated_streamflow==1e31] = np.nan  # PCRaster will put 1e31 instead of NaN, set to NaN to catch errors
        simulated_streamflow.index = [(datetime.strptime(start, "%d/%m/%Y %H:%M") + timedelta(hours=6*i)).strftime('%d/%m/%Y %H:%M') for i in range(len(simulated_streamflow.index))]
        assert simulated_streamflow.index[-1] == end

        return simulated_streamflow

    def resample_streamflows(self, simulated_streamflow, observed_streamflow):
        cfg = self.cfg
        start = self.subcatch.cal_start
        end = self.subcatch.cal_end

        # check that dates are compatible
        if not simulated_streamflow.index.equals(observed_streamflow.index):
            raise Exception('Simulated and observed streamflow dates not aligned!')

        # Finally, extract equal-length arrays from it
        Qobs = np.array(observed_streamflow[start:end])
        Qsim = np.array(simulated_streamflow[start:end])

        if cfg.calibration_freq == r"6-hourly":
            # DD: Check if daily or 6-hourly observed streamflow is available
            # DD: Aggregate 6-hourly simulated streamflow to daily ones
            if self.subcatch.data["CAL_TYPE"].find("_24h") > -1:
                # start and end have to be in datetime format to avoid "dayfirst" type bugs
                start_pd = datetime.strptime(start, "%d/%m/%Y %H:%M")
                end_pd = datetime.strptime(end, "%d/%m/%Y %H:%M")
                date_range = pd.date_range(start_pd, end_pd, freq="360min")
                # DD: Overwrite index with date range so we can use Pandas' resampling + mean function to easily average 6-hourly to daily data
                Qsim = simulated_streamflow[start:end]
                Qsim.index = date_range
                Qsim = Qsim.resample('24H', label="right", closed="right").mean()
                Qsim = np.array(Qsim)
                # Same for Qobs
                Qobs = observed_streamflow[start:end]
                Qobs.index = date_range
                Qobs = Qobs.resample('24H', label="right", closed="right").mean()
                Qobs = np.array(Qobs)

        elif cfg.calibration_freq == r"daily":
            # DD Untested code! DEBUG TODO
            start = datetime.strptime(start, "%d/%m/%Y %H:%M")
            end = datetime.strptime(end, "%d/%m/%Y %H:%M")
            date_range = pd.date_range(start, end, freq="360min")
            Qobs = observed_streamflow[start:end]
            Qobs.index = date_range
            Qobs = Qobs.resample('24H', label="right", closed="right").mean()
            Qobs = np.array(Qobs) #[1].values + 0.001

        # Trim nans
        # Qsim = Qsim[~np.isnan(Qobs)]
        # Qobs = Qobs[~np.isnan(Qobs)]

        # we shouldn't have NaNs in the sim array at this point
        if np.isnan(Qsim).any():
            raise Exception('NaN found in Qsim')
        # if np.isnan(Qobs).any():
        #     raise Exception('NaN found in Qobs')

        return Qsim, Qobs

    def update_parameter_history(self, run_id, parameters, fKGEComponents, gen, run):

        cfg = self.cfg

        KGE = fKGEComponents[0]

        print('Generation {}, run {} done. KGE: {:.3f}'.format(gen, run, KGE))

        with open(os.path.join(self.subcatch.path, "runs_log.csv"), "a") as myfile:
            myfile.write(str(run_id)+","+str(KGE)+"\n")

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
        paramsHistory += str(run_id) + ","
        for i in [str(ip) + "," for ip in parameters]:
            paramsHistory += i
        for i in [str(ip) + "," for ip in fKGEComponents]:
            paramsHistory += i
        paramsHistory += str(gen) + ","
        paramsHistory += str(run)
        paramsHistory += "\n"
        paramsHistoryFile.write(paramsHistory)
        paramsHistoryFile.close()

    def compute_objectives(self, run_id):

        # DD Extract simulation
        simulated_streamflow = self.read_simulated_streamflow(run_id)

        Qsim, Qobs = self.resample_streamflows(simulated_streamflow, self.observed_streamflow)
        if len(Qobs) != len(Qsim):
            raise Exception("run_id: "+str(run_id)+": observed and simulated streamflow arrays have different number of elements ("+str(len(Qobs))+" and "+str(len(Qsim))+" elements, respectively)")

        kge_components = hydro_stats.fKGE(s=Qsim, o=Qobs, spinup=self.subcatch.spinup)

        return kge_components

    def compute_statistics(self, run_id):

        # DD Extract simulation
        simulated_streamflow = self.read_simulated_streamflow(run_id)

        Qsim, Qobs = self.resample_streamflows(simulated_streamflow, self.observed_streamflow)
        if len(Qobs) != len(Qsim):
            raise Exception("run_id: "+str(run_id)+": observed and simulated streamflow arrays have different number of elements ("+str(len(Qobs))+" and "+str(len(Qsim))+" elements, respectively)")


        stats = {}
        kge_components = hydro_stats.fKGE(s=Qsim, o=Qobs, spinup=self.subcatch.spinup)
        stats['kge'] = kge_components[0]
        stats['corr'] = kge_components[1]
        stats['bias'] = kge_components[2]
        stats['spread'] = kge_components[3]
        stats['sae'] = kge_components[4]
        stats['nse'] = hydro_stats.NS(s=Qsim, o=Qobs, spinup=self.subcatch.spinup)

        index = simulated_streamflow.index
        print(index)
        print(Qsim)
        print(Qobs)
        Q = pd.DataFrame(data={'Sim': Qsim, 'Obs': Qobs}, index=index)

        return Q, stats


    def read_param_history(self):
        path_subcatch = self.subcatch.path
        pHistory = pd.read_csv(os.path.join(path_subcatch, "paramsHistory.csv"), sep=",")[3:]
        return pHistory

    def write_ranked_solution(self, pHistory, path_out=None):
        if path_out is None:
            path_subcatch = self.subcatch.path
        else:
            path_subcatch = path_out
        # Keep only the best 10% of the runs for the selection of the parameters for the next generation
        pHistory = pHistory.sort_values(by="Kling Gupta Efficiency", ascending=False)
        pHistory = pHistory.head(int(max(2, round(len(pHistory) * 0.1))))
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
        pHistory.to_csv(os.path.join(path_subcatch, "pHistoryWRanks.csv"), ',', float_format='%g')

        return pHistory

    def write_pareto_front(self, pHistory, path_out=None):
        if path_out is None:
            path_subcatch = self.subcatch.path
        else:
            path_subcatch = path_out
        param_ranges = self.param_ranges
        # Select the best pareto candidate
        bestParetoIndex = pHistory["paretoRank"].nsmallest(1).index
        # Save the pareto front
        paramvals = np.zeros(shape=(1, len(param_ranges)))
        paramvals[:] = np.NaN
        for ipar, par in enumerate(param_ranges.index):
            paramvals[0][ipar] = pHistory.loc[bestParetoIndex][par]
        pareto_front = pd.DataFrame(
            {
                'effover': pHistory["Kling Gupta Efficiency"].loc[bestParetoIndex],
                'R': pHistory["Kling Gupta Efficiency"].    loc[bestParetoIndex]
            }, index=[0]
        )
        for ii in range(len(param_ranges)):
            pareto_front["param_"+str(ii).zfill(2)+"_"+param_ranges.index[ii]] = paramvals[0,ii]
        pareto_front.to_csv(os.path.join(path_subcatch, "pareto_front.csv"), ',', float_format='%g')

    def process_results(self):

        pHistory = self.read_param_history()
        pHistory_ranked = self.write_ranked_solution(pHistory)

        self.write_pareto_front(pHistory_ranked)
