import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from liscal import hydro_stats, utils


class ObjectiveKGE():
    """
    A class representing the objective function based on the Kling-Gupta Efficiency (KGE) metric.

    Attributes
    ----------
    cfg : ConfigCalibration
        A global configuration settings object.
    subcatch : Subcatchment
        Subcatchment information and data.
    param_ranges : DataFrame
        Parameter ranges for the model.
    weights : list
        Weights for different components of the KGE.
    observed_streamflow : DataFrame
        Observed streamflow data.

    Methods
    -------
    get_parameters(Individual)
        Converts an individual's genes to actual parameter values.
    read_observed_streamflow(observations_file)
        Reads observed streamflow data from a CSV file.
    read_simulated_streamflow_best()
        Reads the best simulated streamflow data.
    read_simulated_streamflow(run_id, start, end)
        Reads simulated streamflow data for a given run.
    resample_streamflows(start, end, simulated_streamflow, observed_streamflow)
        Resamples streamflow data to align simulated and observed streamflows.
    compute_objectives(run_id, start, end, simulated_streamflow, compute_additional_metrics=False)
        Computes the KGE and its components for a given simulation.
    compute_statistics(start, end, simulated_streamflow)
        Computes various statistics for a given simulation.
    update_parameter_history(run_id, parameters, fKGEComponents, evap_objective, additional_metrics, gen, run)
        Updates the parameter history log.
    read_param_history()
        Reads the parameter history from a log file.
    write_ranked_solution(pHistory, path_out=None)
        Ranks solutions based on different metrics and writes the ranked solutions to a file.
    write_pareto_front(pHistory, path_out=None)
        Writes the pareto front to a file.
    process_results()
        Processes and writes results after model execution.
    """

    def __init__(self, cfg, subcatch, read_observations=True):
        self.cfg = cfg
        self.subcatch = subcatch
        self.param_ranges = cfg.param_ranges
        self.weights = [1, 0, 0, 0, 0, 0]

        if read_observations:
            observations_file = os.path.join(subcatch.path_station, 'observations.csv')
            self.observed_streamflow = self.read_observed_streamflow(observations_file)

    def set_custom_multiobjective_weights(self, objective_KGE, objective_corr, objective_bias, objective_y, objective_sae, objective_JSD, objective_KGE_JSD):
        # objective vector in fKGE obective function: [aKGE, r (corr), B, y, se (sae)]
        # the final objective vector is [KGE, (r-1)^2, (B-1)^2, (y-1)^1, sae]
        # THUS: only KGE shoud be maximized, while other terms need to be minimized
        self.weights = [int(objective_KGE), -int(objective_corr), -int(objective_bias), -int(objective_y), -int(objective_sae), -int(objective_JSD), int(objective_KGE_JSD)]


    def get_parameters(self, Individual):
        param_ranges = self.param_ranges
        parameters = [None] * len(param_ranges)
        for ii in range(len(param_ranges)):
            parameters[ii] = Individual[ii]*(float(param_ranges.iloc[ii,1])-float(param_ranges.iloc[ii,0]))+float(param_ranges.iloc[ii,0])

        return parameters

    def read_observed_streamflow(self, observations_file):
        cfg = self.cfg
        
        observed_streamflow = pd.read_csv(observations_file, sep=",", index_col=0)
        # check that date format is correct
        pd.to_datetime(observed_streamflow.index, format='%d/%m/%Y %H:%M', errors='raise')
        observed_streamflow = observed_streamflow[str(self.subcatch.obsid)]
        print('Observations:')
        print('---------------------------------------')
        print(observed_streamflow)
        print('---------------------------------------')

        print(self.subcatch.data.loc['Obs_start'], self.subcatch.data.loc['Obs_end'])

        if (observed_streamflow.index[0] != self.subcatch.data.loc['Obs_start'] or 
            observed_streamflow.index[-1] != self.subcatch.data.loc['Obs_end']):
            raise ValueError('Dates in observations ({} - {}) not coherent with station data({} - {})',
                observed_streamflow.index[0], observed_streamflow.index[-1],
                self.subcatch.data.loc['Obs_start'], self.subcatch.data.loc['Obs_end'])

        return observed_streamflow

    def read_simulated_streamflow_best(self):
        simulated_streamflow = os.path.join(self.subcatch.path_out, 'streamflow_simulated_best.csv')
        simulated_streamflow = pd.read_csv(simulated_streamflow, sep=",", index_col=0)
        simulated_streamflow = simulated_streamflow[str(self.subcatch.obsid)]
        print('Simulated streamflow best:')
        print('---------------------------------------')
        print(simulated_streamflow)
        print('---------------------------------------')

        return simulated_streamflow

    def read_simulated_streamflow(self, run_id, start, end):

        timestep = self.cfg.timestep

        Qsim_tss = os.path.join(self.subcatch.path_out, run_id, 'dis.tss')
        if os.path.isfile(Qsim_tss)==False:
            print('run_id: {}'.format(str(run_id)))
            print('Discharge file path: {}'.format(Qsim_tss))
            raise Exception("No simulated streamflow found. Probably LISFLOOD failed to start? Check the log files of the run!")

        simulated_streamflow = utils.read_tss(Qsim_tss)[1]  # need to take [1] or we get 2d array
        simulated_streamflow[simulated_streamflow==1e31] = np.nan  # PCRaster will put 1e31 instead of NaN, set to NaN to catch errors
        sim_index = []
        for i in range(len(simulated_streamflow.index)):
            sim_index.append((datetime.strptime(start, "%d/%m/%Y %H:%M") + timedelta(minutes=timestep*i)).strftime('%d/%m/%Y %H:%M'))
        simulated_streamflow.index = sim_index
        
        if simulated_streamflow.index[-1] != end:
            raise ValueError('Simulated streamflow with run_id {} not consistent: end date is {} and should be {}'.format(run_id, simulated_streamflow.index[-1], end))

        return simulated_streamflow

    def resample_streamflows(self, start, end, simulated_streamflow, observed_streamflow):
        cfg = self.cfg
        start_pd = datetime.strptime(start, "%d/%m/%Y %H:%M")
        end_pd = datetime.strptime(end, "%d/%m/%Y %H:%M")
        freq = '{}min'.format(cfg.timestep)

        # Finally, extract equal-length arrays from it
        Qobs = observed_streamflow[start:end]
        Qsim = simulated_streamflow[start:end]
        date_range = pd.date_range(start_pd, end_pd, freq=freq)
        if cfg.timestep == 360:
            # DD: Check if daily or 6-hourly observed streamflow is available
            # DD: Aggregate 6-hourly simulated streamflow to daily ones
            if self.subcatch.data["CAL_TYPE"].find("_24h") > -1:
                # start and end have to be in datetime format to avoid "dayfirst" type bugs
                # DD: Overwrite index with date range so we can use Pandas' resampling + mean function to easily average 6-hourly to daily data
                Qsim.index = date_range
                Qsim = Qsim.resample('24H', label="right", closed="right").mean()
                # Same for Qobs
                Qobs.index = date_range
                Qobs = Qobs.resample('24H', label="right", closed="right").mean()
                # return date range 
                date_range = Qobs.index

        elif cfg.timestep == 1440:
            # DD Untested code! DEBUG TODO
            Qobs.index = date_range
            Qobs = Qobs.resample('24H', label="right", closed="right").mean()
            date_range = Qobs.index
        
        else:
            raise Exception('Calibration timesteup {} not supported'.format(cfg.timestep))
       
        Qsim = np.array(Qsim)
        Qobs = np.array(Qobs)

        # Trim nans
        # Qsim = Qsim[~np.isnan(Qobs)]
        # Qobs = Qobs[~np.isnan(Qobs)]

        # we shouldn't have NaNs in the sim array at this point
        if np.isnan(Qsim).any():
            raise Exception('NaN found in Qsim')
        # if np.isnan(Qobs).any():
        #     raise Exception('NaN found in Qobs')

        # if len(date_range) != len(Qsim) or len(date_range) != len(Qobs):
        #     raise Exception('dates, observaed and simulated streamflows not aligned: sizes({}, {}, {})'.format(len(date_range), len(Qobs), len(Qsim)))

        return date_range, Qsim, Qobs

    def compute_objectives(self, run_id, start, end, simulated_streamflow, compute_additional_metrics=False):

        date_range, Qsim, Qobs = self.resample_streamflows(start, end, simulated_streamflow, self.observed_streamflow)
        if len(Qobs) != len(Qsim):
            raise Exception("run_id: "+str(run_id)+": observed and simulated streamflow arrays have different number of elements ("+str(len(Qobs))+" and "+str(len(Qsim))+" elements, respectively)")

        kge_components = hydro_stats.fKGE(s=Qsim, o=Qobs)

        additional_metrics = {}
        if compute_additional_metrics:
            additional_metrics["NSE"] = hydro_stats.NS(s=Qsim, o=Qobs)
            additional_metrics["FDC_FHV"] = hydro_stats.fdc_fhv(sim=Qsim, obs=Qobs)
            additional_metrics["FDC_FLV"] = hydro_stats.fdc_flv(sim=Qsim, obs=Qobs)
            additional_metrics["FDC_mFHV"] = hydro_stats.mFHV(s=Qsim, o=Qobs)
            additional_metrics["FDC_mFLV"] = hydro_stats.mFLV(s=Qsim, o=Qobs)
            additional_metrics["KGE_JSD"],_,_,_,_,additional_metrics["JSD"] = hydro_stats.fKGE_JSD(s=Qsim, o=Qobs)            

        return kge_components, additional_metrics

    def compute_evap_index(self, run_id, precip_budyko, PET_budyko):
        """
        computing evaporative index and budyko compliance
        """
        # print(self.subcatch)
        etactBudyko_tss = os.path.join(self.subcatch.path_out, run_id, 'actETPBUDYKOUpsTS.tss')
        if os.path.isfile(etactBudyko_tss)==False:
            # print('run_id: {}'.format(str(run_id)))
            # print('etactBUDYKO file path: {}'.format(etactBudyko_tss))
            raise Exception("No simulated etactBudyko found. Probably LISFLOOD failed to start? or you are not creating the correct output? check settings.xml file")

        etactBudyko = utils.read_tss(etactBudyko_tss)[1]  # need to take [1] or we get 2d array
        etactBudyko[etactBudyko==1e31] = np.nan
        evap_index=hydro_stats.evap_index_BUDYKO(precip_budyko,etactBudyko.sum(),PET_budyko)
        return evap_index

    def compute_statistics(self, start, end, simulated_streamflow):

        date_range, Qsim, Qobs = self.resample_streamflows(start, end, simulated_streamflow, self.observed_streamflow)
        if len(Qobs) != len(Qsim):
            raise Exception("Observed and simulated streamflow arrays have different number of elements ("+str(len(Qobs))+" and "+str(len(Qsim))+" elements, respectively)")

        stats = {}
        kge_components = hydro_stats.fKGE(s=Qsim, o=Qobs)
        stats['kge'] = kge_components[0]
        stats['corr'] = kge_components[1]
        stats['bias'] = kge_components[2]
        stats['spread'] = kge_components[3]
        stats['sae'] = kge_components[4]
        stats['nse'] = hydro_stats.NS(s=Qsim, o=Qobs)

        index = pd.to_datetime(date_range, format='%d/%m/%Y %H:%M')
        Q = pd.DataFrame(data={'Sim': Qsim, 'Obs': Qobs}, index=index)

        return Q, stats

    def update_parameter_history(self, run_id, parameters, fKGEComponents, EVAP_index, additional_metrics, gen, run):

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
            for i in [str(ip) + "," for ip in ["Kling Gupta Efficiency", "Correlation", "Signal ratio (s/o) (Bias)", "Noise ratio (s/o) (Spread)", "sae","Evaporative Index","Fractional Budyko Distance"]]:            
                paramsHistory += i
            for i in [str(ip) + "," for ip in additional_metrics]:
                paramsHistory += i
            for i in [str(ip) + "," for ip in ["generation", "runNumber"]]:
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
        for i in [str(ip) + "," for ip in EVAP_index]:
            paramsHistory += i
        for i in [str(additional_metrics[ip]) + "," for ip in additional_metrics]:
            paramsHistory += i
        paramsHistory += str(gen) + ","
        paramsHistory += str(run)
        paramsHistory += "\n"
        paramsHistoryFile.write(paramsHistory)
        paramsHistoryFile.close()

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
