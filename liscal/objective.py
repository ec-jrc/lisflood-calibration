import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from liscal import hydro_stats, utils
from liscal.stations import time_step_from_type


class Objective():
    """
    Base class for objective functions used in calibration.

    Provides shared methods for all objective subclasses and abstract stubs
    for properties and methods that must be implemented by subclasses
    (ObjectiveKGE, ObjectiveKGEJSD, ObjectiveMulti).

    Uses raise NotImplementedError instead of ABC metaclass.
    """

    def __init__(self, cfg, subcatch, read_observations=True):
        self.cfg = cfg
        self.subcatch = subcatch
        self.param_ranges = cfg.param_ranges

        if read_observations:
            observations_file = os.path.join(subcatch.path_station, 'observations.csv')
            self.observed_streamflow = self.read_observed_streamflow(observations_file)

    @property
    def weights(self):
        raise NotImplementedError("Subclasses must define weights")

    def filter_objectives_for_deap(self, objectives, additional_metrics):
        raise NotImplementedError("Subclasses must implement filter_objectives_for_deap")

    def compute_objectives(self, run_id, start, end, simulated_streamflow, compute_additional_metrics=False):
        raise NotImplementedError("Subclasses must implement compute_objectives")

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

    def read_simulated_streamflow(self, run_id, start, end, Qsim_tss = None):

        timestep = self.cfg.timestep
        if Qsim_tss is None:
            Qsim_tss = os.path.join(self.subcatch.path_out, run_id, 'dis.tss')
        if os.path.isfile(Qsim_tss)==False:
            print('run_id: {}'.format(str(run_id)))
            print('Discharge file path: {}'.format(Qsim_tss))
            raise Exception(f"No simulated streamflow {Qsim_tss} found. Probably LISFLOOD failed to start? Check the log files of the run!")

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
            if time_step_from_type(self.subcatch.data["CAL_TYPE"]) == 24:
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

    def compute_evap_index(self, run_id, precip_budyko, PET_budyko, etactBudyko_tss):
        """
        computing evaporative index and budyko compliance
        """
        # print(self.subcatch)        
        if os.path.isfile(etactBudyko_tss)==False:
            # print('run_id: {}'.format(str(run_id)))
            # print('etactBUDYKO file path: {}'.format(etactBudyko_tss))
            raise Exception(f"No simulated etactBudyko found {etactBudyko_tss}. Probably LISFLOOD failed to start? or you are not creating the correct output? check settings.xml file")

        etactBudyko = utils.read_tss(etactBudyko_tss)[1]  # need to take [1] or we get 2d array
        etactBudyko[etactBudyko==1e31] = np.nan
        evap_index=hydro_stats.evap_index_BUDYKO(precip_budyko,etactBudyko.sum(),PET_budyko)
        return evap_index

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
            # these columns should match the order of the updatePopulationFromHistory function
            for i in [str(ip) + "," for ip in ["Kling Gupta Efficiency", "Correlation", "Signal ratio (s/o) (Bias)", "Noise ratio (s/o) (Spread)", "sae", "Evaporative Index", "Fractional Budyko Distance"]]:
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



class ObjectiveKGEBase(Objective):
    """Intermediate base class for all KGE-based objective functions.

    Consolidates KGE-related shared logic (statistics computation, ranked
    solution writing, Pareto front generation, summary file writing) so that
    future non-KGE objectives can inherit from Objective directly without
    KGE-specific baggage.
    """

    @property
    def kge_column(self):
        raise NotImplementedError("Subclasses must define kge_column")

    @property
    def kge_rank_name(self):
        raise NotImplementedError("Subclasses must define kge_rank_name")

    @property
    def is_kge_jsd(self):
        """True if the primary objective is KGE_JSD (not plain KGE)."""
        return (self.weights["KGE"] == 0) and (self.weights["KGE_JSD"] != 0)

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

    def _compute_additional_metrics(self, Qsim, Qobs):
        """
        Compute additional hydrological metrics beyond the core KGE components.

        Parameters
        ----------
        Qsim : np.ndarray
            Simulated streamflow array.
        Qobs : np.ndarray
            Observed streamflow array.

        Returns
        -------
        dict
            Dictionary of additional metric names to values.
        """
        dt = time_step_from_type(self.subcatch.data["CAL_TYPE"])
        metrics = {}
        metrics["NSE"] = hydro_stats.NS(s=Qsim, o=Qobs)
        metrics["FDC_FHV"] = hydro_stats.fdc_fhv(sim=Qsim, obs=Qobs)
        metrics["FDC_FLV"] = hydro_stats.fdc_flv(sim=Qsim, obs=Qobs)
        metrics["FDC_mFHV"] = hydro_stats.mFHV(s=Qsim, o=Qobs)
        metrics["FDC_mFLV"] = hydro_stats.mFLV(s=Qsim, o=Qobs)
        metrics["KGE_JSD"], _, _, _, _, metrics["JSD"] = hydro_stats.fKGE_JSD(s=Qsim, o=Qobs, dt=dt)
        return metrics

    def write_ranked_solution(self, pHistory, path_out=None):
        KGEcolumn = self.kge_column
        KGERankName = self.kge_rank_name
        if path_out is None:
            path_subcatch = self.subcatch.path
        else:
            path_subcatch = path_out
        # Keep only the best 10% of the runs for the selection of the parameters for the next generation
        pHistory = pHistory.sort_values(by=KGEcolumn, ascending=False)
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
        pHistory = pHistory.sort_values(by=KGEcolumn, ascending=False)
        pHistory[KGERankName] = [minOffset + float(i + 1) * (maxOffset - minOffset) / n for i, ii in enumerate(pHistory[KGEcolumn].values)]
        # Give pareto score
        pHistory["paretoRank"] = pHistory["corrRank"].values * pHistory["saeRank"].values * pHistory[KGERankName].values
        pHistory = pHistory.sort_values(by="paretoRank", ascending=True)
        pHistory.to_csv(os.path.join(path_subcatch, "pHistoryWRanks.csv"), ',', float_format='%g')

        return pHistory

    def write_pareto_front(self, pHistory, isKGE_JSD=False, path_out=None, pareto_front_filename = "pareto_front.csv"):
        if isKGE_JSD: # we are using KGE_JSD objective
            KGEcolumn="KGE_JSD"
            KGERankName="(KGEJSD)"
        else:
            KGEcolumn="Kling Gupta Efficiency"
            KGERankName="(KGE)"
        if path_out is None:
            path_subcatch = self.subcatch.path
        else:
            path_subcatch = path_out
        param_ranges = self.param_ranges
        # Select the best pareto candidate
        bestParetoIndex = pHistory["paretoRank"].nsmallest(1).index
        # Save the pareto front
        paramvals = np.zeros(shape=(1, len(param_ranges)))
        paramvals[:] = np.nan
        for ipar, par in enumerate(param_ranges.index):
            paramvals[0][ipar] = pHistory.loc[bestParetoIndex][par]

        pareto_front = pd.DataFrame(
            {
                f'effover{KGERankName}': pHistory[KGEcolumn].loc[bestParetoIndex],
                f'R{KGERankName}': pHistory[KGEcolumn].    loc[bestParetoIndex]
            }
        )
        for ii in range(len(param_ranges)):
            pareto_front["param_"+str(ii).zfill(2)+"_"+param_ranges.index[ii]] = paramvals[0,ii]
        pareto_front.to_csv(os.path.join(path_subcatch, pareto_front_filename), ',', float_format='%g')

    def write_summary_file(self, selObjFun):
        # use station_data.csv info, as they are consistent with StepStart and StepEnd dates used in calibration xml files
        spinup = int(float(self.subcatch.data['Spinup_days']))
        obs_start = datetime.strptime(self.subcatch.data['Split_date'],"%d/%m/%Y %H:%M").strftime('%d/%m/%Y %H:%M')
        obs_end = datetime.strptime(self.subcatch.data['Obs_end'],"%d/%m/%Y %H:%M").strftime('%d/%m/%Y %H:%M')
        cal_start = (datetime.strptime(obs_start,"%d/%m/%Y %H:%M") - timedelta(days=spinup)).strftime('%d/%m/%Y %H:%M')
        cal_end = datetime.strptime(obs_end,"%d/%m/%Y %H:%M").strftime('%d/%m/%Y %H:%M')
        # Check if reservoir dynamic event affected calibration:
        resDyn = "OFF"
        if self.cfg.reservoir_events is not None:
            strFilteredReservoirMap=os.path.join(self.subcatch.path_station, 'FilteredReservoirMap.nc')
            if os.path.exists(strFilteredReservoirMap):     
                resDyn = "ON"   
        message = f"Calibration summary for catchment {self.subcatch.obsid}\n" \
            f"Selected objective function = {selObjFun}\n" \
            f"Calibration start date = {cal_start}\n" \
            f"Calibration end date = {cal_end}\n" \
            f"Reservoir dynamic events = {resDyn}\n"
        print(message)
        summary_filename = f"calibration_summary_{selObjFun}_{datetime.strptime(cal_start, '%d/%m/%Y %H:%M').strftime('%d%m%Y')}_{datetime.strptime(cal_end, '%d/%m/%Y %H:%M').strftime('%d%m%Y')}_res{resDyn}.txt"
        file_path = os.path.join(self.subcatch.path,summary_filename)
        # Open the file in write mode
        with open(file_path, 'w') as file:
            # Write the message to the file
            file.write(message)


class ObjectiveKGE(ObjectiveKGEBase):
    """
    Objective subclass for single-KGE calibration.

    Uses only the Kling-Gupta Efficiency as the fitness metric.
    Inherits all shared methods from Objective and implements the
    type-specific properties and methods with hardcoded KGE-only behavior.
    """

    @property
    def weights(self):
        return {"KGE": 1, "CORR": 0, "BIAS": 0, "Y": 0, "SAE": 0, "JSD": 0, "KGE_JSD": 0}

    @property
    def kge_column(self):
        return "Kling Gupta Efficiency"

    @property
    def kge_rank_name(self):
        return "KGERank"

    def filter_objectives_for_deap(self, objectives, additional_metrics):
        """Return single-element list: [KGE value]."""
        return [objectives[0]]

    def compute_objectives(self, run_id, start, end, simulated_streamflow, compute_additional_metrics=False):
        """Compute KGE components only. Calls fKGE."""
        date_range, Qsim, Qobs = self.resample_streamflows(start, end, simulated_streamflow, self.observed_streamflow)
        if len(Qobs) != len(Qsim):
            raise Exception("run_id: "+str(run_id)+": observed and simulated streamflow arrays have different number of elements ("+str(len(Qobs))+" and "+str(len(Qsim))+" elements, respectively)")

        kge_components = hydro_stats.fKGE(s=Qsim, o=Qobs)

        additional_metrics = {}
        if compute_additional_metrics:
            additional_metrics = self._compute_additional_metrics(Qsim, Qobs)

        return kge_components, additional_metrics


class ObjectiveKGEJSD(ObjectiveKGEBase):
    """
    Objective subclass for single-KGE_JSD calibration.

    Uses the combined KGE-JSD metric as the fitness metric.
    Always calls both fKGE and fKGE_JSD since the JSD value is needed
    for the fitness function.
    """

    @property
    def weights(self):
        return {"KGE": 0, "CORR": 0, "BIAS": 0, "Y": 0, "SAE": 0, "JSD": 0, "KGE_JSD": 1}

    @property
    def kge_column(self):
        return "KGE_JSD"

    @property
    def kge_rank_name(self):
        return "KGEJSDRank"

    def filter_objectives_for_deap(self, objectives, additional_metrics):
        """Return single-element list: [KGE_JSD value]."""
        return [additional_metrics["KGE_JSD"]]

    def compute_objectives(self, run_id, start, end, simulated_streamflow, compute_additional_metrics=False):
        """Compute KGE + JSD components. Always calls both fKGE and fKGE_JSD."""
        date_range, Qsim, Qobs = self.resample_streamflows(
            start, end, simulated_streamflow, self.observed_streamflow
        )
        if len(Qobs) != len(Qsim):
            raise Exception("run_id: "+str(run_id)+": observed and simulated streamflow arrays have different number of elements ("+str(len(Qobs))+" and "+str(len(Qsim))+" elements, respectively)")

        kge_components = hydro_stats.fKGE(s=Qsim, o=Qobs)

        # Always compute JSD since it's the fitness metric
        dt = time_step_from_type(self.subcatch.data["CAL_TYPE"])
        kge_jsd, _, _, _, _, jsd = hydro_stats.fKGE_JSD(s=Qsim, o=Qobs, dt=dt)

        additional_metrics = {"KGE_JSD": kge_jsd, "JSD": jsd}
        if compute_additional_metrics:
            extra = self._compute_additional_metrics(Qsim, Qobs)
            # Merge without overwriting KGE_JSD/JSD which are already computed
            for k, v in extra.items():
                if k not in additional_metrics:
                    additional_metrics[k] = v

        return kge_components, additional_metrics


class ObjectiveMulti(ObjectiveKGEBase):
    """
    Objective subclass for multi-objective calibration.

    Handles arbitrary combinations of objectives (KGE, CORR, BIAS, Y, SAE, JSD, KGE_JSD).
    Weights are computed from cfg.deap_param.objectives_list.
    Applies (x-1)^2 transform to CORR/BIAS/Y for minimization in DEAP.
    """

    def __init__(self, cfg, subcatch, read_observations=True):
        super().__init__(cfg, subcatch, read_observations)
        # Pre-compute weights from cfg
        obj_list = cfg.deap_param.objectives_list
        self._weights = {
            "KGE": 1 if 'KGE' in obj_list else 0,
            "CORR": -1 if 'CORR' in obj_list else 0,
            "BIAS": -1 if 'BIAS' in obj_list else 0,
            "Y": -1 if 'Y' in obj_list else 0,
            "SAE": -1 if 'SAE' in obj_list else 0,
            "JSD": -1 if 'JSD' in obj_list else 0,
            "KGE_JSD": 1 if 'KGE_JSD' in obj_list else 0,
        }

    @property
    def weights(self):
        return self._weights

    @property
    def kge_column(self):
        if self._weights["KGE_JSD"] != 0 and self._weights["KGE"] == 0:
            return "KGE_JSD"
        return "Kling Gupta Efficiency"

    @property
    def kge_rank_name(self):
        if self._weights["KGE_JSD"] != 0 and self._weights["KGE"] == 0:
            return "KGEJSDRank"
        return "KGERank"

    def filter_objectives_for_deap(self, objectives, additional_metrics):
        """
        Apply (x-1)^2 transform to CORR/BIAS/Y, filter by non-zero weights,
        append JSD/KGE_JSD if active.

        Parameters
        ----------
        objectives : tuple
            A 5-tuple: (KGE, corr, bias, spread, SAE)
        additional_metrics : dict
            Dictionary containing 'JSD' and/or 'KGE_JSD' values if computed.

        Returns
        -------
        list
            Filtered list of objective values for DEAP fitness evaluation.
        """
        # Map the 5-tuple to their dict keys
        component_map = {
            "KGE": objectives[0],
            "CORR": objectives[1],
            "BIAS": objectives[2],
            "Y": objectives[3],
            "SAE": objectives[4],
        }
        # Transform components for minimization
        component_map["CORR"] = (component_map["CORR"] - 1) ** 2
        component_map["BIAS"] = (component_map["BIAS"] - 1) ** 2
        component_map["Y"] = (component_map["Y"] - 1) ** 2

        filtered = [component_map[k] for k in ("KGE", "CORR", "BIAS", "Y", "SAE")
                    if self._weights[k] != 0]
        if self._weights["JSD"] != 0:
            filtered.append(additional_metrics["JSD"])
        if self._weights["KGE_JSD"] != 0:
            filtered.append(additional_metrics["KGE_JSD"])
        return filtered

    def compute_objectives(self, run_id, start, end, simulated_streamflow, compute_additional_metrics=False):
        """
        Compute KGE, and if JSD/KGE_JSD in objectives, also compute fKGE_JSD.

        Parameters
        ----------
        run_id : str or int
            Identifier for the current run.
        start : str
            Start date string in '%d/%m/%Y %H:%M' format.
        end : str
            End date string in '%d/%m/%Y %H:%M' format.
        simulated_streamflow : pd.Series
            Simulated streamflow time series.
        compute_additional_metrics : bool, optional
            Whether to compute extra metrics (NSE, FDC, etc.). Default False.

        Returns
        -------
        tuple
            (kge_components, additional_metrics) where kge_components is the
            output of fKGE and additional_metrics is a dict of extra metrics.
        """
        date_range, Qsim, Qobs = self.resample_streamflows(
            start, end, simulated_streamflow, self.observed_streamflow
        )
        if len(Qobs) != len(Qsim):
            raise Exception("run_id: "+str(run_id)+": observed and simulated streamflow arrays have different number of elements ("+str(len(Qobs))+" and "+str(len(Qsim))+" elements, respectively)")

        kge_components = hydro_stats.fKGE(s=Qsim, o=Qobs)

        additional_metrics = {}
        # Compute JSD if any JSD-related objective is active
        if self._weights["JSD"] != 0 or self._weights["KGE_JSD"] != 0:
            dt = time_step_from_type(self.subcatch.data["CAL_TYPE"])
            kge_jsd, _, _, _, _, jsd = hydro_stats.fKGE_JSD(
                s=Qsim, o=Qobs, dt=dt
            )
            additional_metrics["KGE_JSD"] = kge_jsd
            additional_metrics["JSD"] = jsd

        if compute_additional_metrics:
            extra = self._compute_additional_metrics(Qsim, Qobs)
            # Don't overwrite JSD values if already computed
            for k, v in extra.items():
                if k not in additional_metrics:
                    additional_metrics[k] = v

        return kge_components, additional_metrics


def create_objective(cfg, subcatch, read_observations=True):
    """
    Factory function: create the appropriate Objective subclass
    based on the configured objectives_list.
    """
    objectives_list = cfg.deap_param.objectives_list

    if not objectives_list:
        raise ValueError(
            "objectives_list is empty. At least one objective must be configured."
        )

    if len(objectives_list) == 1:
        if objectives_list[0] == 'KGE':
            return ObjectiveKGE(cfg, subcatch, read_observations)
        elif objectives_list[0] == 'KGE_JSD':
            return ObjectiveKGEJSD(cfg, subcatch, read_observations)
        else:
            # Single non-standard objective → treat as multi
            return ObjectiveMulti(cfg, subcatch, read_observations)
    else:
        return ObjectiveMulti(cfg, subcatch, read_observations)
