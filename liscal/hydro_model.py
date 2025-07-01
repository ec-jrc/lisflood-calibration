import os
import shutil
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import time

# lisflood
import lisf1
from lisflood.global_modules.decorators import Cache
from lisflood.global_modules.settings import LisSettings

from liscal import stations, utils


class HydrologicalModel():
    """
    A class representing a hydrological model for calibration and simulation.

    Attributes
    ----------
    cfg : ConfigCalibration
        A global configuration settings object.
    subcatch : Subcatchment
        Subcatchment information and data.
    lis_template : LisfloodTemplate
        Template for LISFLOOD input files.
    lock_mgr : LockManager
        Manager for synchronization locks and parallelisation.
    objective : Objective
        Objective function class for calibration.
    obs_start : str
        Start date for the observation period.
    obs_end : str
        End date for the observation period.
    cal_start : str
        Start date for the calibration period.
    cal_end : str
        End date for the calibration period.
    prerun_start : str
        Start date for the prerun period.
    prerun_end : str
        End date for the prerun period.
    lisflood_cache_size : int
        Size of the cache after initial LISFLOOD run.

    Methods
    -------
    init_run()
        Initialize the model run, caching static maps and forcings.
    run(Individual)
        Run the model for a given set of parameters.
    """

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
            spinup = int(float(subcatch.data['Spinup_days']))
            self.obs_start = datetime.strptime(subcatch.data['Split_date'],"%d/%m/%Y %H:%M").strftime('%d/%m/%Y %H:%M')
            self.obs_end = datetime.strptime(subcatch.data['Obs_end'],"%d/%m/%Y %H:%M").strftime('%d/%m/%Y %H:%M')
            self.cal_start = (datetime.strptime(self.obs_start,"%d/%m/%Y %H:%M") - timedelta(days=spinup)).strftime('%d/%m/%Y %H:%M')
            self.cal_end = datetime.strptime(subcatch.data['Obs_end'],"%d/%m/%Y %H:%M").strftime('%d/%m/%Y %H:%M')

        self.prerun_start = cfg.prerun_start.strftime('%d/%m/%Y %H:%M')
        self.prerun_end = cfg.prerun_end.strftime('%d/%m/%Y %H:%M')

    def init_settings(self):
        """
        Initialize the settings file
        """
        # dummy Individual, doesn't matter here
        param_ranges = self.cfg.param_ranges
        Individual = 0.5*np.ones(len(param_ranges))

        cfg = self.cfg

        run_id = str(0)

        out_dir = os.path.join(self.subcatch.path_out, run_id)
        os.makedirs(out_dir, exist_ok=True)

        parameters = self.objective.get_parameters(Individual)
        prerun_file, run_file = self.lis_template.write_init(run_id, self.prerun_start, self.prerun_end, self.cal_start, self.cal_end, cfg.param_ranges, parameters)  
        return prerun_file, run_file       

    def init_run(self):
        """
        Initialize the model run. This method prepares the model by caching static maps and forcings.
        It runs LISFLOOD in initialization mode.
        """

        prerun_file, run_file = self.init_settings()

        print('---------------------------------------------------------')
        print('Intialising prerun: caching static maps and forcings')
        print('---------------------------------------------------------')
        print('Cache size before initialisation: {}'.format(Cache.size()))
        lisf1.main(prerun_file, '-i')
        print('Cache size after initialising prerun: {}'.format(Cache.size()))

        print('---------------------------------------------------------')
        print('Intialising run: caching static maps and forcings')
        print('---------------------------------------------------------')
        lisf1.main(run_file, '-i')
        print('Cache size after initialising run: {}'.format(Cache.size()))

        print('---------------------------------------------------------')
        print('End of the Initialisaton')
        print('---------------------------------------------------------')
        # store lisflood cache size to make sure we don't load anything else after that
        self.lisflood_cache_size = Cache.size()
        self.lissettings = LisSettings.instance()

    def run(self, Individual):
        """
        Run the model for a given set of parameters.

        Parameters
        ----------
        Individual : array-like
            Array of parameter values for the model run.

        Returns
        -------
        array-like
            The computed objectives for the given set of parameters.
        """

        cfg = self.cfg

        gen = self.lock_mgr.get_gen()
        run = self.lock_mgr.increment_run()
        print('Generation {}, run {}'.format(gen, run))

        run_id = '{}_{}'.format(gen, run)
        out_dir = os.path.join(self.subcatch.path_out, run_id)
        os.makedirs(out_dir, exist_ok=True)

        parameters = self.objective.get_parameters(Individual)

        prerun_file, run_file = self.lis_template.write_template(run_id, self.prerun_start, self.prerun_end, self.cal_start, 
                                                                 self.cal_end, cfg, out_dir, self.subcatch.path_station, parameters)

        
            
        lisf1.main(prerun_file, '-v')
        lisf1.main(run_file, '-v')
        Qsim_tss=LisSettings.instance().binding['DisTS']  
        simulated_streamflow = self.objective.read_simulated_streamflow(run_id, self.cal_start, self.cal_end, Qsim_tss)
        objectives, additional_metrics = self.objective.compute_objectives(run_id, self.obs_start, self.obs_end, simulated_streamflow, compute_additional_metrics=True)
        precip_budyko=self.subcatch.data['precip_budyko']
        PET_budyko=self.subcatch.data['PET_budyko']

        etactBudyko_tss=LisSettings.instance().binding['actETPBUDYKOUpsTS'] 
        evap_objective=self.objective.compute_evap_index(run_id,precip_budyko,PET_budyko, etactBudyko_tss)
        with self.lock_mgr.lock:
            self.objective.update_parameter_history(run_id, parameters, objectives, evap_objective, additional_metrics, gen, run)

        # return only obectives with non zero weight!
        non_zero_indices = [index for index, weight in enumerate(self.objective.weights) if weight != 0]
        objectives=list(objectives)
        # the KGE formula is aKGE = 1 - np.sqrt((r - 1) ** 2 + (B - 1) ** 2 + (y - 1) ** 2) 
        # THUS: r (corr), B (bias) and y terms need to be adjusted to be minimized:
        objectives[1] = (objectives[1]-1)**2    # r (corr)
        objectives[2] = (objectives[2]-1)**2    # B (bias)
        objectives[3] = (objectives[3]-1)**2    # y
        filtered_objectives = [objectives[i] for i in non_zero_indices if i<5]     
        #add JSD to objective vector
        if 5 in non_zero_indices:
            filtered_objectives.append(additional_metrics["JSD"])
        if 6 in non_zero_indices:
            filtered_objectives.append(additional_metrics["KGE_JSD"])   # KGE_JSD
        return filtered_objectives              


def read_parameters(path_subcatch):
    """
    Read optimised parameter values from a CSV file.

    Parameters
    ----------
    path_subcatch : str
        Path to the subcatchment directory.

    Returns
    -------
    list
        List of parameter values.
    """

    paramvals = pd.read_csv(os.path.join(path_subcatch, "pareto_front.csv"),sep=",")

    name_params= paramvals.columns
    names=name_params[3:]
    parameters=list()
    for indx in range(0,len(names)):
        print(names[indx], paramvals[names[indx]].values[0])
        parameters.append(paramvals[names[indx]].values[0])

    return parameters


def simulated_best_tss2csv(cfg, subcatch, run_id, forcing_start, dataname, outname):
    """
    Convert a .tss file to a CSV file and copy the .tss file to a specified location.

    Parameters
    ----------
    cfg : ConfigCalibration
        A global configuration settings object.
    subcatch : Subcatchment
        Subcatchment information and data.
    run_id : str
        ID of the model run.
    forcing_start : datetime
        Start date of forcing data.
    dataname : str
        Name of the data in the .tss file.
    outname : str
        Prefix for the output CSV file.
    """

    tss_file = os.path.join(subcatch.path_out, run_id, dataname)

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
    """
    Move around inflow .tss files for a subcatchment to make sure they are not overwritten.

    Parameters
    ----------
    path_subcatch : str
        Path to the subcatchment directory.
    """

    inflow_tss = os.path.join(path_subcatch, "inflow", "chanq.tss")
    inflow_tss_last_run = os.path.join(path_subcatch, "inflow", "chanq_last_run.tss")
    inflow_tss_cal = os.path.join(path_subcatch, "inflow", "chanq_cal.tss")
    if os.path.isfile(inflow_tss) or os.path.isfile(inflow_tss_cal):
        print(inflow_tss)
        print(inflow_tss_cal)
        print(inflow_tss_last_run)
        os.rename(inflow_tss, inflow_tss_cal)
        os.rename(inflow_tss_last_run, inflow_tss)

# utility function for merging tss files from dynamic reservoir warmstart routine
# to be moved in a new file reservoir.py, together with all dynamic reservoir management code
def merge_tss_files(tss_file_list, output_tss_file):
    concatenated_data = []
    all_column_names = set()
    file_data = []

    for file_path in tss_file_list:
        if not os.path.exists(file_path):
            print(f'{file_path} not found. Skipping...')
        else:
            with open(file_path, 'r') as file:
                lines = file.readlines()

            # Remove empty lines at the end of the file
            lines = [line for line in lines if line.strip()]

            # Parse header to determine metadata lines and column count
            header_lines, data_start_index, n_columns, column_names = parse_tss_header(lines)
            all_column_names.update(column_names)
            file_data.append((header_lines, data_start_index, column_names, lines[data_start_index:]))
    if len(file_data)>0:
        # Sort column names to have a consistent order
        all_column_names = sorted(all_column_names)

        # Build the final merged data
        merged_header = create_merged_header(file_data[0][0], all_column_names)
        concatenated_data.extend(merged_header)

        for _, data_start_index, column_names, data_lines in file_data:
            column_index_map = {name: i for i, name in enumerate(column_names)}
            for line in data_lines:
                parts = line.split()
                timestep = parts[0]
                values = parts[1:]
                merged_line = [f"{timestep:>9}"]  # Format the timestep with width of 9

                for col_name in all_column_names:
                    if col_name in column_index_map:
                        value_index = column_index_map[col_name]
                        merged_line.append(f"{float(values[value_index]):>15}")  # Format values with width of 15
                    else:
                        merged_line.append(f"{np.nan:>15}")  # Fill missing columns with nan

                concatenated_data.append(''.join(merged_line) + '\n')

        # Write the concatenated data to the output file
        with open(output_tss_file, 'w') as output_file:
            output_file.writelines(concatenated_data)

        print(f'Merged file created: {output_tss_file}')

def parse_tss_header(lines):
    # Read header to get number of columns and metadata
    n_columns = int(lines[1].strip()) - 1     # exclude timestep from columns
    column_names = [lines[i].strip() for i in range(3, 3 + n_columns)]
    data_start_index = 3 + n_columns
    return lines[:data_start_index], data_start_index, n_columns, column_names

def create_merged_header(header_lines, all_column_names):
    # Create a new header with all column names
    merged_header = header_lines[:3]  # Keep the first three lines (timeseries, number of columns, timestep)
    merged_header[1] = f"{len(all_column_names) + 1}\n"  # Update column count + timestep
    merged_header.extend([name + '\n' for name in all_column_names])
    return merged_header

def generate_outlet_streamflow(cfg, subcatch, lis_template, subperiods, filtered_reservoir_events):
    """
    Generate outlet streamflow using the calibrated parameters set by running LISFLOOD.

    Parameters
    ----------
    cfg : ConfigCalibration
        A global configuration settings object.
    subcatch : Subcatchment
        Subcatchment information and data.
    lis_template : LisfloodTemplate
        Template object for LISFLOOD input settings file.
    """

    # stage_inflows(subcatch.path)

    print(">> Running LISFLOOD using the \"best\" parameter set")
    parameters = read_parameters(subcatch.path)

    run_id = 'long_term_run'
    out_dir = os.path.join(subcatch.path_out, run_id)
    os.makedirs(out_dir, exist_ok=True)

    # use forcings start and end for prerun and run
    prerun_start = cfg.longterm_prerun_start.strftime('%d/%m/%Y %H:%M')
    prerun_end = cfg.longterm_prerun_end.strftime('%d/%m/%Y %H:%M')
    run_start = cfg.forcing_start.strftime('%d/%m/%Y %H:%M')
    run_end = cfg.forcing_end.strftime('%d/%m/%Y %H:%M')
    prerun_file, run_file = lis_template.write_template(run_id, prerun_start, prerun_end, run_start, 
                                                        run_end, cfg, out_dir, subcatch.path_station, parameters, write_states=True)

    # FIRST LISFLOOD RUN
    lisf1.main(prerun_file, '-v')

    # DD JIRA issue https://efascom.smhi.se/jira/browse/ECC-1210 to avoid overwriting the bestrun avgdis.end.nc
    cmd = 'cp {0}/out/{1}/avgdis.nc {0}/out/{1}/avgdis.simulated_bestend.nc'.format(subcatch.path, run_id)
    utils.run_cmd(cmd)
    cmd = 'cp {0}/out/{1}/lzavin.nc {0}/out/{1}/lzavin.simulated_bestend.nc'.format(subcatch.path, run_id)
    utils.run_cmd(cmd)

    if subperiods is None:
        # SECOND LISFLOOD RUN
        lisf1.main(run_file, '-q')
    else:
        #check if lakes and MCT were switched to OFF during the FIRST LISFLOOD RUN, for the write_warmstart_settings_files
        instsettings = LisSettings.instance()
        includeLakes, includeMCT = instsettings.options['simulateLakes'], instsettings.options['MCTRouting']

        # generate subperiods settings files        
        warmstart_run_files = lis_template.write_warmstart_settings_files(run_id, run_file, subcatch.path_station, subperiods, includeLakes, includeMCT)
        # run different periods with warm start
        idx=0
        list_of_variable_to_merge = None
        var_files = {}
        original_var_tss_name = {}
        for idx, warmstart_run_file in enumerate(warmstart_run_files):
            settings = LisSettings(warmstart_run_file, "")
            if idx>0:
                rsfil_map_name = settings.binding['ReservoirFillEnd']
                rsfil_map_name = rsfil_map_name[:-3] if rsfil_map_name.lower().endswith('.nc') else rsfil_map_name
                # copy the ReservoirFill end map to a backup before editing it for the next run
                cmd = f'cp {rsfil_map_name}.nc {rsfil_map_name}_ws_{idx-1}.nc'
                utils.run_cmd(cmd)
                sub_start, _ = subperiods[idx]
                map_name=os.path.basename(rsfil_map_name)       # map_name will not contain ".nc"
                stations.update_rsfil_netcdf_map(map_name, filtered_reservoir_events, settings, sub_start)
            else:
                list_of_variable_to_merge = [k for k in settings.report_timeseries.keys()]
                for varName in list_of_variable_to_merge:
                    var_files[varName] = []
                    original_var_tss_name[varName] = None
            lisf1.main(warmstart_run_file, '-q')
            # rename chanq, dischage and other tss files to keep info for the final merge
            for varName in list_of_variable_to_merge:
                original_var_tss_name[varName] = settings.binding[varName]
                var_tss_name = original_var_tss_name[varName][:-4] if original_var_tss_name[varName].lower().endswith('.tss') else original_var_tss_name[varName]
                var_tss_name_dest = f'{var_tss_name}_ws_{idx}.tss'
                cmd = f'mv {var_tss_name}.tss {var_tss_name_dest}'
                utils.run_cmd(cmd)
                var_files[varName].append(var_tss_name_dest)

        # merge dis and chanq files
        for varName in list_of_variable_to_merge:
            merge_tss_files(var_files[varName], original_var_tss_name[varName])
            
            

    # DD JIRA issue https://efascom.smhi.se/jira/browse/ECC-1210 restore the backup
    cmd = 'rm {0}/out/{1}/avgdis.nc {0}/out/{1}/lzavin.nc'.format(subcatch.path, run_id)
    utils.run_cmd(cmd)
    Qsim_tss=LisSettings.instance().binding['DisTS']
    Chanq_tss=LisSettings.instance().binding['ChanqTS']
    simulated_best_tss2csv(cfg, subcatch, run_id, cfg.forcing_start, Qsim_tss, 'streamflow')
    simulated_best_tss2csv(cfg, subcatch, run_id, cfg.forcing_start, Chanq_tss, 'chanq')


def generate_timing(cfg, subcatch, lis_template, param_target, outfile, start, end):
    """
    Generate timing benchmarks for the model run.

    Parameters
    ----------
    cfg : ConfigCalibration
        A global configuration settings object.
    subcatch : Subcatchment
        Subcatchment information and data.
    lis_template : LisfloodTemplate
        Template object for LISFLOOD input settings file.
    param_target : array-like
        Target parameter values for the benchmark.
    outfile : str
        Path for the output file.
    start : str
        Start date for the benchmark period.
    end : str
        End date for the benchmark period.
    """

    run_id = 'T'
    out_dir = os.path.join(subcatch.path_out, run_id)
    os.makedirs(out_dir, exist_ok=True)

    param_ranges = cfg.param_ranges
    parameters = [None] * len(param_ranges)
    for ii in range(len(param_ranges)):
        parameters[ii] = param_target[ii] * (float(param_ranges.iloc[ii, 1]) - float(param_ranges.iloc[ii, 0])) + float(param_ranges.iloc[ii, 0])

    prerun_file, run_file = lis_template.write_template(run_id, start, end, start, 
                                                        end, cfg, out_dir, subcatch.path_station, parameters)

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
    """
    Generate a benchmark run for the model with specified parameters.

    Parameters
    ----------
    cfg : ConfigCalibration
        A global configuration settings object.
    subcatch : Subcatchment
        Subcatchment information and data.
    lis_template : LisfloodTemplate
        Template object for LISFLOOD input settings file.
    param_target : array-like
        Target parameter values for the benchmark.
    outfile : str
        Path for the output file.
    start : str
        Start date for the benchmark period.
    end : str
        End date for the benchmark period.
    """

    run_id = 'Z'
    out_dir = os.path.join(subcatch.path_out, run_id)
    os.makedirs(out_dir, exist_ok=True)

    param_ranges = cfg.param_ranges
    parameters = [None] * len(param_ranges)
    for ii in range(len(param_ranges)):
        parameters[ii] = param_target[ii] * (float(param_ranges.iloc[ii, 1]) - float(param_ranges.iloc[ii, 0])) + float(param_ranges.iloc[ii, 0])

    prerun_file, run_file = lis_template.write_template(run_id, start, end, start, 
                                                        end, cfg, out_dir, subcatch.path_station, parameters)

    lisf1.main(prerun_file, '-v')
    lisf1.main(run_file, '-q')

    # Outputing synthetic observed discharge
    print( ">> Saving simulated streamflow with default parameters in {}".format(outfile))
    Qsim_tss=LisSettings.instance().binding['DisTS']
    simulated_streamflow = utils.read_tss(Qsim_tss)
    simulated_streamflow[1][simulated_streamflow[1] == 1e31] = np.nan
    Qsim = simulated_streamflow[1].values
    freq = '{}min'.format(cfg.timestep)
    index = pd.to_datetime(pd.date_range(start, end, freq=freq), format='%d/%m/%Y %H:%M', errors='raise')
    Qsim = pd.DataFrame(data=Qsim, index=index)
    Qsim.columns = [str(subcatch.obsid)]
    Qsim.index.name = 'Timestamp'
    Qsim.to_csv(outfile, ',', date_format='%d/%m/%Y %H:%M')

    # required for downstream catchments
    simulated_best_tss2csv(cfg, subcatch, run_id, start, 'dis', 'streamflow')
    simulated_best_tss2csv(cfg, subcatch, run_id, start, 'chanq', 'chanq')
