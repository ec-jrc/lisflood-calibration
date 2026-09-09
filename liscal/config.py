import os
import numpy as np
import pandas
from datetime import datetime
from configparser import ConfigParser, NoOptionError

class Config():
    """
    A class to handle the configuration settings from a settings file.

    Parameters
    ----------
    settings_file : str
        Path to the settings file.
    print_settings : bool, optional
        Flag to print the settings after reading the file (default is True).

    Raises
    ------
    FileNotFoundError
        If the specified settings file does not exist.
    """

    def __init__(self, settings_file, print_settings=True):
        self.parser = ConfigParser()
        if os.path.isfile(settings_file):
            self.parser.read(settings_file)
        else:
            raise FileNotFoundError('Incorrect path to setting file: {}'.format(settings_file))

        if print_settings:
            print('Settings:')
            for section in self.parser.sections():
                print('- {}'.format(section))
                for key, value in dict(self.parser[section]).items():
                    print('  - {}: {}'.format(key, value)) 


class DEAPParameters():
    """
    A class to store DEAP algorithm parameters.

    Parameters
    ----------
    parser : ConfigParser
        ConfigParser object to extract DEAP parameters.

    Attributes
    ----------
    min_gen : int
        Minimum number of generations.
    max_gen : int
        Maximum number of generations.
    pop : int
        Population size.
    mu : int
        The number of individuals to select for the next generation.
    lambda_ : int
        The number of children to produce at each generation.
    cxpb : float
        Crossover probability.
    mutpb : float
        Mutation probability.
    gen_offset : int
        Generation offset.
    effmax_tol : float
        Tolerance for maximum efficiency.
    """

    def __init__(self, parser):
        self.min_gen = int(parser.get('DEAP','min_gen'))
        self.max_gen = int(parser.get('DEAP','max_gen'))
        self.pop = int(parser.get('DEAP','pop'))
        self.mu = int(parser.get('DEAP','mu'))
        self.lambda_ = int(parser.get('DEAP','lambda_'))
        self.elite = int(parser.get('DEAP','elite', fallback=0)) # usually take 10% of mu as elites to keep in new population.
        self.cxpb = 0.6
        self.mutpb = 0.4
        self.gen_offset = int(parser.get('DEAP','gen_offset'))
        self.effmax_tol = float(parser.get('DEAP','effmax_tol'))
        self.split_lake_params = bool(int(parser.get('DEAP','split_lake_params', fallback=0)))
        self.apply_statistical_stall_check = bool(int(parser.get('DEAP','apply_statistical_stall_check', fallback=0)))
        self.use_filtered_population  = bool(int(parser.get('DEAP','use_filtered_population', fallback=0)))

        self.stop_on_low_kgejsd = int(parser.get('DEAP','stop_on_low_kgejsd', fallback=0))

        try:
            objectives_str = parser.get('DEAP', 'objectives')
            self.objectives_list = [obj.strip().upper() for obj in objectives_str.split(',')]
        except NoOptionError:
            self.objectives_list =['KGE']

        # check for valid objectives
        valid_objectives = {'KGE', 'CORR', 'BIAS', 'Y', 'SAE', 'JSD', 'KGE_JSD'}
        
        # Check for any unknown objectives
        unknown_objectives = [obj for obj in self.objectives_list if obj not in valid_objectives]
        if unknown_objectives:
            raise ValueError(f"Unknown objectives found: {', '.join(unknown_objectives)}")

        # Ensure at least "KGE" or "KGE_JSD" or all three ["CORR", "BIAS", "Y"] are present
        has_kge = 'KGE' in self.objectives_list
        has_kge_jsd = 'KGE_JSD' in self.objectives_list
        has_kge_terms = all(obj in self.objectives_list for obj in ['CORR', 'BIAS', 'Y'])
        
        if not (has_kge or has_kge_jsd or has_kge_terms):
            raise ValueError("At least 'KGE', 'KGE_JSD', or all of ['CORR', 'BIAS', 'Y'] must be included in objectives.")


class ConfigCalibration(Config):
    """
    A class for configuration and calibration settings specific to hydrological modeling.

    Extends the Config class with additional parameters and validation specific to hydrological model calibration.

    Parameters
    ----------
    settings_file : str
        Path to the settings file.
    n_cpus : int, optional
        Number of CPUs to use (default is 1).
    seed : int or None, optional
        Seed for random number generation (default is None).

    Attributes
    ----------
    num_cpus : int
        Number of CPUs to use.
    seed : int or None
        Seed for random number generation.
    subcatchment_path : str
        File path to the subcatchment data.
    forcing_start : datetime
        Start time of forcing data.
    forcing_end : datetime
        End time of forcing data.
    timestep : int
        Time step in minutes.
    prerun_start : datetime
        Start time of pre-run period in calibration
    prerun_end : datetime
        End time of pre-run period in calibration
    longterm_prerun_start : datetime
        Optional: Start time of pre-run period in the long term run (by Default will take the forcing start date)
    longterm_prerun_end : datetime
        Optional: End time of pre-run period in the long term run (by Default will take the forcing end date)
    prerun_timestep : int
        Pre-run time step in minutes.
    deap_param : DEAPParameters
        DEAP algorithm parameters.
    param_ranges : DataFrame
        Parameter ranges for calibration.
    lisflood_template : str
        Path to the LISFLOOD settings template.
    fast_debug : bool
        Flag to enable fast debugging mode.
    stations_links : str
        File path to the stations links data.
    pcraster_cmd : dict
        Commands for PCRaster processing.

    Raises
    ------
    Exception
        If the provided timestep or prerun timestep is not supported.
    """

    def __init__(self, settings_file, n_cpus=1, seed=None):
        super().__init__(settings_file)

        self.num_cpus = int(n_cpus)
        self.seed = seed

        # paths
        self.subcatchment_path = self.parser.get('Path','subcatchment_path')

        # Date parameters
        self.forcing_start = datetime.strptime(self.parser.get('Main','forcing_start'),"%d/%m/%Y %H:%M")  # Start of forcing
        self.forcing_end = datetime.strptime(self.parser.get('Main','forcing_end'),"%d/%m/%Y %H:%M")  # end of forcing
        self.timestep = int(self.parser.get('Main', 'timestep'))  # in minutes
        if self.timestep != 360 and self.timestep != 1440:
            raise Exception('Calibration timestep {} not supported'.format(self.timestep))

        self.prerun_start = datetime.strptime(self.parser.get('Main','prerun_start'),"%d/%m/%Y %H:%M")  # Start of prerun in calibration
        self.prerun_end = datetime.strptime(self.parser.get('Main','prerun_end'),"%d/%m/%Y %H:%M")  # end of prerun in calibration
        
        # Optional: specify longterm run prerun start and end
        # if not set, take the full forcing period
        self.longterm_prerun_start = self.parser.get('Main','longterm_prerun_start',fallback=None)  # Start of prerun in calibration
        self.longterm_prerun_end = self.parser.get('Main','longterm_prerun_end',fallback=None)  # end of prerun in calibration
        self.longterm_prerun_start = (self.forcing_start if self.longterm_prerun_start is None else datetime.strptime(self.longterm_prerun_start,"%d/%m/%Y %H:%M"))
        self.longterm_prerun_end = (self.forcing_end if self.longterm_prerun_end is None else datetime.strptime(self.longterm_prerun_end,"%d/%m/%Y %H:%M"))

        self.prerun_timestep = int(self.parser.get('Main', 'prerun_timestep'))  # in minutes
        if self.prerun_timestep != 360 and self.prerun_timestep != 1440:
            raise Exception('Pre-run timestep {} not supported'.format(self.prerun_timestep))
        
        self.num_max_calib_years = int(self.parser.get('Main', 'num_max_calib_years', fallback=20))  # max calibration years, used to compute split date

        # deap
        self.deap_param = DEAPParameters(self.parser)

        # Load param ranges file
        self.param_ranges = pandas.read_csv(self.parser.get('Path','param_ranges'), sep=",", index_col=0)

        # template
        self.lisflood_template = self.parser.get('Templates','LISFLOODSettings')

        # Debug/test parameters
        self.fast_debug = bool(int(self.parser.get('Main', 'fast_debug')))
        if self.fast_debug:
            # Turn this on for debugging faster. You can speed up further by setting maxGen = 1
            self.deap_param.lambda_ = 2
            self.deap_param.mu = 2
            self.deap_param.pop = 2

        # stations
        self.stations_links = self.parser.get('Stations', 'stations_links')

        # observations
        self.observed_discharges = self.parser.get('Stations', 'observed_discharges')
        self.stations_data = self.parser.get('Stations', 'stations_data')

        # Reservoir creation/demolition dates
        self.reservoir_events = self.parser.get('Stations', 'reservoir_events', fallback=None)

        # flag to enable aridity index check 
        self.use_aridity_index_check = bool(int(self.parser.get('Main', 'use_aridity_index_check', fallback=0)))


        # pcraster commands
        self.pcraster_cmd = {}
        for execname in ["pcrcalc", "map2asc", "asc2map", "col2map", "map2col", "mapattr", "resample", "readmap"]:
            self.pcraster_cmd[execname] = execname

    def filter_param_ranges_after_init(self, model_initialized, split_lake_params):
        """
        Filter parameter ranges based on the initialized LISFLOOD model state.

        Moved to hydro_model.filter_param_ranges() but kept here as a delegate
        for backward compatibility with CAL_6 and CAL_7 scripts.

        Parameters
        ----------
        model_initialized : HydrologicalModel
            Initialized model with loaded LISFLOOD settings.
        split_lake_params : bool
            Whether to create per-lake LakeMultiplier parameters.
        """
        from liscal.hydro_model import filter_param_ranges
        filter_param_ranges(self, model_initialized, split_lake_params)

class PlotParameters():

    title_size_big = 32
    title_size_small = 18
    label_size = 30
    axes_size = 24
    legend_size_small = 16
    threshold_size = 24

    file_format = 'svg'

    text = {
        'figure': {'autolayout': True},
        'font': {
            'size': 14,
            'family':'sans-serif',
            'sans-serif':['Arial'],
            'weight': 'bold'
        },
        'text': {'usetex': True},
        'axes': {'labelweight': 'bold'},
    }


class ConfigPostProcessing(ConfigCalibration):

    def __init__(self, settings_file):
        super().__init__(settings_file)

        # paths
        self.summary_path = self.parser.get('Path','summary_path')

        # # Date parameters
        # self.forcing_start = datetime.strptime(self.parser.get('Main','forcing_start'),"%d/%m/%Y %H:%M")
        # self.forcing_end = datetime.strptime(self.parser.get('Main','forcing_end'),"%d/%m/%Y %H:%M")
        # self.timestep = int(self.parser.get('Main', 'timestep'))  # in minutes
        # if self.timestep != 360 and self.timestep != 1440:
        #     raise Exception('Calibration timestep {} not supported'.format(self.timestep))

        # # we don't use it but required for objectives object
        # self.param_ranges = None

        # # stations
        # self.stations_data = self.parser.get('Stations', 'stations_data')

        # plot parameters
        self.plot_params = PlotParameters()