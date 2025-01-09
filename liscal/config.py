import os
import pandas
from datetime import datetime
from configparser import ConfigParser
from liscal import pcr_utils, calibration


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
        self.cxpb = 0.6
        self.mutpb = 0.4
        self.gen_offset = int(parser.get('DEAP','gen_offset'))
        self.effmax_tol = float(parser.get('DEAP','effmax_tol'))
        self.apply_statistical_stall_check = bool(parser.get('DEAP','apply_statistical_stall_check'))


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
        Start time of pre-run period.
    prerun_end : datetime
        End time of pre-run period.
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

        self.prerun_start = datetime.strptime(self.parser.get('Main','prerun_start'),"%d/%m/%Y %H:%M")  # Start of forcing
        self.prerun_end = datetime.strptime(self.parser.get('Main','prerun_end'),"%d/%m/%Y %H:%M")  # end of forcing
        self.prerun_timestep = int(self.parser.get('Main', 'prerun_timestep'))  # in minutes
        if self.prerun_timestep != 360 and self.prerun_timestep != 1440:
            raise Exception('Pre-run timestep {} not supported'.format(self.prerun_timestep))

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
        
        # pcraster commands
        self.pcraster_cmd = {}
        for execname in ["pcrcalc", "map2asc", "asc2map", "col2map", "map2col", "mapattr", "resample", "readmap"]:
            self.pcraster_cmd[execname] = execname
