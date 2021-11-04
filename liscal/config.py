import os
import pandas
from datetime import datetime
from configparser import ConfigParser
from liscal import pcr_utils, calibration


class Config():

    def __init__(self, settings_file):
        self.parser = ConfigParser()
        if os.path.isfile(settings_file):
            self.parser.read(settings_file)
        else:
            raise FileNotFoundError('Incorrect path to setting file: {}'.format(settings_file))

    def info(self):
        print('Settings:')
        for section in self.parser.sections():
            print('- {}'.format(section))
            for key, value in dict(self.parser[section]).items():
                print('  - {}: {}'.format(key, value)) 


class DEAPParameters():

    def __init__(self, parser):
        self.min_gen = int(parser.get('DEAP','min_gen'))
        self.max_gen = int(parser.get('DEAP','max_gen'))
        self.pop = int(parser.get('DEAP','pop'))
        self.mu = int(parser.get('DEAP','mu'))
        self.lambda_ = int(parser.get('DEAP','lambda_'))
        self.cxpb = 0.6
        self.mutpb = 0.4


class ConfigCalibration(Config):

    def __init__(self, settings_file, n_cpus=1):
        super().__init__(settings_file)

        self.num_cpus = int(n_cpus)

        # paths
        self.subcatchment_path = self.parser.get('Path','subcatchment_path')

        # Date parameters
        self.forcing_start = datetime.strptime(self.parser.get('Main','forcing_start'),"%d/%m/%Y %H:%M")  # Start of forcing
        self.forcing_end = datetime.strptime(self.parser.get('Main','forcing_end'),"%d/%m/%Y %H:%M")  # end of forcing
        self.timestep = int(self.parser.get('Main', 'timestep'))  # in minutes
        if self.timestep != 360 and self.timestep != 1440:
            raise Exception('Calibration timestep {} not supported'.format(self.timestep))


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
