import os
import pandas
from datetime import datetime
from configparser import ConfigParser
from liscal import pcr_utils, calibration


class DEAPParameters():

    def __init__(self, parser):
        self.num_cpus = int(parser.get('DEAP','numCPUs'))
        self.min_gen = int(parser.get('DEAP','min_gen'))
        self.max_gen = int(parser.get('DEAP','max_gen'))
        self.pop = int(parser.get('DEAP','pop'))
        self.mu = int(parser.get('DEAP','mu'))
        self.lambda_ = int(parser.get('DEAP','lambda_'))
        self.cxpb = 0.6
        self.mutpb = 0.4


class Config():

    def __init__(self, settings_file):

        parser = ConfigParser()
        if os.path.isfile(settings_file):
            parser.read(settings_file)
        else:
            raise FileNotFoundError('Incorrect path to setting file: {}'.format(settings_file))

        print('Calibration settings:')
        for section in parser.sections():
            print('- {}'.format(section))
            for key, value in dict(parser[section]).items():
                print('  - {}: {}'.format(key, value)) 

        # paths
        self.path_result = parser.get('Path', 'result')
        self.subcatchment_path = parser.get('Path','subcatchment_path')

        self.pcraster_cmd = {}
        for execname in ["pcrcalc", "map2asc", "asc2map", "col2map", "map2col", "mapattr", "resample", "readmap"]:
            self.pcraster_cmd[execname] = execname

        # deap
        self.param_ranges = pandas.read_csv(parser.get('Path','param_ranges'), sep=",", index_col=0)
        self.deap_param = DEAPParameters(parser)
        # Load param ranges file

        # template
        self.lisflood_template = parser.get('Templates','LISFLOODSettings')

        # Debug/test parameters
        self.fast_debug = bool(int(parser.get('MAIN', 'fast_debug')))

        # Date parameters
        self.observations_start = datetime.strptime(parser.get('MAIN', 'observations_start'), "%d/%m/%Y %H:%M")  # Start of forcing
        self.observations_end = datetime.strptime(parser.get('MAIN', 'observations_end'), "%d/%m/%Y %H:%M")  # Start of forcing
        self.forcing_start = datetime.strptime(parser.get('MAIN','forcing_start'),"%d/%m/%Y %H:%M")  # Start of forcing
        self.forcing_end = datetime.strptime(parser.get('MAIN','forcing_end'),"%d/%m/%Y %H:%M")  # Start of forcing
        self.spinup_days = int(parser.get('MAIN', 'spinup_days'))
        self.calibration_freq = parser.get('MAIN', 'calibration_freq')

        # observations
        self.observed_discharges = parser.get('CSV', 'observed_discharges')
        self.stations_data = parser.get('CSV', 'stations_data')
        self.direct_links = parser.get('CSV', 'direct_links')