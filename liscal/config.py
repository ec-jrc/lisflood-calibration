import os
import pandas
from datetime import datetime
from configparser import ConfigParser
from liscal import pcr_utils, calibration


class DEAPParameters():

    def __init__(self, parser):
        self.num_cpus = int(parser.get('DEAP','numCPUs'))
        self.min_gen = int(parser.get('DEAP','minGen'))
        self.max_gen = int(parser.get('DEAP','maxGen'))
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
        self.path_result = parser.get('Path', 'Result')
        self.subcatchment_path = parser.get('Path','SubCatchmentPath')

        self.pcraster_cmd = {}
        for execname in ["pcrcalc", "map2asc", "asc2map", "col2map", "map2col", "mapattr", "resample", "readmap"]:
            self.pcraster_cmd[execname] = execname

        # deap
        self.param_ranges = pandas.read_csv(parser.get('Path','ParamRanges'), sep=",", index_col=0)
        self.deap_param = DEAPParameters(parser)
        # Load param ranges file

        # template
        self.lisflood_template = parser.get('Templates','LISFLOODSettings')

        # Debug/test parameters
        self.fast_debug = bool(int(parser.get('MAIN', 'fastDebug')))

        # Date parameters
        self.ObservationsStart = datetime.strptime(parser.get('MAIN', 'ObservationsStart'), "%d/%m/%Y %H:%M")  # Start of forcing
        self.ObservationsEnd = datetime.strptime(parser.get('MAIN', 'ObservationsEnd'), "%d/%m/%Y %H:%M")  # Start of forcing
        self.forcing_start = datetime.strptime(parser.get('MAIN','ForcingStart'),"%d/%m/%Y %H:%M")  # Start of forcing
        self.forcing_end = datetime.strptime(parser.get('MAIN','ForcingEnd'),"%d/%m/%Y %H:%M")  # Start of forcing
        self.WarmupDays = int(parser.get('MAIN', 'WarmupDays'))
        self.calibration_freq = parser.get('MAIN', 'calibrationFreq')

        # observations
        self.Qtss_csv = parser.get('CSV', 'Qtss')
        self.Qmeta_csv = parser.get('CSV', 'Qmeta')
