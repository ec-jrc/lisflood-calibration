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
        self.fast_debug = bool(int(parser.get('DEFAULT', 'fastDebug')))

        # Date parameters
        self.ObservationsStart = datetime.strptime(parser.get('DEFAULT', 'ObservationsStart'), "%d/%m/%Y %H:%M")  # Start of forcing
        self.ObservationsEnd = datetime.strptime(parser.get('DEFAULT', 'ObservationsEnd'), "%d/%m/%Y %H:%M")  # Start of forcing
        self.forcing_start = datetime.strptime(parser.get('DEFAULT','ForcingStart'),"%d/%m/%Y %H:%M")  # Start of forcing
        self.forcing_end = datetime.strptime(parser.get('DEFAULT','ForcingEnd'),"%d/%m/%Y %H:%M")  # Start of forcing
        self.WarmupDays = int(parser.get('DEFAULT', 'WarmupDays'))
        self.calibration_freq = parser.get('DEFAULT', 'calibrationFreq')

        # observations
        self.Qtss_csv = parser.get('CSV', 'Qtss')
        self.Qmeta_csv = parser.get('CSV', 'Qmeta')
