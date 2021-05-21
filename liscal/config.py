import pandas
from datetime import datetime
from configparser import ConfigParser
from liscal import pcr_utils, calibration


class Config():

    def __init__(self, settings_file):

        parser = ConfigParser()
        parser.read(settings_file)

        # paths
        self.path_result = parser.get('Path', 'Result')
        self.subcatchment_path = parser.get('Path','SubCatchmentPath')

        pcraster_path = parser.get('Path', 'PCRHOME')
        self.pcraster_cmd = {}
        for execname in ["pcrcalc", "map2asc", "asc2map", "col2map", "map2col", "mapattr", "resample", "readmap"]:
            self.pcraster_cmd[execname] = pcr_utils.getPCrasterPath(pcraster_path, settings_file, alias=execname)

        # deap
        self.deap_param = calibration.DEAPParameters(parser)
        # Load param ranges file
        self.param_ranges = pandas.read_csv(parser.get('Path','ParamRanges'), sep=",", index_col=0)

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
