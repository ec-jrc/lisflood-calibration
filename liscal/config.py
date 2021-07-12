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
        self.parser = ConfigParser()
        if os.path.isfile(settings_file):
            self.parser.read(settings_file)
        else:
            raise FileNotFoundError('Incorrect path to setting file: {}'.format(settings_file))

        print('Settings:')
        for section in self.parser.sections():
            print('- {}'.format(section))
            for key, value in dict(self.parser[section]).items():
                print('  - {}: {}'.format(key, value)) 

        # paths
        self.subcatchment_path = self.parser.get('Path','subcatchment_path')

        # Date parameters
        self.observations_start = datetime.strptime(self.parser.get('Main', 'observations_start'), "%d/%m/%Y %H:%M")  # Start of forcing
        self.observations_end = datetime.strptime(self.parser.get('Main', 'observations_end'), "%d/%m/%Y %H:%M")  # Start of forcing
        self.forcing_start = datetime.strptime(self.parser.get('Main','forcing_start'),"%d/%m/%Y %H:%M")  # Start of forcing
        self.forcing_end = datetime.strptime(self.parser.get('Main','forcing_end'),"%d/%m/%Y %H:%M")  # Start of forcing
        self.spinup_days = int(self.parser.get('Main', 'spinup_days'))
        self.calibration_freq = self.parser.get('Main', 'calibration_freq')

        # observations
        self.observed_discharges = self.parser.get('Stations', 'observed_discharges')
        self.stations_data = self.parser.get('Stations', 'stations_data')


class ConfigCalibration(Config):

    def __init__(self, settings_file):
        super().__init__(settings_file)

        self.pcraster_cmd = {}
        for execname in ["pcrcalc", "map2asc", "asc2map", "col2map", "map2col", "mapattr", "resample", "readmap"]:
            self.pcraster_cmd[execname] = execname

        # deap
        self.param_ranges = pandas.read_csv(self.parser.get('Path','param_ranges'), sep=",", index_col=0)
        self.deap_param = DEAPParameters(self.parser)
        # Load param ranges file

        # template
        self.lisflood_template = self.parser.get('Templates','LISFLOODSettings')

        # Debug/test parameters
        self.fast_debug = bool(int(self.parser.get('Main', 'fast_debug')))

        # stations
        self.stations_links = self.parser.get('Stations', 'stations_links')


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


class ConfigPostProcessing(Config):

    def __init__(self, settings_file):
        super().__init__(settings_file)

        self.summary_path = self.parser.get('Path','summary_path')

        # parse validation dates as string and make sure format is correct
        self.validation_start = datetime.strptime(self.parser.get('Main','validation_start'),"%d/%m/%Y %H:%M").strftime('%d/%m/%Y %H:%M')
        self.validation_end = datetime.strptime(self.parser.get('Main','validation_end'),"%d/%m/%Y %H:%M").strftime('%d/%m/%Y %H:%M')

        # we don't use it but required for objectives object
        self.param_ranges = None

        # stations
        self.return_periods = self.parser.get('Stations', 'return_periods')

        # plot parameters
        self.plot_params = PlotParameters()
