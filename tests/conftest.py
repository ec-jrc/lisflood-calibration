import pytest
import pandas
import numpy as np
from datetime import datetime
from os import path

from liscal import utils

ROOT_DIR = path.join(path.dirname(path.realpath(__file__)), '..')
TEST_DIR = path.join(ROOT_DIR, 'tests')
DATA_DIR = path.join(TEST_DIR, 'data')
OUT_DIR = path.join(TEST_DIR, 'outputs')


class DummyDEAPParameters():

    def __init__(self):
        self.use_multiprocessing = 1
        self.num_cpus = 2
        self.min_gen = 1
        self.max_gen = 1
        self.pop = 2
        self.mu = 2
        self.lambda_ = 2

        self.cxpb = 0.6
        self.mutpb = 0.4


class DummyConfig():

    def __init__(self):

        # paths
        self.path_result = OUT_DIR
        self.subcatchment_path = path.join(DATA_DIR)
        self.path_subcatch = path.join(self.subcatchment_path, '380')

        pcraster_path = ''
        self.pcraster_cmd = {}
        for execname in ["pcrcalc", "map2asc", "asc2map", "col2map", "map2col", "mapattr", "resample", "readmap"]:
            self.pcraster_cmd[execname] = execname

        # deap
        self.deap_param = DummyDEAPParameters()

        # Load param ranges file
        param_ranges_file = path.join(DATA_DIR, 'ParamRanges_LISFLOOD.csv')
        self.param_ranges = pandas.read_csv(param_ranges_file, sep=",", index_col=0)

        # template
        self.lisflood_template = path.join(ROOT_DIR, 'templates')

        # Debug/test parameters
        self.fast_debug = False

        # Date parametersObservationsStart = 1/1/1990 00:00

        self.forcing_start = datetime.strptime('2/1/1990 06:00', "%d/%m/%Y %H:%M")  # Start of forcing
        self.forcing_end = datetime.strptime('31/12/2017 06:00', "%d/%m/%Y %H:%M")  # Start of forcing
        self.WarmupDays = 30
        self.calibration_freq = '6-hourly'


@pytest.fixture
def dummy_cfg():
    return DummyConfig()


class DummyModel():

    def __init__(self, dummy_cfg):

        self.observations = 0.1*np.ones(len(dummy_cfg.param_ranges))

    def init_run(self):
        return

    def run(self, Individual):

        error = np.sqrt(np.mean((Individual - self.observations)**2))

        return error


@pytest.fixture
def dummy_model(dummy_cfg):
    return DummyModel(dummy_cfg)


@pytest.fixture(autouse=True)
def run_around_tests():
    print('Creating output directory')
    ret, out = utils.run_cmd("mkdir -p {}".format(OUT_DIR))
    assert out == ''
    print(out)

    # running test here
    yield

    # print('Removing output directory')
    # ret, out = utils.run_cmd('rm -rf {}'.format(OUT_DIR))
    # assert out == ''
    # print(out)
