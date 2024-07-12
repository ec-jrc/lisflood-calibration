import pytest
import os
import shutil
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
        self.min_gen = 1
        self.max_gen = 1
        self.pop = 2
        self.mu = 2
        self.lambda_ = 2

        self.cxpb = 0.6
        self.mutpb = 0.4


class DummyConfig():

    def __init__(self):

        self.num_cpus = 1
        
        # paths
        self.path_out = path.join(OUT_DIR)
        self.subcatchment_path = path.join(DATA_DIR)
        self.path_subcatch = path.join(self.subcatchment_path, '380')
        self.stations_links = path.join(DATA_DIR, 'stations_links.csv')

        pcraster_path = ''
        self.pcraster_cmd = {}
        for execname in ["pcrcalc", "map2asc", "asc2map", "col2map", "map2col", "mapattr", "resample", "readmap"]:
            self.pcraster_cmd[execname] = execname

        # deap
        self.deap_param = DummyDEAPParameters()

        # Load param ranges file
        param_ranges_file = path.join(DATA_DIR, 'param_ranges.csv')
        self.param_ranges = pandas.read_csv(param_ranges_file, sep=",", index_col=0)

        # template
        self.lisflood_template = path.join(ROOT_DIR, 'templates', 'settings_lisflood.xml')

        # Debug/test parameters
        self.fast_debug = False

        # Date params
        self.forcing_start = datetime.strptime('31/12/2016 06:00', "%d/%m/%Y %H:%M")
        self.forcing_end = datetime.strptime('31/12/2017 06:00', "%d/%m/%Y %H:%M")
        self.timestep = 360


@pytest.fixture
def dummy_cfg():
    return DummyConfig()


@pytest.fixture(autouse=True)
def run_around_tests():
    print('Creating output directory')
    os.makedirs(OUT_DIR, exist_ok=True)

    # running test here
    yield

    print('Removing output directory')
    shutil.rmtree(OUT_DIR)
